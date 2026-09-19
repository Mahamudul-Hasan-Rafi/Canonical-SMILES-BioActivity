"""
V3 model + training loop, generalised over the SMILES backbone.

Faithful to `bioactivity_dl 8_2_C_v3_re.ipynb` (FineTunedBERTaECFP_v3, FocalLoss,
LLRD, warmup-cosine, WeightedRandomSampler, SMILES augmentation, early stopping
on val AUROC). With backbone="molformer" the parameter initialisation, forward
pass and (without augmentation) the whole training run are bit-identical to
`v3_core.py`, which produced the reproduction/ablation results
(tests/test_core_parity.py). One deliberate difference: SMILES augmentation is
now seeded (RDKit's doRandom ignores every seed), so runs are fully reproducible.

Speed-ups that do not change the maths: dynamic padding, MoLFormer sync-free
attention, fused AdamW, TF32, best checkpoint kept in RAM.
"""
from __future__ import annotations

import contextlib
import math
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import backbones
from data import ECFP_BITS, MACCS_BITS, N_DESC

D = 768
MODALITIES = ("smiles", "ecfp", "maccs", "desc")
MODALITIES_ALL = MODALITIES + ("graph",)       # "graph" only with the V4-C D-MPNN branch
# V4 parameters trained with base_lr * new_lr_mult (modules with no V3 counterpart)
NEW_PREFIXES = ("graph_enc.", "graph_proj.", "tok_", "fp_emb.", "fp_cnt.")

HP_SETS = {
    # optuna_results_v3__.json (the notebook's tuned V3)
    "tuned": {"num_heads": 4, "hidden_dim": 256, "dropout": 0.1, "num_classifier_layers": 2,
              "n_cross_layers": 2, "batch_size": 32, "learning_rate": 9.8916998354799e-06,
              "weight_decay": 2.1247834038360546e-05},
    # notebook cell 15 "manual fallback if HPO hasn't been run" (pre-tuning config)
    "untuned": {"num_heads": 8, "hidden_dim": 512, "dropout": 0.2, "num_classifier_layers": 3,
                "n_cross_layers": 3, "batch_size": 32, "learning_rate": 5e-6, "weight_decay": 1e-4},
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset with dynamic padding
# ─────────────────────────────────────────────────────────────────────────────
def random_smiles(smi):
    """Random non-canonical SMILES (same algorithm as MolToSmiles(doRandom=True), which the
    notebook used), but seeded from Python's `random`, so augmentation is reproducible.
    RDKit's doRandom draws from an internal generator that no seed call resets."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return smi
    return Chem.MolToRandomSmilesVect(mol, 1, randomSeed=random.randrange(2 ** 31))[0]


class V3Dataset(torch.utils.data.Dataset):
    def __init__(self, feats, idx, tokenizer, scaler, augment=False, max_len=512, fp_key="ecfp", graphs=False,
                 pic50=False):
        self.tok = tokenizer
        self.smiles = [feats["smiles"][i] for i in idx]
        self.labels = feats["labels"][idx].astype(np.float32)
        self.ecfp = feats[fp_key][idx].astype(np.float32)
        self.graphs = [feats["graphs"][i] for i in idx] if graphs else None
        self.pic50 = feats["pic50"][idx].astype(np.float32) if pic50 else None
        self.maccs = feats["maccs"][idx].astype(np.float32)
        self.desc = scaler.transform(feats["desc_raw"][idx]).astype(np.float32)
        self.augment, self.max_len = augment, max_len
        enc = tokenizer(self.smiles, truncation=True, max_length=max_len)
        self.ids = [np.array(x, dtype=np.int64) for x in enc["input_ids"]]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        if self.augment and random.random() < 0.5:
            ids = np.array(self.tok(random_smiles(self.smiles[i]), truncation=True,
                                    max_length=self.max_len)["input_ids"], dtype=np.int64)
        else:
            ids = self.ids[i]
        item = (ids, self.ecfp[i], self.maccs[i], self.desc[i], self.labels[i])
        if self.graphs is not None:
            item = item + (self.graphs[i],)
        return item + (self.pic50[i],) if self.pic50 is not None else item


def make_collate(pad_id, fixed_len=None, fp_tokens=False, graphs=False, pic50=False):
    def collate(batch):
        L = fixed_len or max(len(b[0]) for b in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for j, b in enumerate(batch):
            ids[j, :len(b[0])] = torch.from_numpy(b[0])
            mask[j, :len(b[0])] = 1
        out = {
            "input_ids": ids, "attention_mask": mask,
            "ecfp": torch.from_numpy(np.stack([b[1] for b in batch])),
            "maccs": torch.from_numpy(np.stack([b[2] for b in batch])),
            "descriptors": torch.from_numpy(np.stack([b[3] for b in batch])),
            "labels": torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float),
        }
        if fp_tokens:     # V4-B: non-zero fingerprint bits as a substructure token set
            import v4_modules
            out.update(v4_modules.collate_fp_tokens([b[1] for b in batch]))
        if graphs:        # V4-C: molecular graphs for the D-MPNN branch
            import v4_modules
            out.update(v4_modules.collate_graphs([b[5] for b in batch]))
        if pic50:         # V5: measured potency as an auxiliary / regression target (always last)
            out["pic50"] = torch.tensor(np.array([b[-1] for b in batch]), dtype=torch.float)
        return out
    return collate


# ─────────────────────────────────────────────────────────────────────────────
# MoLFormer speed patch (removes 24 GPU->CPU syncs per forward pass)
# ─────────────────────────────────────────────────────────────────────────────
def _fast_feature_map_forward(self, query, key):
    """MolformerFeatureMap.forward with the orthogonal random features drawn and
    QR-decomposed on the CPU (64x64, ~0.3 ms) and copied asynchronously instead of
    a blocking GPU QR in every layer. Same distribution, CPU random generator."""
    if not self.deterministic or self.training:
        q = self.query_size
        blocks = []
        for _ in range(math.ceil(self.num_components / q)):
            block = torch.randn(q, q)
            norms = torch.linalg.norm(block, dim=1).unsqueeze(0)
            Q, _ = torch.linalg.qr(block)
            blocks.append(Q * norms)
        w = torch.cat(blocks, dim=1)[:, : self.num_components]
        self.weight = w.pin_memory().to(query.device, non_blocking=True)
    query = torch.matmul(query, self.weight)
    key = torch.matmul(key, self.weight)
    return self.kernel(query), self.kernel(key)


def _make_fast_attn_forward(mod):
    apply_rotary_pos_emb = mod.apply_rotary_pos_emb

    def forward(self, hidden_states, attention_mask=None, position_ids=None, head_mask=None,
                output_attentions=False):
        """MolformerSelfAttention.forward minus the blocking `torch.equal` sanity check."""
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        kv_seq_len = key_layer.shape[-2]
        cos, sin = self.rotary_embeddings(value_layer, seq_len=kv_seq_len)
        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, position_ids)
        query_layer, key_layer = self.feature_map(query_layer, key_layer)
        if attention_mask is not None:
            attention_mask = (attention_mask == 0).to(attention_mask.dtype)
            per_query_attn = attention_mask[:, 0, -1]
            key_layer = key_layer * per_query_attn[:, None, -kv_seq_len:, None]
        key_value = torch.matmul(key_layer.transpose(-1, -2), value_layer)
        norm = torch.matmul(query_layer, key_layer.sum(dim=-2).unsqueeze(-1)).clamp(min=self.eps)
        context_layer = torch.matmul(query_layer, key_value) / norm
        if head_mask is not None:
            context_layer = context_layer * head_mask
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(*(context_layer.size()[:-2] + (self.all_head_size,)))
        return (context_layer,)
    return forward


def patch_molformer_fast(bert):
    import sys
    import types
    for layer in bert.encoder.layer:
        attn = layer.attention.self
        mod = sys.modules[type(attn).__module__]
        attn.forward = types.MethodType(_make_fast_attn_forward(mod), attn)
        attn.feature_map.forward = types.MethodType(_fast_feature_map_forward, attn.feature_map)
    return bert


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing

    def forward(self, logits, targets):
        y = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        pt = torch.exp(-bce)
        a_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (a_t * (1 - pt) ** self.gamma * bce).mean()


class V3(nn.Module):
    """FineTunedBERTaECFP_v3 with ablation switches and a pluggable SMILES encoder.
    Defaults (backbone='molformer', all switches on) == the notebook model."""

    def __init__(self, backbone="molformer", modalities=MODALITIES, fusion="xattn", use_type_emb=True,
                 use_gate=True, pooling="triple", n_pool_layers=4, freeze_bottom_n=None,
                 freeze_all_bert=False, num_heads=8, dropout=0.2, hidden_dim=512,
                 num_classifier_layers=4, n_cross_layers=3, fast_molformer=True,
                 ecfp_dim=ECFP_BITS, fp_dropout=0.0, graph=False, token_dim=256, reg_head=False, task="cls"):
        super().__init__()
        self.backbone = backbone
        self.modalities = tuple(m for m in MODALITIES if m in modalities) + (("graph",) if graph else ())
        self.fp_dropout = fp_dropout
        self.fusion, self.use_type_emb, self.use_gate = fusion, use_type_emb, use_gate
        self.pooling = pooling
        self.freeze_all_bert = freeze_all_bert
        n_mod = len(self.modalities)

        if "smiles" in self.modalities:
            self.bert = backbones.load_encoder(backbone, fast=fast_molformer)
            n_layers = len(self.bert.encoder.layer)
            h = self.bert.config.hidden_size
            # MoLFormer: freeze 4/12 (notebook). Others: same fraction (bottom third).
            if freeze_bottom_n is None:
                freeze_bottom_n = round(n_layers / 3)
            self.n_pool_layers = min(n_pool_layers, n_layers)
            n_freeze = n_layers if freeze_all_bert else freeze_bottom_n
            for layer in self.bert.encoder.layer[:n_freeze]:
                for p in layer.parameters():
                    p.requires_grad = False
            if freeze_all_bert:
                for p in self.bert.parameters():
                    p.requires_grad = False
            self.layer_weights = nn.Parameter(torch.zeros(self.n_pool_layers))
            self.token_attn = nn.Linear(h, 1)
            if h != D:     # e.g. ChemBERTa-77M (384) -> 768; absent for MoLFormer
                self.smiles_proj = nn.Linear(h, D)

        self.modal_type_embed = nn.Embedding(5 if graph else 4, D)
        nn.init.xavier_uniform_(self.modal_type_embed.weight)
        vec = fusion != "token"          # V3-style per-modality vectors (all V3 variants)

        if "ecfp" in self.modalities and vec:
            self.ecfp_proj = nn.Sequential(
                nn.Linear(ecfp_dim, 512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(dropout * 0.5),
                nn.Linear(512, D), nn.GELU(), nn.LayerNorm(D), nn.Dropout(dropout))
        if "maccs" in self.modalities and vec:
            self.maccs_proj = nn.Sequential(
                nn.Linear(MACCS_BITS, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(dropout * 0.5),
                nn.Linear(256, D), nn.GELU(), nn.LayerNorm(D), nn.Dropout(dropout))
        if "desc" in self.modalities and vec:
            self.desc_proj = nn.Sequential(
                nn.Linear(N_DESC, 64), nn.GELU(), nn.LayerNorm(64), nn.Dropout(dropout * 0.5),
                nn.Linear(64, D), nn.GELU(), nn.LayerNorm(D), nn.Dropout(dropout))

        if fusion == "xattn":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=D, nhead=num_heads, dim_feedforward=D * 4,
                dropout=dropout, batch_first=True, norm_first=True)
            self.cross_modal = nn.TransformerEncoder(enc_layer, num_layers=n_cross_layers)
        if vec:
            self.gate = nn.Sequential(nn.Linear(D, D), nn.LayerNorm(D), nn.Sigmoid())
            self.fusion_proj = nn.Linear(D * len(self.modalities), D)

        if graph:                         # V4-C: D-MPNN over the molecular graph -> 5th modality token
            import v4_modules
            self.graph_enc = v4_modules.DMPNN(hidden=300, depth=3, dropout=dropout)
            self.graph_proj = nn.Sequential(nn.Linear(300, D), nn.GELU(), nn.LayerNorm(D), nn.Dropout(dropout))

        if fusion == "token":             # V4-B: token-level fusion of SMILES tokens + substructure tokens
            dt = token_dim
            self.tok_smiles = nn.Linear(self.bert.config.hidden_size, dt)
            self.fp_emb = nn.Embedding(ecfp_dim, dt)
            self.fp_cnt = nn.Linear(1, dt)
            self.tok_maccs = nn.Linear(MACCS_BITS, dt)
            self.tok_desc = nn.Linear(N_DESC, dt)
            self.tok_type = nn.Embedding(5, dt)          # cls, smiles, substructure, maccs, desc
            self.tok_cls = nn.Parameter(torch.zeros(1, 1, dt))
            tl = nn.TransformerEncoderLayer(d_model=dt, nhead=num_heads, dim_feedforward=dt * 4,
                                            dropout=dropout, batch_first=True, norm_first=True)
            self.tok_encoder = nn.TransformerEncoder(tl, num_layers=n_cross_layers)
            self.tok_norm = nn.LayerNorm(dt)

        layers, in_dim = [], (token_dim if fusion == "token" else D)
        for i in range(num_classifier_layers):
            out_dim = hidden_dim // (2 ** i)
            layers += [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
                       nn.Dropout(dropout if i < num_classifier_layers - 1 else dropout * 0.5)]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.classifier = nn.Sequential(*layers)

        self.task = task
        if reg_head:                      # V5: pIC50 head on the fused representation
            self.reg_head = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1))
            self.register_buffer("pic_mu", torch.tensor(0.0))
            self.register_buffer("pic_sd", torch.tensor(1.0))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_all_bert and hasattr(self, "bert"):
            self.bert.eval()   # frozen feature extractor: no dropout
        return self

    # split into steps so explainability code can hook each stage
    def smiles_hidden(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        with torch.no_grad() if self.freeze_all_bert else contextlib.nullcontext():
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds,
                            output_hidden_states=True)
        selected = out.hidden_states[-self.n_pool_layers:]
        w = torch.softmax(self.layer_weights, dim=0)
        return sum(wi * h for wi, h in zip(w, selected))

    def pool(self, hidden, attention_mask):
        cls_pool = hidden[:, 0]
        if self.pooling == "cls":
            v = cls_pool
        else:
            m = attention_mask.unsqueeze(-1).to(hidden.dtype)
            mean_pool = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
            if self.pooling == "mean":
                v = mean_pool
            else:
                scores = self.token_attn(hidden).squeeze(-1).masked_fill(~attention_mask.bool(), float("-inf"))
                attn_pool = (hidden * torch.softmax(scores, dim=-1).unsqueeze(-1)).sum(1)
                v = (cls_pool + mean_pool + attn_pool) / 3
        return self.smiles_proj(v) if hasattr(self, "smiles_proj") else v

    def modality_tokens(self, input_ids, attention_mask, ecfp, maccs, descriptors, inputs_embeds=None,
                        graph_batch=None):
        """[B, n_mod, D] projected modality tokens, in MODALITIES_ALL order."""
        embs = []
        for m in self.modalities:
            if m == "smiles":
                embs.append(self.pool(self.smiles_hidden(input_ids, attention_mask, inputs_embeds), attention_mask))
            elif m == "ecfp":
                if self.fp_dropout:
                    ecfp = F.dropout(ecfp, self.fp_dropout, self.training)
                embs.append(self.ecfp_proj(ecfp))
            elif m == "graph":
                embs.append(self.graph_proj(self.graph_enc(**graph_batch, n_graphs=ecfp.size(0))))
            elif m == "maccs":
                embs.append(self.maccs_proj(maccs))
            else:
                embs.append(self.desc_proj(descriptors))
        return torch.stack(embs, dim=1)

    def fuse(self, tokens):
        """modality tokens [B, n_mod, D] -> logit [B]"""
        B = tokens.size(0)
        if self.fusion == "xattn":
            if self.use_type_emb:
                ids = [MODALITIES_ALL.index(m) for m in self.modalities]
                t = self.modal_type_embed(torch.tensor(ids, device=tokens.device))
                tokens = tokens + t.unsqueeze(0)
            tokens = self.cross_modal(tokens)
        if self.use_gate:
            tokens = self.gate(tokens) * tokens
        fused = self.fusion_proj(tokens.reshape(B, -1))
        logit = self.classifier(fused).squeeze(-1)
        if not hasattr(self, "reg_head"):
            return logit
        z = self.reg_head(fused).squeeze(-1)                 # standardised pIC50
        if self.task == "reg":        # score = predicted pIC50 relative to the class gap (5-6), 2 logits / log unit
            logit = 2.0 * (z.float() * self.pic_sd + self.pic_mu - 5.5)
        return (logit, z) if self._want_reg else logit

    def token_forward(self, input_ids, attention_mask, ecfp, maccs, descriptors, fp_ids, fp_cnt, fp_mask):
        """V4-B: one Transformer over [CLS] + SMILES tokens + substructure tokens + MACCS + descriptors."""
        B = input_ids.size(0)
        tt = self.tok_type.weight
        S = self.tok_smiles(self.smiles_hidden(input_ids, attention_mask)) + tt[1]
        if self.training and self.fp_dropout:          # substructure-token dropout
            fp_mask = fp_mask & (torch.rand(fp_mask.shape, device=fp_mask.device) >= self.fp_dropout)
        Fp = self.fp_emb(fp_ids) + self.fp_cnt(fp_cnt.unsqueeze(-1)) + tt[2]
        Mc = (self.tok_maccs(maccs) + tt[3]).unsqueeze(1)
        Ds = (self.tok_desc(descriptors) + tt[4]).unsqueeze(1)
        C = self.tok_cls.expand(B, -1, -1) + tt[0]
        X = torch.cat([C.float(), S.float(), Fp.float(), Mc.float(), Ds.float()], 1)
        one = torch.ones(B, 1, dtype=torch.bool, device=X.device)
        keep = torch.cat([one, attention_mask.bool(), fp_mask, one, one], 1)
        Z = self.tok_encoder(X, src_key_padding_mask=~keep)
        return self.classifier(self.tok_norm(Z[:, 0])).squeeze(-1)

    _want_reg = False

    def forward(self, input_ids, attention_mask, ecfp, maccs, descriptors, return_reg=False, **extra):
        self._want_reg = return_reg
        if self.fusion == "token":
            return self.token_forward(input_ids, attention_mask, ecfp, maccs, descriptors,
                                      extra["fp_ids"], extra["fp_cnt"], extra["fp_mask"])
        gb = {k: v for k, v in extra.items() if k.startswith("g_")} or None
        return self.fuse(self.modality_tokens(input_ids, attention_mask, ecfp, maccs, descriptors, graph_batch=gb))


def build_llrd_optimizer(model, base_lr, weight_decay, decay_factor=0.95, new_lr_mult=1.0):
    """Same grouping as the notebook's build_llrd_optimizer. V4 modules without a V3 counterpart
    (NEW_PREFIXES) get base_lr * new_lr_mult; for every V3 variant that set is empty."""
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    groups = []
    new = [(n, p) for n, p in model.named_parameters() if n.startswith(NEW_PREFIXES) and p.requires_grad]
    head = [(n, p) for n, p in model.named_parameters()
            if not n.startswith("bert.") and not n.startswith(NEW_PREFIXES) and p.requires_grad]
    groups += [
        {"params": [p for n, p in head if not any(nd in n for nd in no_decay)], "lr": base_lr, "weight_decay": weight_decay},
        {"params": [p for n, p in head if any(nd in n for nd in no_decay)], "lr": base_lr, "weight_decay": 0.0},
    ]
    if hasattr(model, "bert"):
        n_layers = len(model.bert.encoder.layer)
        for depth, layer in enumerate(reversed(list(model.bert.encoder.layer))):
            lr_l = base_lr * (decay_factor ** depth)
            lp = [(n, p) for n, p in layer.named_parameters() if p.requires_grad]
            if lp:
                groups += [
                    {"params": [p for n, p in lp if not any(nd in n for nd in no_decay)], "lr": lr_l, "weight_decay": weight_decay},
                    {"params": [p for n, p in lp if any(nd in n for nd in no_decay)], "lr": lr_l, "weight_decay": 0.0},
                ]
        ep = [(n, p) for n, p in model.bert.embeddings.named_parameters() if p.requires_grad]
        if ep:
            emb_lr = base_lr * (decay_factor ** n_layers)
            groups += [
                {"params": [p for n, p in ep if not any(nd in n for nd in no_decay)], "lr": emb_lr, "weight_decay": weight_decay},
                {"params": [p for n, p in ep if any(nd in n for nd in no_decay)], "lr": emb_lr, "weight_decay": 0.0},
            ]
        # Parameters outside encoder/embeddings (MoLFormer's final LayerNorm, RoBERTa's
        # unused pooler) are left out of the optimizer, exactly like the notebook.
    groups += [
        {"params": [p for n, p in new if not any(nd in n for nd in no_decay)], "lr": base_lr * new_lr_mult, "weight_decay": weight_decay},
        {"params": [p for n, p in new if any(nd in n for nd in no_decay)], "lr": base_lr * new_lr_mult, "weight_decay": 0.0},
    ]
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, fused=torch.cuda.is_available())


# ─────────────────────────────────────────────────────────────────────────────
# Run configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RunCfg:
    variant: str = "full"
    backbone: str = "molformer"
    hp_set: str = "tuned"
    modalities: tuple = MODALITIES
    fusion: str = "xattn"
    use_type_emb: bool = True
    use_gate: bool = True
    pooling: str = "triple"
    n_pool_layers: int = 4
    freeze_bottom_n: int | None = None      # None = bottom third (4 for MoLFormer)
    freeze_all_bert: bool = False
    loss: str = "focal"                     # focal | bce
    augment: bool = True
    sampler: bool = True
    llrd: float = 0.95
    fast_molformer: bool = True
    max_epochs: int = 50
    patience: int = 12
    grad_accum: int = 2
    # V4 options (omitted from to_dict() while at their defaults, so V3 config hashes are unchanged)
    fp_kind: str = "ecfp1024"               # ecfp1024 (notebook) | ecfp2048c (count, chirality, log1p)
    fp_dropout: float = 0.0
    graph: bool = False
    new_lr_mult: float = 1.0
    aux_pic50: float = 0.0                  # V5: weight of the auxiliary pIC50 regression loss
    task: str = "cls"                       # V5: cls | reg (classify by predicted pIC50)

    @property
    def hp(self):
        return HP_SETS[self.hp_set]

    def to_dict(self):
        d = asdict(self)
        d["modalities"] = list(d["modalities"])
        d["hp"] = dict(self.hp)
        for k, v in V4_DEFAULTS.items():
            if d.get(k) == v:
                d.pop(k)
        return d


V4_DEFAULTS = {"fp_kind": "ecfp1024", "fp_dropout": 0.0, "graph": False, "new_lr_mult": 1.0,
               "aux_pic50": 0.0, "task": "cls"}
_V4A = {"fp_kind": "ecfp2048c", "fp_dropout": 0.1}


VARIANTS = {
    # name: (description, overrides)
    "full":            ("Full V3 (reference)", {}),
    "no_smiles":       ("- SMILES language-model branch", {"modalities": ("ecfp", "maccs", "desc")}),
    "no_ecfp":         ("- ECFP", {"modalities": ("smiles", "maccs", "desc")}),
    "no_maccs":        ("- MACCS", {"modalities": ("smiles", "ecfp", "desc")}),
    "no_desc":         ("- Descriptors", {"modalities": ("smiles", "ecfp", "maccs")}),
    "smiles_only":     ("SMILES language model only", {"modalities": ("smiles",)}),
    "concat_fusion":   ("Concat fusion (no cross-modal Transformer, no type emb, no gate)",
                        {"fusion": "concat", "use_type_emb": False, "use_gate": False}),
    "no_type_emb":     ("- Modality type embeddings", {"use_type_emb": False}),
    "no_gate":         ("- Per-modality gate", {"use_gate": False}),
    "last_layer_pool": ("Last encoder layer only (no multi-layer mix)", {"n_pool_layers": 1}),
    "cls_pool":        ("CLS pooling only (no mean/attn pooling)", {"pooling": "cls"}),
    "frozen_bert":     ("Frozen language model (no fine-tuning)", {"freeze_all_bert": True}),
    "bce_loss":        ("Plain BCE instead of Focal loss", {"loss": "bce"}),
    "no_augment":      ("- SMILES enumeration augmentation", {"augment": False}),
    "no_sampler":      ("Unbalanced (a): no WeightedRandomSampler, natural 79/21 ratio", {"sampler": False}),
    "no_llrd":         ("- LLRD (flat LR for all encoder layers)", {"llrd": 1.0}),
    "vanilla":         ("Unbalanced (b): no sampler + plain BCE (no imbalance handling at all)",
                        {"sampler": False, "loss": "bce"}),
    # ── V4 ──
    "fp_upgrade":      ("V4-A: count ECFP4 2048 bits + chirality + fingerprint dropout", dict(_V4A)),
    # new_lr_mult from pilot_v4_lr.py (CV fold 0 validation only): token fusion x30, D-MPNN x10
    "token_fusion":    ("V4-B: V4-A + substructure tokens, token-level cross-attention (no gate)",
                        dict(_V4A, fusion="token", use_gate=False, new_lr_mult=30.0)),
    "graph_branch":    ("V4-C: V4-A + D-MPNN graph branch as a 5th modality", dict(_V4A, graph=True, new_lr_mult=10.0)),
    # ── V5: potency-aware training (train-fold pIC50 only; test labels unchanged) ──
    "graph_mt":        ("V5-MT: V4-C + auxiliary pIC50 regression head (loss weight 0.1)",
                        dict(_V4A, graph=True, new_lr_mult=10.0, aux_pic50=0.1)),
    "graph_reg":       ("V5-REG: V4-C trained to regress pIC50; active if predicted pIC50 is high",
                        dict(_V4A, graph=True, new_lr_mult=10.0, task="reg")),
}


def make_cfg(variant="full", backbone="molformer", hp_set="tuned", **extra):
    _, ov = VARIANTS[variant]
    cfg = RunCfg(variant=variant, backbone=backbone, hp_set=hp_set, **ov)
    for k, v in extra.items():
        setattr(cfg, k, v)
    return cfg


def build_model(cfg: RunCfg):
    hp = cfg.hp
    return V3(backbone=cfg.backbone, modalities=cfg.modalities, fusion=cfg.fusion,
              use_type_emb=cfg.use_type_emb, use_gate=cfg.use_gate, pooling=cfg.pooling,
              n_pool_layers=cfg.n_pool_layers, freeze_bottom_n=cfg.freeze_bottom_n,
              freeze_all_bert=cfg.freeze_all_bert, num_heads=hp["num_heads"], dropout=hp["dropout"],
              hidden_dim=hp["hidden_dim"], num_classifier_layers=hp["num_classifier_layers"],
              n_cross_layers=hp["n_cross_layers"], fast_molformer=cfg.fast_molformer,
              ecfp_dim=2048 if cfg.fp_kind == "ecfp2048c" else ECFP_BITS, fp_dropout=cfg.fp_dropout,
              graph=cfg.graph, reg_head=cfg.aux_pic50 > 0 or cfg.task == "reg", task=cfg.task)


# ─────────────────────────────────────────────────────────────────────────────
# Train / predict
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_torch():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False  # variable sequence lengths


@torch.no_grad()
def predict(model, ds, device, collate, batch_size=64, eval_seed=None, amp_dtype=torch.bfloat16):
    """Probabilities for ds. MoLFormer redraws its random attention features on every
    forward pass; eval_seed makes that draw repeatable without touching the training RNG."""
    model.eval()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    if eval_seed is not None:
        torch.manual_seed(eval_seed)
    probs = []
    try:
        for i in range(0, len(ds), batch_size):
            b = collate([ds[j] for j in range(i, min(i + batch_size, len(ds)))])
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                lg = model(**{k: v.to(device, non_blocking=True) for k, v in b.items() if k not in ("labels", "pic50")})
            probs.append(torch.sigmoid(lg.float()).cpu().numpy())
    finally:
        if eval_seed is not None:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
    return np.concatenate(probs)


def train_one(cfg: RunCfg, feats, tr_idx, vl_idx, eval_sets: dict, seed: int, device,
              tokenizer, log=print, guard=None, save_path=None):
    """Train on tr_idx, early-stop on AUROC of vl_idx; return predictions of the best
    checkpoint on vl_idx and on every eval set (optionally save that checkpoint)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from torch.optim.lr_scheduler import LambdaLR

    set_seed(seed)
    hp = cfg.hp
    augment_ok = cfg.augment and "smiles" in cfg.modalities
    scaler = StandardScaler().fit(feats["desc_raw"][tr_idx])
    dkw, ckw = {}, {}
    if cfg.fp_kind != "ecfp1024" or cfg.graph:          # V4 inputs (cached separately)
        import v4_modules
        feats = dict(feats, **v4_modules.get_v4_features(feats["smiles"]))
        dkw = {"fp_key": "ecfp2048c" if cfg.fp_kind == "ecfp2048c" else "ecfp", "graphs": cfg.graph}
        ckw = {"fp_tokens": cfg.fusion == "token", "graphs": cfg.graph}
    use_reg = cfg.aux_pic50 > 0 or cfg.task == "reg"
    if use_reg:                                          # V5: measured potency, training molecules only
        import data as _data
        feats = dict(feats, pic50=_data.get_pic50(feats["smiles"]))
        dkw = dict(dkw, pic50=True)
        ckw = dict(ckw, pic50=True)
    tr_ds = V3Dataset(feats, tr_idx, tokenizer, scaler, augment=augment_ok, **dkw)
    vl_ds = V3Dataset(feats, vl_idx, tokenizer, scaler, **dkw)
    ev_ds = {k: V3Dataset(feats, idx, tokenizer, scaler, **dkw) for k, idx in eval_sets.items()}
    collate = make_collate(tokenizer.pad_token_id, **ckw)

    if cfg.sampler:
        lbl = tr_ds.labels.astype(int)
        w = (1.0 / np.bincount(lbl))[lbl]
        sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w, dtype=torch.float), len(w), replacement=True)
        loader = torch.utils.data.DataLoader(tr_ds, batch_size=hp["batch_size"], sampler=sampler,
                                             collate_fn=collate, num_workers=0, pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(tr_ds, batch_size=hp["batch_size"], shuffle=True,
                                             collate_fn=collate, num_workers=0, pin_memory=True)

    model = build_model(cfg).to(device)
    if use_reg:
        p_tr = feats["pic50"][tr_idx]
        model.pic_mu.fill_(float(p_tr.mean()))
        model.pic_sd.fill_(float(p_tr.std()))
    opt = build_llrd_optimizer(model, hp["learning_rate"], hp["weight_decay"], cfg.llrd, cfg.new_lr_mult)
    crit = FocalLoss(alpha=0.5, gamma=2.0, smoothing=0.05) if cfg.loss == "focal" else nn.BCEWithLogitsLoss()

    total = (len(loader) // cfg.grad_accum) * cfg.max_epochs
    warm = max(1, int(0.10 * total))

    def lr_lambda(step):
        if step < warm:
            return step / warm
        prog = (step - warm) / max(1, total - warm)
        return max(1e-7, 0.5 * (1.0 + math.cos(math.pi * prog)))
    sch = LambdaLR(opt, lr_lambda)

    best_auc, best_state, best_ep, pat = -1.0, None, 0, 0
    history = []
    t0 = time.time()
    for ep in range(cfg.max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        run_loss = 0.0
        for bi, b in enumerate(loader):
            if guard is not None:
                guard.check()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                inputs = {k: v.to(device, non_blocking=True) for k, v in b.items() if k not in ("labels", "pic50")}
                if use_reg:
                    lg, zr = model(**inputs, return_reg=True)
                    zt = (b["pic50"].to(device) - model.pic_mu) / model.pic_sd
                    reg = F.smooth_l1_loss(zr.float(), zt)
                    loss = (reg if cfg.task == "reg" else crit(lg.float(), b["labels"].to(device)) + cfg.aux_pic50 * reg)
                    loss = loss / cfg.grad_accum
                else:
                    lg = model(**inputs)
                    loss = crit(lg.float(), b["labels"].to(device)) / cfg.grad_accum
            loss.backward()
            run_loss += loss.item() * cfg.grad_accum
            if (bi + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sch.step()
        vp = predict(model, vl_ds, device, collate, eval_seed=1000 + ep)
        auc = roc_auc_score(vl_ds.labels, vp)
        history.append((ep + 1, run_loss / len(loader), float(auc)))
        if auc > best_auc:
            best_auc, best_ep, pat = auc, ep + 1, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= cfg.patience:
                break
    train_s = time.time() - t0

    model.load_state_dict(best_state)
    if save_path:
        torch.save({"cfg": cfg.to_dict(), "state_dict": best_state, "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_}, save_path)
    out = {"val_probs": predict(model, vl_ds, device, collate, eval_seed=7), "val_labels": vl_ds.labels}
    for k, ds in ev_ds.items():
        out[f"{k}_probs"] = predict(model, ds, device, collate, eval_seed=7)
        out[f"{k}_labels"] = ds.labels
    out.update(best_val_auc=float(best_auc), best_epoch=best_ep, epochs_run=len(history),
               train_seconds=train_s, history=history,
               n_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    del model, opt
    torch.cuda.empty_cache()
    return out
