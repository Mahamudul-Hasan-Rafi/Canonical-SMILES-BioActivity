"""
FROZEN LEGACY IMPLEMENTATION - do not edit. It produced the reproduction and ablation
results and is kept only as the reference for tests/test_core_parity.py; use core.py.

Core of the V3 reproduction + ablation study.

This is a faithful port of `bioactivity_dl 8_2_C_v3_re.ipynb` (dataset, V3 model,
FocalLoss, LLRD, warmup-cosine schedule, WeightedRandomSampler, SMILES
augmentation, early stopping on val AUROC) with three kinds of changes:

  1. SPEED (no change to the maths):
     - Dynamic padding: sequences are padded to the longest SMILES in the batch
       instead of 512. MoLFormer's linear attention zeroes padded keys and V3
       pools with masks, so outputs for real tokens are identical
       (verified numerically by `check_saved_models.py`).
     - Features (ECFP / MACCS / descriptors / token ids) are computed once and
       cached on disk.
     - TF32 matmuls, larger eval batches, no per-epoch model saving to disk
       (the best state_dict is kept in RAM).
  2. ABLATION SWITCHES on the model / training recipe (defaults == original V3).
  3. THERMAL GUARD: training pauses whenever the GPU is hotter than a limit.

Nothing here writes into the notebook's `models/` folder.
"""
from __future__ import annotations

import contextlib
import math
import os
import random
import subprocess
import time
import warnings
from dataclasses import dataclass, field, asdict

# Never talk to the Hub: the notebook used MoLFormer revision 7b12d94; the Hub's
# current revision needs a newer transformers and would break everything.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_XLSX = r"E:\ML\BioActivity\Dataset\bioactivity_dataset_cleaned_new_v3.xlsx"
CACHE_NPZ = os.path.join(HERE, "cache", "features_v3.npz")
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
REVISION = "7b12d946c181a37f6012b9dc3b002275de070314"

ECFP_BITS, ECFP_RADIUS, MACCS_BITS, N_DESC = 1024, 2, 167, 6
D = 768

# Best V3 hyper-parameters (optuna_results_v3__.json, used by the notebook)
BEST_PARAMS = {
    "num_heads": 4, "hidden_dim": 256, "dropout": 0.1,
    "num_classifier_layers": 2, "n_cross_layers": 2, "batch_size": 32,
    "learning_rate": 9.8916998354799e-06, "weight_decay": 2.1247834038360546e-05,
}


# ─────────────────────────────────────────────────────────────────────────────
# Thermal guard
# ─────────────────────────────────────────────────────────────────────────────
class ThermalGuard:
    """Polls nvidia-smi every `every_s` seconds; sleeps while GPU temp >= hot_c
    until it cools to cool_c. Cheap enough to call every training step."""

    def __init__(self, hot_c=80, cool_c=72, every_s=20, log=print):
        self.hot_c, self.cool_c, self.every_s, self.log = hot_c, cool_c, every_s, log
        self._last = 0.0
        self.paused_s = 0.0

    @staticmethod
    def gpu_temp():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip().splitlines()
            return int(out[0])
        except Exception:
            return None

    def check(self):
        now = time.time()
        if now - self._last < self.every_s:
            return
        self._last = now
        t = self.gpu_temp()
        if t is None or t < self.hot_c:
            return
        self.log(f"[thermal] GPU {t}C >= {self.hot_c}C - pausing until <= {self.cool_c}C")
        t0 = time.time()
        while True:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time.sleep(15)
            t = self.gpu_temp()
            if t is None or t <= self.cool_c:
                break
        self.paused_s += time.time() - t0
        self.log(f"[thermal] resumed at {t}C after {time.time() - t0:.0f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Features (computed once, cached)
# ─────────────────────────────────────────────────────────────────────────────
def register_remote_code():
    """Import MoLFormer's remote code so `transformers_modules.*` exists for unpickling."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    get_class_from_dynamic_module("modeling_molformer.MolformerModel", MODEL_NAME, revision=REVISION)


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, revision=REVISION)


def load_df():
    return pd.read_excel(DATA_XLSX, sheet_name="Sheet1")


def _featurize_one(smi):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
    mol = Chem.MolFromSmiles(smi)
    ecfp = np.zeros(ECFP_BITS, np.float32)
    maccs = np.zeros(MACCS_BITS, np.float32)
    desc = np.zeros(N_DESC, np.float32)
    if mol is not None:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_BITS)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), ecfp)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), maccs)
        try:
            desc = np.array([
                Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
            ], np.float32)
        except Exception:
            pass
    return ecfp, maccs, desc


def get_features(df=None):
    """Returns dict with smiles, labels, ecfp, maccs, desc_raw (unscaled)."""
    if df is None:
        df = load_df()
    smiles = df["canonical_smiles"].tolist()
    if os.path.exists(CACHE_NPZ):
        z = np.load(CACHE_NPZ, allow_pickle=True)
        if list(z["smiles"]) == smiles:
            return {k: z[k] for k in z.files}
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(_featurize_one, smiles, chunksize=64))
    feats = {
        "smiles": np.array(smiles, dtype=object),
        "labels": df["bioactivity"].values.astype(np.float32),
        "ecfp": np.stack([r[0] for r in res]),
        "maccs": np.stack([r[1] for r in res]),
        "desc_raw": np.stack([r[2] for r in res]),
    }
    os.makedirs(os.path.dirname(CACHE_NPZ), exist_ok=True)
    np.savez_compressed(CACHE_NPZ, **feats)
    return feats


def notebook_splits(df):
    """Exactly the notebook's splits (same calls, same random_state)."""
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["bioactivity"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    return train_df.index.values, val_df.index.values, test_df.index.values


def notebook_folds(df, train_idx, val_idx, n_folds=5):
    """Notebook's K-fold: dev = concat(train, val) (reset index), StratifiedKFold(42).
    Returns dev_idx (positions into df) and list of (tr, vl) positions into dev."""
    from sklearn.model_selection import StratifiedKFold
    dev_idx = np.concatenate([train_idx, val_idx])
    y = df["bioactivity"].values[dev_idx].astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    return dev_idx, [(tr, vl) for tr, vl in skf.split(np.zeros(len(y)), y)]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset with dynamic padding
# ─────────────────────────────────────────────────────────────────────────────
def random_smiles(smi):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return smi
    return Chem.MolToSmiles(mol, doRandom=True, canonical=False)


class V3Dataset(torch.utils.data.Dataset):
    def __init__(self, feats, idx, tokenizer, scaler, augment=False, max_len=512):
        self.tok = tokenizer
        self.smiles = [feats["smiles"][i] for i in idx]
        self.labels = feats["labels"][idx].astype(np.float32)
        self.ecfp = feats["ecfp"][idx].astype(np.float32)
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
        return ids, self.ecfp[i], self.maccs[i], self.desc[i], self.labels[i]


def make_collate(pad_id, fixed_len=None):
    def collate(batch):
        L = fixed_len or max(len(b[0]) for b in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for j, b in enumerate(batch):
            ids[j, :len(b[0])] = torch.from_numpy(b[0])
            mask[j, :len(b[0])] = 1
        return {
            "input_ids": ids, "attention_mask": mask,
            "ecfp": torch.from_numpy(np.stack([b[1] for b in batch])),
            "maccs": torch.from_numpy(np.stack([b[2] for b in batch])),
            "descriptors": torch.from_numpy(np.stack([b[3] for b in batch])),
            "labels": torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float),
        }
    return collate


# ─────────────────────────────────────────────────────────────────────────────
# MoLFormer speed patch (removes 24 GPU->CPU syncs per forward pass)
# ─────────────────────────────────────────────────────────────────────────────
def _fast_feature_map_forward(self, query, key):
    """Same as MolformerFeatureMap.forward, but the orthogonal random features
    are drawn + QR-decomposed on the CPU (64x64, ~0.3 ms) and copied
    asynchronously, instead of a GPU QR that blocks the CPU every layer.
    Same distribution; the random numbers come from the CPU generator."""
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
        """MolformerSelfAttention.forward minus the `torch.equal` sanity check
        (a blocking sync per layer). V3 always passes a plain 2-D padding mask."""
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
# Model (V3 with ablation switches; defaults reproduce the notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
MODALITIES = ("smiles", "ecfp", "maccs", "desc")


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


class V3Ablatable(nn.Module):
    def __init__(self, modalities=MODALITIES, fusion="xattn", use_type_emb=True, use_gate=True,
                 pooling="triple", n_pool_layers=4, freeze_bottom_n=4, freeze_all_bert=False,
                 num_heads=8, dropout=0.2, hidden_dim=512, num_classifier_layers=4,
                 n_cross_layers=3, ecfp_bits=ECFP_BITS, maccs_bits=MACCS_BITS, n_descriptors=N_DESC,
                 fast_molformer=True):
        super().__init__()
        from transformers import AutoModel
        self.modalities = tuple(m for m in MODALITIES if m in modalities)
        self.fusion, self.use_type_emb, self.use_gate = fusion, use_type_emb, use_gate
        self.pooling, self.n_pool_layers = pooling, n_pool_layers
        self.freeze_all_bert = freeze_all_bert
        n_mod = len(self.modalities)

        if "smiles" in self.modalities:
            self.bert = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, revision=REVISION)
            if fast_molformer:
                patch_molformer_fast(self.bert)
            n_freeze = len(self.bert.encoder.layer) if freeze_all_bert else freeze_bottom_n
            for layer in self.bert.encoder.layer[:n_freeze]:
                for p in layer.parameters():
                    p.requires_grad = False
            if freeze_all_bert:
                for p in self.bert.parameters():
                    p.requires_grad = False
            self.layer_weights = nn.Parameter(torch.zeros(n_pool_layers))
            self.token_attn = nn.Linear(D, 1)

        self.modal_type_embed = nn.Embedding(4, D)
        nn.init.xavier_uniform_(self.modal_type_embed.weight)

        if "ecfp" in self.modalities:
            self.ecfp_proj = nn.Sequential(
                nn.Linear(ecfp_bits, 512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(dropout * 0.5),
                nn.Linear(512, D), nn.GELU(), nn.LayerNorm(D), nn.Dropout(dropout))
        if "maccs" in self.modalities:
            self.maccs_proj = nn.Sequential(
                nn.Linear(maccs_bits, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(dropout * 0.5),
                nn.Linear(256, D), nn.GELU(), nn.LayerNorm(D), nn.Dropout(dropout))
        if "desc" in self.modalities:
            self.desc_proj = nn.Sequential(
                nn.Linear(n_descriptors, 64), nn.GELU(), nn.LayerNorm(64), nn.Dropout(dropout * 0.5),
                nn.Linear(64, D), nn.GELU(), nn.LayerNorm(D), nn.Dropout(dropout))

        if fusion == "xattn":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=D, nhead=num_heads, dim_feedforward=D * 4,
                dropout=dropout, batch_first=True, norm_first=True)
            self.cross_modal = nn.TransformerEncoder(enc_layer, num_layers=n_cross_layers)
        self.gate = nn.Sequential(nn.Linear(D, D), nn.LayerNorm(D), nn.Sigmoid())
        self.fusion_proj = nn.Linear(D * n_mod, D)

        layers, in_dim = [], D
        for i in range(num_classifier_layers):
            out_dim = hidden_dim // (2 ** i)
            layers += [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
                       nn.Dropout(dropout if i < num_classifier_layers - 1 else dropout * 0.5)]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_all_bert and hasattr(self, "bert"):
            self.bert.eval()   # frozen feature extractor: no dropout
        return self

    def _smiles_vec(self, input_ids, attention_mask):
        with torch.no_grad() if self.freeze_all_bert else contextlib.nullcontext():
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        selected = out.hidden_states[-self.n_pool_layers:]
        w = torch.softmax(self.layer_weights, dim=0)
        hidden = sum(wi * h for wi, h in zip(w, selected))
        cls_pool = hidden[:, 0]
        if self.pooling == "cls":
            return cls_pool
        m = attention_mask.unsqueeze(-1).to(hidden.dtype)
        mean_pool = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
        if self.pooling == "mean":
            return mean_pool
        scores = self.token_attn(hidden).squeeze(-1).masked_fill(~attention_mask.bool(), float("-inf"))
        attn_pool = (hidden * torch.softmax(scores, dim=-1).unsqueeze(-1)).sum(1)
        return (cls_pool + mean_pool + attn_pool) / 3

    def forward(self, input_ids, attention_mask, ecfp, maccs, descriptors):
        embs, ids = [], []
        for k, m in enumerate(MODALITIES):
            if m not in self.modalities:
                continue
            if m == "smiles":
                embs.append(self._smiles_vec(input_ids, attention_mask))
            elif m == "ecfp":
                embs.append(self.ecfp_proj(ecfp))
            elif m == "maccs":
                embs.append(self.maccs_proj(maccs))
            else:
                embs.append(self.desc_proj(descriptors))
            ids.append(k)
        tokens = torch.stack(embs, dim=1)                         # [B, n_mod, D]
        B = tokens.size(0)
        if self.fusion == "xattn":
            if self.use_type_emb:
                t = self.modal_type_embed(torch.tensor(ids, device=tokens.device))
                tokens = tokens + t.unsqueeze(0)
            tokens = self.cross_modal(tokens)
        if self.use_gate:
            tokens = self.gate(tokens) * tokens
        fused = self.fusion_proj(tokens.reshape(B, -1))
        return self.classifier(fused).squeeze(-1)


def build_llrd_optimizer(model, base_lr, weight_decay, decay_factor=0.95):
    """Same grouping as the notebook's build_llrd_optimizer."""
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    groups = []
    head = [(n, p) for n, p in model.named_parameters() if not n.startswith("bert.") and p.requires_grad]
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
        # MoLFormer's final LayerNorm (bert.LayerNorm) is outside encoder/embeddings and,
        # exactly like the notebook, is left out of the optimizer (never updated).
    groups = [g for g in groups if g["params"]]
    # fused=True: identical AdamW update, one CUDA kernel instead of dozens per step
    return torch.optim.AdamW(groups, fused=torch.cuda.is_available())


# ─────────────────────────────────────────────────────────────────────────────
# Experiment config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RunCfg:
    variant: str = "full"
    # model switches
    modalities: tuple = MODALITIES
    fusion: str = "xattn"
    use_type_emb: bool = True
    use_gate: bool = True
    pooling: str = "triple"
    n_pool_layers: int = 4
    freeze_bottom_n: int = 4
    freeze_all_bert: bool = False
    # recipe switches
    loss: str = "focal"          # focal | bce
    augment: bool = True
    sampler: bool = True
    llrd: float = 0.95           # 1.0 == no layer-wise decay
    fast_molformer: bool = True
    # schedule
    max_epochs: int = 50
    patience: int = 12
    grad_accum: int = 2
    hp: dict = field(default_factory=lambda: dict(BEST_PARAMS))


ABLATIONS = {
    # name: (description, overrides)
    "full":            ("Full V3 (reference)", {}),
    "no_smiles":       ("- MoLFormer / SMILES branch", {"modalities": ("ecfp", "maccs", "desc")}),
    "no_ecfp":         ("- ECFP", {"modalities": ("smiles", "maccs", "desc")}),
    "no_maccs":        ("- MACCS", {"modalities": ("smiles", "ecfp", "desc")}),
    "no_desc":         ("- Descriptors", {"modalities": ("smiles", "ecfp", "maccs")}),
    "smiles_only":     ("MoLFormer only", {"modalities": ("smiles",)}),
    "concat_fusion":   ("Concat fusion (no cross-modal Transformer, no type emb, no gate)",
                        {"fusion": "concat", "use_type_emb": False, "use_gate": False}),
    "no_type_emb":     ("- Modality type embeddings", {"use_type_emb": False}),
    "no_gate":         ("- Per-modality gate", {"use_gate": False}),
    "last_layer_pool": ("Last BERT layer only (no multi-layer mix)", {"n_pool_layers": 1}),
    "cls_pool":        ("CLS pooling only (no mean/attn pooling)", {"pooling": "cls"}),
    "frozen_bert":     ("Frozen MoLFormer (no fine-tuning)", {"freeze_all_bert": True}),
    "bce_loss":        ("Plain BCE instead of Focal loss", {"loss": "bce"}),
    "no_augment":      ("- SMILES enumeration augmentation", {"augment": False}),
    "no_sampler":      ("- WeightedRandomSampler (natural imbalance)", {"sampler": False}),
    "no_llrd":         ("- LLRD (flat LR for all BERT layers)", {"llrd": 1.0}),
}


def make_cfg(variant, **extra):
    desc, ov = ABLATIONS[variant]
    cfg = RunCfg(variant=variant, **ov)
    for k, v in extra.items():
        setattr(cfg, k, v)
    return cfg


def build_model(cfg: RunCfg):
    hp = cfg.hp
    return V3Ablatable(
        modalities=cfg.modalities, fusion=cfg.fusion, use_type_emb=cfg.use_type_emb,
        use_gate=cfg.use_gate, pooling=cfg.pooling, n_pool_layers=cfg.n_pool_layers,
        freeze_bottom_n=cfg.freeze_bottom_n, freeze_all_bert=cfg.freeze_all_bert,
        num_heads=hp["num_heads"], dropout=hp["dropout"], hidden_dim=hp["hidden_dim"],
        num_classifier_layers=hp["num_classifier_layers"], n_cross_layers=hp["n_cross_layers"],
        fast_molformer=cfg.fast_molformer)


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
    torch.backends.cudnn.benchmark = False  # variable seq lengths -> benchmark hurts


@torch.no_grad()
def predict(model, ds, device, collate, batch_size=64, eval_seed=None, amp_dtype=torch.bfloat16):
    """Returns probabilities. MoLFormer redraws random attention features on every
    forward pass (deterministic_eval=False); eval_seed makes that draw repeatable
    without disturbing the training RNG stream."""
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
                lg = model(input_ids=b["input_ids"].to(device, non_blocking=True),
                           attention_mask=b["attention_mask"].to(device, non_blocking=True),
                           ecfp=b["ecfp"].to(device), maccs=b["maccs"].to(device),
                           descriptors=b["descriptors"].to(device))
            probs.append(torch.sigmoid(lg.float()).cpu().numpy())
    finally:
        if eval_seed is not None:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
    return np.concatenate(probs)


def train_one(cfg: RunCfg, feats, tr_idx, vl_idx, eval_sets: dict, seed: int, device,
              tokenizer, log=print, guard: ThermalGuard | None = None):
    """Train one model on tr_idx, early-stop on AUROC of vl_idx, return
    predictions of the best checkpoint on vl_idx and every eval set."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from torch.optim.lr_scheduler import LambdaLR

    set_seed(seed)
    hp = cfg.hp
    augment_ok = cfg.augment and "smiles" in cfg.modalities
    scaler = StandardScaler().fit(feats["desc_raw"][tr_idx])
    tr_ds = V3Dataset(feats, tr_idx, tokenizer, scaler, augment=augment_ok)
    vl_ds = V3Dataset(feats, vl_idx, tokenizer, scaler)
    ev_ds = {k: V3Dataset(feats, idx, tokenizer, scaler) for k, idx in eval_sets.items()}
    collate = make_collate(tokenizer.pad_token_id)

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
    opt = build_llrd_optimizer(model, hp["learning_rate"], hp["weight_decay"], cfg.llrd)
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
                lg = model(input_ids=b["input_ids"].to(device, non_blocking=True),
                           attention_mask=b["attention_mask"].to(device, non_blocking=True),
                           ecfp=b["ecfp"].to(device, non_blocking=True),
                           maccs=b["maccs"].to(device, non_blocking=True),
                           descriptors=b["descriptors"].to(device, non_blocking=True))
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
    out = {"val_probs": predict(model, vl_ds, device, collate, eval_seed=7),
           "val_labels": vl_ds.labels}
    for k, ds in ev_ds.items():
        out[f"{k}_probs"] = predict(model, ds, device, collate, eval_seed=7)
        out[f"{k}_labels"] = ds.labels
    out.update(best_val_auc=float(best_auc), best_epoch=best_ep, epochs_run=len(history),
               train_seconds=train_s, history=history,
               n_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    del model, opt
    torch.cuda.empty_cache()
    return out
