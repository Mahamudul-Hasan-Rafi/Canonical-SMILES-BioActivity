# Verbatim copy of the notebook classes (cells 9 and 11) so torch.load can unpickle
# the models saved from the notebook's __main__.
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
ECFP_BITS, MACCS_BITS, N_DESCRIPTORS_EXT = 1024, 167, 6

class FocalLoss(nn.Module):
    """
    Binary Focal Loss with label smoothing.

    Intuition
    ---------
    Standard BCE penalises every error equally. When the model is very
    confident and correct (p ≈ 1 for a positive), the gradient is tiny
    anyway — but it still wastes capacity. Focal loss multiplies by
    (1-p)^gamma, so easy well-classified examples contribute almost
    nothing to training and the model focuses its capacity on the hard
    boundary cases that cause the accuracy plateau.

      alpha   : upweight for positives  (≈ class_neg / class_pos)
      gamma   : focusing exponent       (0 = plain BCE, 2 = standard focal)
      smoothing: label smoothing ε      (prevents overconfident logits)
    """
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.smoothing = smoothing

    def forward(self, logits, targets):
        y    = targets * (1 - self.smoothing) + 0.5 * self.smoothing   # smooth labels
        bce  = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
        pt   = torch.exp(-bce)                                          # p_correct
        a_t  = self.alpha * targets + (1 - self.alpha) * (1 - targets) # per-sample alpha
        loss = a_t * (1 - pt) ** self.gamma * bce
        return loss.mean()




# ─────────────────────────────────────────────────────────────────────────────
# V3 Architecture — Improvements over V2:
#
#   1. Multi-layer weighted BERT pooling (last 4 hidden states)
#   2. Deeper cross-modal Transformer (3 layers, Pre-LN)
#   3. FIXED: Modality type embeddings — 4 learnable vectors tell the
#      cross-modal Transformer which token is SMILES / ECFP / MACCS / DESC.
#      Without these, attention cannot distinguish modalities by type.
#   4. FIXED: modal_type_embed uses xavier_uniform_ init (not Normal(0,1))
#      so type embeddings start at the same scale as projected features.
#   5. FIXED: Gate has LayerNorm before Sigmoid to prevent saturation.
#   6. FIXED: Projections use GELU → LayerNorm (not LayerNorm → GELU) so
#      each modality token enters the cross-modal Transformer with unit
#      variance, making type-embedding addition scale-consistent.
#   7. FIXED: mean_pool excludes padding tokens (masked mean).
#   8. FIXED: Per-modality gate (D→D per token) replaces global gate (4D→D).
#   9. FIXED: freeze_bottom_n layers frozen in __init__ — class is self-
#      protecting against catastrophic forgetting if LLRD is not used.
# ─────────────────────────────────────────────────────────────────────────────

class FineTunedBERTaECFP_v3(nn.Module):
    def __init__(
        self,
        ecfp_bits             = ECFP_BITS,
        maccs_bits            = MACCS_BITS,
        n_descriptors         = N_DESCRIPTORS_EXT,
        num_heads             = 8,
        dropout               = 0.2,
        hidden_dim            = 512,
        num_classifier_layers = 4,
        n_cross_layers        = 3,
        n_pool_layers         = 4,
        freeze_bottom_n       = 4,
    ):
        super().__init__()
        D = 768

        self.bert = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True, revision="7b12d946c181a37f6012b9dc3b002275de070314"
        )

        # Freeze bottom freeze_bottom_n encoder layers to prevent catastrophic
        # forgetting of chemistry pretraining. Top layers remain trainable via
        # LLRD. Default=4 matches the training-cell freeze setting.
        for layer in self.bert.encoder.layer[:freeze_bottom_n]:
            for p in layer.parameters():
                p.requires_grad = False

        self.n_pool_layers = n_pool_layers
        self.layer_weights = nn.Parameter(torch.zeros(n_pool_layers))
        self.token_attn    = nn.Linear(D, 1)

        # ── Modality type embeddings ──────────────────────────────────────────
        # 4 learnable vectors: idx 0=SMILES, 1=ECFP, 2=MACCS, 3=DESC.
        # xavier_uniform_ keeps initial scale ≈ 0.05 — same order of magnitude
        # as the LayerNorm-normalised projection outputs (unit variance).
        self.modal_type_embed = nn.Embedding(4, D)
        nn.init.xavier_uniform_(self.modal_type_embed.weight)

        # ── Modality projections (GELU → LayerNorm at each stage) ─────────────
        # Ending each stage with LayerNorm ensures the token entering the
        # cross-modal Transformer has unit variance, so type-embedding addition
        # is scale-consistent across all four modalities.
        self.ecfp_proj = nn.Sequential(
            nn.Linear(ecfp_bits,  512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(dropout * 0.5),
            nn.Linear(512,          D), nn.GELU(), nn.LayerNorm(D),   nn.Dropout(dropout),
        )
        self.maccs_proj = nn.Sequential(
            nn.Linear(maccs_bits, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(dropout * 0.5),
            nn.Linear(256,          D), nn.GELU(), nn.LayerNorm(D),   nn.Dropout(dropout),
        )
        self.desc_proj = nn.Sequential(
            nn.Linear(n_descriptors, 64), nn.GELU(), nn.LayerNorm(64), nn.Dropout(dropout * 0.5),
            nn.Linear(64,              D), nn.GELU(), nn.LayerNorm(D),  nn.Dropout(dropout),
        )

        # ── Cross-modal Transformer (Pre-LN) ──────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=num_heads, dim_feedforward=D * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.cross_modal = nn.TransformerEncoder(enc_layer, num_layers=n_cross_layers)

        # ── Gated fusion (per-modality) ───────────────────────────────────────
        # Gate applied to each of the 4 tokens independently [B,4,D] → [B,4,D].
        # Each modality+dimension gets its own learned gate value — not a single
        # global gate over the concatenation. 590K params vs 2.36M previously.
        self.gate        = nn.Sequential(nn.Linear(D, D), nn.LayerNorm(D), nn.Sigmoid())
        self.fusion_proj = nn.Linear(D * 4, D)

        # ── Classifier MLP ────────────────────────────────────────────────────
        layers, in_dim = [], D
        for i in range(num_classifier_layers):
            out_dim = hidden_dim // (2 ** i)
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout if i < num_classifier_layers - 1 else dropout * 0.5),
            ])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, ecfp, maccs, descriptors):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # ── Weighted aggregation of last n BERT hidden states ─────────────────
        # outputs.hidden_states: 13 tensors — [embedding, layer1, ..., layer12].
        # Slice [-n_pool_layers:] selects the top encoder layers only, never
        # the embedding (layer 0). With n_pool_layers=4: layers 9, 10, 11, 12.
        # weights[0] → layer 9 (deepest selected), weights[3] → layer 12 (last).
        all_hidden = outputs.hidden_states           # 13 tensors
        selected   = all_hidden[-self.n_pool_layers:]    # top n encoder layers
        weights    = torch.softmax(self.layer_weights, dim=0)
        hidden     = sum(w * h for w, h in zip(weights, selected))

        # ── Multi-strategy SMILES pooling ─────────────────────────────────────
        cls_pool      = hidden[:, 0]
        mask_expanded = attention_mask.unsqueeze(-1).float()   # [B, seq_len, 1]
        mean_pool     = (hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        scores        = self.token_attn(hidden).squeeze(-1)
        scores        = scores.masked_fill(~attention_mask.bool(), float('-inf'))
        attn_pool     = (hidden * torch.softmax(scores, dim=-1).unsqueeze(-1)).sum(dim=1)
        smiles        = (cls_pool + mean_pool + attn_pool) / 3

        # ── Modality projections ──────────────────────────────────────────────
        ecfp_emb  = self.ecfp_proj(ecfp)
        maccs_emb = self.maccs_proj(maccs)
        desc_emb  = self.desc_proj(descriptors)

        # ── Add modality type embeddings before cross-modal attention ─────────
        B         = smiles.size(0)
        modal_ids = torch.arange(4, device=smiles.device)           # [0,1,2,3]
        type_embs = self.modal_type_embed(modal_ids)                # [4, D]
        tokens    = torch.stack([smiles, ecfp_emb, maccs_emb, desc_emb], dim=1)  # [B, 4, D]
        tokens    = tokens + type_embs.unsqueeze(0)                 # broadcast [1, 4, D]

        # ── Cross-modal Transformer + per-modality gated fusion ───────────────
        tokens = self.cross_modal(tokens)                           # [B, 4, D]
        gates  = self.gate(tokens)                                  # [B, 4, D] — per-modality gate
        flat   = (gates * tokens).reshape(B, -1)                    # [B, 4D]   — gated & flattened
        fused  = self.fusion_proj(flat)                             # [B, D]

        return self.classifier(fused).squeeze(-1)

