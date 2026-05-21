import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(22, 28))
ax.set_xlim(0, 22)
ax.set_ylim(0, 28)
ax.axis('off')
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

# ── colour palette ────────────────────────────────────────────────────────────
C = {
    'smiles_dark':  '#1A3A5C',
    'smiles_mid':   '#2563EB',
    'smiles_light': '#60A5FA',
    'ecfp_dark':    '#1A3A1A',
    'ecfp_mid':     '#16A34A',
    'ecfp_light':   '#4ADE80',
    'desc_dark':    '#3B1A5C',
    'desc_mid':     '#7C3AED',
    'desc_light':   '#A78BFA',
    'attn_dark':    '#5C3A1A',
    'attn_mid':     '#D97706',
    'attn_light':   '#FCD34D',
    'gate_dark':    '#5C1A1A',
    'gate_mid':     '#DC2626',
    'gate_light':   '#FCA5A5',
    'cls_dark':     '#164E63',
    'cls_mid':      '#0891B2',
    'cls_light':    '#67E8F9',
    'white':        '#F8FAFC',
    'grey':         '#94A3B8',
    'panel':        '#1E2330',
    'panel2':       '#252B3B',
    'arrow':        '#CBD5E1',
}

def box(ax, x, y, w, h, fc, ec, lw=1.5, alpha=1.0, radius=0.25):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(p)
    return p

def label(ax, x, y, txt, fs=9, color='white', bold=False, ha='center', va='center', zorder=5):
    ax.text(x, y, txt, fontsize=fs, color=color,
            fontweight='bold' if bold else 'normal',
            ha=ha, va=va, zorder=zorder,
            fontfamily='DejaVu Sans')

def arrow(ax, x1, y1, x2, y2, color='#CBD5E1', lw=1.8, style='->', zorder=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle='arc3,rad=0'))

def grad_box(ax, x, y, w, h, c_top, c_bot, ec, lw=1.5, steps=40, radius=0.2):
    """Simulate gradient with stacked thin rectangles."""
    sh = h / steps
    for i in range(steps):
        t = i / steps
        r = tuple(int(c_top[j] + (c_bot[j]-c_top[j])*t) for j in range(3))
        fc = '#{:02x}{:02x}{:02x}'.format(*r)
        rect = plt.Rectangle((x - w/2, y - h/2 + i*sh), w, sh, color=fc, zorder=3)
        ax.add_patch(rect)
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor='none', edgecolor=ec, linewidth=lw, zorder=4)
    ax.add_patch(p)

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# ═══════════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(11, 27.3, 'FineTunedBERTaECFP Architecture',
        fontsize=19, color=C['white'], fontweight='bold',
        ha='center', va='center', zorder=6,
        fontfamily='DejaVu Sans')
ax.text(11, 26.85, 'Bioactivity Classification via Gated Multi-Modal Fusion',
        fontsize=11, color=C['grey'], ha='center', va='center', zorder=6)

# Thin divider line
ax.plot([1, 21], [26.55, 26.55], color='#334155', lw=1.5, zorder=2)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 0 — Inputs  (y ~ 25.3)
# ═══════════════════════════════════════════════════════════════════════════════
Y_IN = 25.2

# SMILES input
box(ax, 5, Y_IN, 5.2, 0.85, C['smiles_dark'], C['smiles_mid'], lw=2)
label(ax, 5, Y_IN+0.15, 'SMILES String', fs=10, bold=True, color=C['smiles_light'])
label(ax, 5, Y_IN-0.18, '"CC(=O)Oc1ccccc1C(=O)O"', fs=7.5, color=C['grey'])

# ECFP input
box(ax, 11, Y_IN, 5.2, 0.85, C['ecfp_dark'], C['ecfp_mid'], lw=2)
label(ax, 11, Y_IN+0.15, 'ECFP Fingerprints', fs=10, bold=True, color=C['ecfp_light'])
label(ax, 11, Y_IN-0.18, f'Bit vector  ∈ ℝ²⁰⁴⁸', fs=7.5, color=C['grey'])

# Descriptor input
box(ax, 17, Y_IN, 5.2, 0.85, C['desc_dark'], C['desc_mid'], lw=2)
label(ax, 17, Y_IN+0.15, 'Molecular Descriptors', fs=10, bold=True, color=C['desc_light'])
label(ax, 17, Y_IN-0.18, 'MW, LogP, TPSA, … ∈ ℝ⁷', fs=7.5, color=C['grey'])

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Encoders  (y ~ 23.2)
# ═══════════════════════════════════════════════════════════════════════════════
Y_ENC = 23.1

arrow(ax, 5,  Y_IN-0.43, 5,  Y_ENC+0.58, C['smiles_light'])
arrow(ax, 11, Y_IN-0.43, 11, Y_ENC+0.58, C['ecfp_light'])
arrow(ax, 17, Y_IN-0.43, 17, Y_ENC+0.58, C['desc_light'])

# MoLFormer block
grad_box(ax, 5, Y_ENC, 5.4, 1.2,
         hex_to_rgb('#1E3A6E'), hex_to_rgb('#0D1B3E'),
         C['smiles_mid'], lw=2.2)
label(ax, 5, Y_ENC+0.32, 'MoLFormer-XL', fs=11, bold=True, color=C['smiles_light'])
label(ax, 5, Y_ENC+0.03, '12-layer Transformer  (frozen)', fs=8, color=C['grey'])
label(ax, 5, Y_ENC-0.25, 'Last 12 layers unfrozen', fs=7.5, color='#93C5FD')

# ECFP projection block
grad_box(ax, 11, Y_ENC, 5.4, 1.2,
         hex_to_rgb('#1A3D1A'), hex_to_rgb('#0D2010'),
         C['ecfp_mid'], lw=2.2)
label(ax, 11, Y_ENC+0.32, 'ECFP Projection', fs=11, bold=True, color=C['ecfp_light'])
label(ax, 11, Y_ENC+0.03, 'Linear(2048→512)→LN→GELU', fs=7.5, color=C['grey'])
label(ax, 11, Y_ENC-0.25, 'Linear(512→768)→LN→GELU', fs=7.5, color=C['grey'])

# Descriptor projection block
grad_box(ax, 17, Y_ENC, 5.4, 1.2,
         hex_to_rgb('#3B1F5E'), hex_to_rgb('#1A0D2E'),
         C['desc_mid'], lw=2.2)
label(ax, 17, Y_ENC+0.32, 'Descriptor Projection', fs=11, bold=True, color=C['desc_light'])
label(ax, 17, Y_ENC+0.03, 'Linear(7→256)→LN→GELU', fs=7.5, color=C['grey'])
label(ax, 17, Y_ENC-0.25, 'Linear(256→512)→Linear(512→768)', fs=7.5, color=C['grey'])

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2 — SMILES pooling  (y ~ 21.3)
# ═══════════════════════════════════════════════════════════════════════════════
Y_POOL = 21.3

arrow(ax, 5,  Y_ENC-0.60, 5,  Y_POOL+0.53, C['smiles_light'])

grad_box(ax, 5, Y_POOL, 5.4, 1.1,
         hex_to_rgb('#163A5A'), hex_to_rgb('#0A1E30'),
         C['cls_mid'], lw=2)
label(ax, 5, Y_POOL+0.30, 'Hybrid SMILES Representation', fs=9.5, bold=True, color=C['cls_light'])
label(ax, 5, Y_POOL+0.02, 'h_s = (h_[CLS] + h_mean + h_max) / 3', fs=8.5, color=C['white'])
label(ax, 5, Y_POOL-0.28, 'h_s  ∈  ℝ⁷⁶⁸', fs=8, color=C['grey'])

# ECFP + Desc embeddings labels
arrow(ax, 11, Y_ENC-0.60, 11, Y_POOL+0.05, C['ecfp_light'])
arrow(ax, 17, Y_ENC-0.60, 17, Y_POOL+0.05, C['desc_light'])

box(ax, 11, Y_POOL, 5.0, 0.80, '#0F2010', C['ecfp_mid'], lw=1.5)
label(ax, 11, Y_POOL+0.12, 'e_ecfp  ∈  ℝ⁷⁶⁸', fs=9, bold=True, color=C['ecfp_light'])
label(ax, 11, Y_POOL-0.16, 'ECFP Embedding', fs=8, color=C['grey'])

box(ax, 17, Y_POOL, 5.0, 0.80, '#1A0D2E', C['desc_mid'], lw=1.5)
label(ax, 17, Y_POOL+0.12, 'e_desc  ∈  ℝ⁷⁶⁸', fs=9, bold=True, color=C['desc_light'])
label(ax, 17, Y_POOL-0.16, 'Descriptor Embedding', fs=8, color=C['grey'])

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Cross-Attention  (y ~ 19.2)
# ═══════════════════════════════════════════════════════════════════════════════
Y_ATT = 19.2

# Arrows into attention
arrow(ax, 5,  Y_POOL-0.55, 11, Y_ATT+0.65, C['smiles_light'], lw=1.4)
arrow(ax, 11, Y_POOL-0.40, 11, Y_ATT+0.65, C['ecfp_light'])
arrow(ax, 17, Y_POOL-0.40, 13, Y_ATT+0.65, C['desc_light'], lw=1.4)

grad_box(ax, 11, Y_ATT, 10.5, 1.4,
         hex_to_rgb('#5C3A00'), hex_to_rgb('#2C1A00'),
         C['attn_mid'], lw=2.5)
label(ax, 11, Y_ATT+0.42, 'Cross-Modal Multi-Head Attention', fs=11, bold=True, color=C['attn_light'])
label(ax, 11, Y_ATT+0.12,
      'Q = stack[e_ecfp, e_desc]  ∈  ℝᴮˣ²ˣ⁷⁶⁸', fs=8.5, color=C['white'])
label(ax, 11, Y_ATT-0.15,
      'K = V = h_s  ∈  ℝᴮˣ¹ˣ⁷⁶⁸   │   8 heads', fs=8.5, color=C['white'])
label(ax, 11, Y_ATT-0.42,
      'Attn(Q,K,V) = softmax(QKᵀ/√dₖ)·V', fs=8, color=C['grey'])

# QKV legend badges
for lbl, cx, col in [('Q', 7.5, C['ecfp_light']), ('K', 11, C['attn_light']), ('V', 14.5, C['desc_light'])]:
    box(ax, cx, Y_ATT+0.42+0.55, 0.6, 0.38, '#1C1C1C', col, lw=1.2)
    label(ax, cx, Y_ATT+0.42+0.55, lbl, fs=8, bold=True, color=col)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 4 — Residual + LayerNorm  (y ~ 17.3)
# ═══════════════════════════════════════════════════════════════════════════════
Y_RES = 17.3

arrow(ax, 8.5,  Y_ATT-0.70, 8.5,  Y_RES+0.55, C['ecfp_light'])
arrow(ax, 13.5, Y_ATT-0.70, 13.5, Y_RES+0.55, C['desc_light'])

box(ax, 8.5, Y_RES, 4.8, 1.1, '#0F2010', C['ecfp_mid'], lw=2)
label(ax, 8.5, Y_RES+0.25, 'Residual + LayerNorm', fs=9, bold=True, color=C['ecfp_light'])
label(ax, 8.5, Y_RES-0.10, 'ê_ecfp = LN(Attn_out[:,0] + e_ecfp)', fs=7.8, color=C['white'])

box(ax, 13.5, Y_RES, 4.8, 1.1, '#1A0D2E', C['desc_mid'], lw=2)
label(ax, 13.5, Y_RES+0.25, 'Residual + LayerNorm', fs=9, bold=True, color=C['desc_light'])
label(ax, 13.5, Y_RES-0.10, 'ê_desc = LN(Attn_out[:,1] + e_desc)', fs=7.8, color=C['white'])

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 5 — Gated Fusion  (y ~ 15.1)
# ═══════════════════════════════════════════════════════════════════════════════
Y_FUSE = 15.1

arrow(ax, 5,    Y_POOL-0.55, 5,    14.5, C['smiles_light'], lw=1.4)
ax.annotate('', xy=(11, Y_FUSE+0.80), xytext=(5, 14.5),
            arrowprops=dict(arrowstyle='->', color=C['smiles_light'],
                            lw=1.4, connectionstyle='arc3,rad=0.18'))
arrow(ax, 8.5,  Y_RES-0.55, 9.5,  Y_FUSE+0.80, C['ecfp_light'], lw=1.4)
arrow(ax, 13.5, Y_RES-0.55, 12.5, Y_FUSE+0.80, C['desc_light'], lw=1.4)

grad_box(ax, 11, Y_FUSE, 10.8, 1.75,
         hex_to_rgb('#5C0A0A'), hex_to_rgb('#2C0404'),
         C['gate_mid'], lw=2.8)
label(ax, 11, Y_FUSE+0.58, 'Gated Fusion Module', fs=12, bold=True, color=C['gate_light'])
label(ax, 11, Y_FUSE+0.22,
      'c = [h_s ; ê_ecfp ; ê_desc]  ∈  ℝ²³⁰⁴', fs=9, color=C['white'])
label(ax, 11, Y_FUSE-0.08,
      'g = σ(W_g · c + b_g)  ∈  ℝ⁷⁶⁸', fs=9, color=C['white'])
label(ax, 11, Y_FUSE-0.38,
      'f = g  ⊙  (W_f · c + b_f)  ∈  ℝ⁷⁶⁸', fs=9, color=C['white'])

# Gate icon
box(ax, 5.9, Y_FUSE, 1.1, 1.1, '#3D0000', C['gate_light'], lw=1.5, radius=0.15)
ax.text(5.9, Y_FUSE+0.18, 'σ', fontsize=18, color=C['gate_light'],
        ha='center', va='center', fontweight='bold', zorder=6)
ax.text(5.9, Y_FUSE-0.28, 'Gate', fontsize=7, color=C['grey'], ha='center', va='center', zorder=6)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 6 — Classifier  (y ~ 12.2)
# ═══════════════════════════════════════════════════════════════════════════════
Y_CLS_TOP = 13.3

arrow(ax, 11, Y_FUSE-0.875, 11, Y_CLS_TOP+1.0, C['gate_light'])

# Panel background
panel_h = 3.2
box(ax, 11, Y_CLS_TOP-panel_h/2+1.0, 10.5, panel_h+0.2,
    C['panel'], '#334155', lw=1.2, alpha=0.8)
label(ax, 11, Y_CLS_TOP+0.72, 'Deep Classifier Head', fs=11, bold=True, color=C['white'])

layer_data = [
    ('Linear(768→768) → LN → GELU → Dropout(0.3)', '768', '#1E3A5C', C['smiles_mid']),
    ('Linear(768→384) → LN → GELU → Dropout(0.3)', '384', '#1A3A1A', C['ecfp_mid']),
    ('Linear(384→192) → LN → GELU → Dropout(0.3)', '192', '#3B1A5C', C['desc_mid']),
    ('Linear(192→96)  → LN → GELU → Dropout(0.15)', '96',  '#5C3A00', C['attn_mid']),
]

ly = Y_CLS_TOP + 0.30
for i, (lbl, dim, fc, ec) in enumerate(layer_data):
    ly -= 0.68
    box(ax, 11, ly, 9.0, 0.52, fc, ec, lw=1.5, radius=0.12)
    label(ax, 10.0, ly, lbl, fs=7.8, color=C['white'])
    box(ax, 15.8, ly, 0.75, 0.40, '#0F0F0F', ec, lw=1.2, radius=0.08)
    label(ax, 15.8, ly, dim, fs=7.5, bold=True, color=ec)

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT  (y ~ 9.2)
# ═══════════════════════════════════════════════════════════════════════════════
Y_OUT = 9.2

arrow(ax, 11, ly - 0.26, 11, Y_OUT+0.58, C['white'])

box(ax, 11, Y_OUT+0.15, 4.0, 0.50, '#1C1C1C', '#64748B', lw=1.2, radius=0.10)
label(ax, 11, Y_OUT+0.15, 'Linear(96→1)', fs=9, color=C['grey'])

arrow(ax, 11, Y_OUT-0.12, 11, Y_OUT-0.60, C['white'])

grad_box(ax, 11, Y_OUT-0.95, 5.0, 0.75,
         hex_to_rgb('#1C3A1C'), hex_to_rgb('#0A1A0A'),
         '#22C55E', lw=2.5)
label(ax, 11, Y_OUT-0.95, 'Bioactivity Score  ŷ  ∈  (0, 1)', fs=10, bold=True, color='#86EFAC')

# BCEWithLogitsLoss note
label(ax, 11, Y_OUT-1.6, 'Loss: BCEWithLogitsLoss   ·   Threshold: ŷ > 0.5 → Active',
      fs=8, color=C['grey'])

# ═══════════════════════════════════════════════════════════════════════════════
# LEGEND  (bottom)
# ═══════════════════════════════════════════════════════════════════════════════
Y_LEG = 7.8
ax.plot([1, 21], [Y_LEG+0.45, Y_LEG+0.45], color='#334155', lw=1, zorder=2)

legend_items = [
    (C['smiles_mid'], 'SMILES / MoLFormer branch'),
    (C['ecfp_mid'],   'ECFP fingerprint branch'),
    (C['desc_mid'],   'Molecular descriptor branch'),
    (C['attn_mid'],   'Cross-modal attention'),
    (C['gate_mid'],   'Gated fusion'),
    (C['cls_mid'],    'Pooling / representation'),
]
cols = 3
for i, (col, txt) in enumerate(legend_items):
    cx = 2.8 + (i % cols) * 6.5
    cy = Y_LEG - (i // cols) * 0.58
    box(ax, cx-1.55, cy, 0.55, 0.38, col, col, lw=0, radius=0.06)
    label(ax, cx-0.65, cy, txt, fs=8.5, color=C['grey'], ha='left')

# ═══════════════════════════════════════════════════════════════════════════════
# Equation box (bottom-right)
# ═══════════════════════════════════════════════════════════════════════════════
Y_EQ = 6.5
box(ax, 16.5, Y_EQ, 8.5, 2.2, C['panel2'], '#475569', lw=1.2, alpha=0.9)
label(ax, 16.5, Y_EQ+0.78, 'Key Equations', fs=9, bold=True, color=C['white'])
eqs = [
    'h_s = (h_[CLS] + h_mean + h_max) / 3',
    'g = σ(W_g · [h_s ; ê_ecfp ; ê_desc])',
    'f = g ⊙ W_f · [h_s ; ê_ecfp ; ê_desc]',
]
for j, eq in enumerate(eqs):
    label(ax, 16.5, Y_EQ+0.30 - j*0.52, eq, fs=8, color='#E2E8F0')

# ═══════════════════════════════════════════════════════════════════════════════
# Hyper-param box (bottom-left)
# ═══════════════════════════════════════════════════════════════════════════════
box(ax, 5.5, Y_EQ, 8.5, 2.2, C['panel2'], '#475569', lw=1.2, alpha=0.9)
label(ax, 5.5, Y_EQ+0.78, 'Model Hyper-parameters', fs=9, bold=True, color=C['white'])
params = [
    'Attention heads: 8   |   Hidden dim: 768',
    'Dropout: 0.3   |   Classifier layers: 4',
    'Unfrozen MoLFormer layers: 12',
]
for j, p in enumerate(params):
    label(ax, 5.5, Y_EQ+0.28 - j*0.52, p, fs=8, color='#E2E8F0')

# ═══════════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(11, 0.35, 'FineTunedBERTaECFP  ·  Bioactivity Classification  ·  Gated Multi-Modal Fusion',
        fontsize=8, color='#475569', ha='center', va='center', zorder=6)

plt.tight_layout(pad=0.3)
output_path = r'E:\ML\BioActivity\classification\DL\FineTunedBERTaECFP_Architecture.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"Saved: {output_path}")
