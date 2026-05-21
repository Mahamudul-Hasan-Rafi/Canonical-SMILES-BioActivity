import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 22))
ax.set_xlim(0, 18)
ax.set_ylim(0, 22)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ── Colour palette (muted, publication-friendly) ──────────────────────────────
BLUE   = '#2166AC'   # SMILES branch
GREEN  = '#1A7940'   # ECFP branch
PURPLE = '#6A3D9A'   # Descriptor branch
AMBER  = '#B45309'   # Attention
RED    = '#B91C1C'   # Gated fusion
TEAL   = '#0E7490'   # Pooling
SLATE  = '#374151'   # Classifier / neutral

BLUE_L   = '#DBEAFE'
GREEN_L  = '#DCFCE7'
PURPLE_L = '#EDE9FE'
AMBER_L  = '#FEF3C7'
RED_L    = '#FEE2E2'
TEAL_L   = '#CFFAFE'
SLATE_L  = '#F1F5F9'

# ── Helpers ───────────────────────────────────────────────────────────────────
def box(ax, cx, cy, w, h, fc, ec, lw=1.4, r=0.18, zorder=3, alpha=1.0):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle=f'round,pad=0,rounding_size={r}',
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       zorder=zorder, alpha=alpha)
    ax.add_patch(p)

def txt(ax, x, y, s, fs=9, color='#111827', bold=False, ha='center',
        va='center', zorder=5, style='normal'):
    ax.text(x, y, s, fontsize=fs, color=color,
            fontweight='bold' if bold else 'normal',
            fontstyle=style,
            ha=ha, va=va, zorder=zorder, fontfamily='DejaVu Sans')

def arrow_v(ax, x, y1, y2, color='#374151', lw=1.3):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                zorder=2)

def arrow_diag(ax, x1, y1, x2, y2, color='#374151', lw=1.3, rad=0.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=2)

def hline(ax, y, x1, x2, color='#D1D5DB', lw=0.8, ls='--'):
    ax.plot([x1, x2], [y, y], color=color, lw=lw, ls=ls, zorder=1)

def section_label(ax, x, y, txt_str):
    ax.text(x, y, txt_str, fontsize=7.5, color='#6B7280',
            ha='left', va='center', zorder=5, style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
txt(ax, 9, 21.55,
    'FineTunedBERTaECFP: Gated Multi-Modal Fusion for Bioactivity Prediction',
    fs=13, bold=True, color='#111827')
ax.plot([0.8, 17.2], [21.2, 21.2], color='#9CA3AF', lw=0.8)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Inputs  (y = 20.2)
# ══════════════════════════════════════════════════════════════════════════════
Y_IN = 20.2
# Left  (SMILES)
box(ax, 3.5, Y_IN, 5.2, 0.72, BLUE_L, BLUE, lw=1.6)
txt(ax, 3.5, Y_IN+0.13, 'SMILES Sequence', fs=9.5, bold=True, color=BLUE)
txt(ax, 3.5, Y_IN-0.18, '"CC(=O)Oc1ccccc1C(=O)O"', fs=7.5, color='#374151', style='italic')

# Centre (ECFP)
box(ax, 9, Y_IN, 5.2, 0.72, GREEN_L, GREEN, lw=1.6)
txt(ax, 9, Y_IN+0.13, 'ECFP4 Fingerprint', fs=9.5, bold=True, color=GREEN)
txt(ax, 9, Y_IN-0.18, r'f$_{ECFP}$  ∈  {0,1}$^{2048}$', fs=8, color='#374151')

# Right (Descriptors)
box(ax, 14.5, Y_IN, 5.2, 0.72, PURPLE_L, PURPLE, lw=1.6)
txt(ax, 14.5, Y_IN+0.13, 'Molecular Descriptors  (7)', fs=9.5, bold=True, color=PURPLE)
txt(ax, 14.5, Y_IN-0.18, 'MW, LogP, TPSA, HBD, HBA, RotBonds, RingCount', fs=7.5, color='#374151')

# ══════════════════════════════════════════════════════════════════════════════
# Arrows → Encoders
# ══════════════════════════════════════════════════════════════════════════════
for x in [3.5, 9, 14.5]:
    arrow_v(ax, x, Y_IN - 0.36, Y_IN - 0.84,
            color=BLUE if x==3.5 else (GREEN if x==9 else PURPLE))

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Encoders  (y = 18.9)
# ══════════════════════════════════════════════════════════════════════════════
Y_ENC = 18.9

# MoLFormer
box(ax, 3.5, Y_ENC, 5.4, 1.10, BLUE_L, BLUE, lw=1.8)
txt(ax, 3.5, Y_ENC+0.30, 'MoLFormer-XL', fs=10.5, bold=True, color=BLUE)
txt(ax, 3.5, Y_ENC+0.02, '12-layer Transformer  (ibm/MoLFormer-XL-both-10pct)', fs=7.5, color='#374151')
txt(ax, 3.5, Y_ENC-0.24, 'Last 12 layers unfrozen for fine-tuning', fs=7.5, color='#6B7280', style='italic')

# ECFP Projection
box(ax, 9, Y_ENC, 5.4, 1.10, GREEN_L, GREEN, lw=1.8)
txt(ax, 9, Y_ENC+0.30, 'ECFP Projection MLP', fs=10.5, bold=True, color=GREEN)
txt(ax, 9, Y_ENC+0.02, 'Linear(2048→512) → LN → GELU → Dropout', fs=7.5, color='#374151')
txt(ax, 9, Y_ENC-0.24, 'Linear(512→768)   → LN → GELU → Dropout', fs=7.5, color='#374151')

# Descriptor Projection
box(ax, 14.5, Y_ENC, 5.4, 1.10, PURPLE_L, PURPLE, lw=1.8)
txt(ax, 14.5, Y_ENC+0.30, 'Descriptor Projection MLP', fs=10.5, bold=True, color=PURPLE)
txt(ax, 14.5, Y_ENC+0.02, 'Linear(7→256) → LN → GELU', fs=7.5, color='#374151')
txt(ax, 14.5, Y_ENC-0.24, 'Linear(256→512) → Linear(512→768) → LN → GELU', fs=7.5, color='#374151')

for x in [3.5, 9, 14.5]:
    arrow_v(ax, x, Y_ENC - 0.55, Y_ENC - 1.00,
            color=BLUE if x==3.5 else (GREEN if x==9 else PURPLE))

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Intermediate embeddings  (y = 17.55)
# ══════════════════════════════════════════════════════════════════════════════
Y_EMB = 17.55

box(ax, 3.5, Y_EMB, 4.2, 0.62, TEAL_L, TEAL, lw=1.6)
txt(ax, 3.5, Y_EMB+0.12, 'Hybrid Pooling', fs=9, bold=True, color=TEAL)
txt(ax, 3.5, Y_EMB-0.14,
    r'$\mathbf{h}_s = \frac{1}{3}(\mathbf{h}_{[CLS]} + \mathbf{h}_{mean} + \mathbf{h}_{max})$',
    fs=8.5, color='#111827')

box(ax, 9, Y_EMB, 3.8, 0.62, GREEN_L, GREEN, lw=1.4)
txt(ax, 9, Y_EMB+0.10, r'$\mathbf{e}_{ECFP}  \in  \mathbb{R}^{768}$', fs=9, bold=True, color=GREEN)
txt(ax, 9, Y_EMB-0.16, 'ECFP embedding', fs=7.5, color='#6B7280')

box(ax, 14.5, Y_EMB, 3.8, 0.62, PURPLE_L, PURPLE, lw=1.4)
txt(ax, 14.5, Y_EMB+0.10, r'$\mathbf{e}_{desc}  \in  \mathbb{R}^{768}$', fs=9, bold=True, color=PURPLE)
txt(ax, 14.5, Y_EMB-0.16, 'Descriptor embedding', fs=7.5, color='#6B7280')

# ══════════════════════════════════════════════════════════════════════════════
# Arrows → Attention
# ══════════════════════════════════════════════════════════════════════════════
Y_ATT = 15.85

# SMILES feeds K,V
arrow_v(ax, 3.5, Y_EMB-0.31, Y_ATT+0.65, color=TEAL, lw=1.3)
# ECFP and Desc feed Q
arrow_v(ax, 9, Y_EMB-0.31, Y_ATT+0.65, color=GREEN, lw=1.3)
arrow_v(ax, 14.5, Y_EMB-0.31, Y_ATT+0.65, color=PURPLE, lw=1.3)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — Cross-modal Attention  (y = 15.85)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 9, Y_ATT, 15.6, 1.42, AMBER_L, AMBER, lw=2.0, r=0.22)
txt(ax, 9, Y_ATT+0.46, 'Cross-Modal Multi-Head Attention  (H = 8 heads)', fs=11, bold=True, color=AMBER)

# Q / K / V annotation
txt(ax, 4.5, Y_ATT+0.10, 'K = V = h_s  ∈  ℝ^(B×1×768)', fs=8.5, color='#374151')
txt(ax, 13.0, Y_ATT+0.10, 'Q = stack[e_ECFP, e_desc]  ∈  ℝ^(B×2×768)', fs=8.5, color='#374151')
txt(ax, 9, Y_ATT-0.20,
    r'$\mathbf{A} = \mathrm{softmax}\!\left(\dfrac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\!\mathbf{V}$,'
    r'  $d_k = 96$',
    fs=9, color='#111827')

# Q / K / V badges
for lbl, cx, col in [('K', 3.5, TEAL), ('Q', 9, GREEN), ('Q', 14.5, PURPLE)]:
    box(ax, cx, Y_ATT+0.46+0.60, 0.55, 0.32, 'white', col, lw=1.2, r=0.06)
    txt(ax, cx, Y_ATT+0.46+0.60, lbl, fs=8, bold=True, color=col)

arrow_v(ax, 9, Y_ATT - 0.71, Y_ATT - 1.14, color=AMBER)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 5 — Residual + LayerNorm  (y = 14.35)
# ══════════════════════════════════════════════════════════════════════════════
Y_RES = 14.35

box(ax, 9, Y_RES, 12.0, 1.10, '#FFF7ED', AMBER, lw=1.5)
txt(ax, 9, Y_RES+0.30, 'Residual Connection + Layer Normalisation', fs=10, bold=True, color=AMBER)
txt(ax, 6, Y_RES-0.12,
    r'$\hat{\mathbf{e}}_{ECFP} = \mathrm{LN}(\mathbf{A}[:,0] + \mathbf{e}_{ECFP})$',
    fs=9, color='#111827')
txt(ax, 12, Y_RES-0.12,
    r'$\hat{\mathbf{e}}_{desc}  = \mathrm{LN}(\mathbf{A}[:,1] + \mathbf{e}_{desc})$',
    fs=9, color='#111827')

# Small definition note
txt(ax, 9, Y_RES-0.36,
    r'$\hat{\mathbf{e}}$  =  attention-enriched embedding  (hat accent defined here; used throughout §4)',
    fs=7.5, color='#6B7280', style='italic')

# Arrows from h_s (staying left) and from res block down to fusion
arrow_diag(ax, 3.5, Y_EMB-0.31, 3.5, 12.70, color=TEAL, lw=1.2, rad=0.0)  # h_s bypass line
arrow_v(ax, 9, Y_RES - 0.55, Y_RES - 1.05, color=AMBER)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 6 — Gated Fusion  (y = 12.90)
# ══════════════════════════════════════════════════════════════════════════════
Y_FUS = 12.90

box(ax, 9, Y_FUS, 15.6, 2.10, RED_L, RED, lw=2.2, r=0.22)
txt(ax, 9, Y_FUS+0.72, 'Gated Multi-Modal Fusion', fs=12, bold=True, color=RED)

# Three equations side by side
txt(ax, 4.0, Y_FUS+0.25,
    r'$\mathbf{c} = [\mathbf{h}_s\,;\,\hat{\mathbf{e}}_{ECFP}\,;\,\hat{\mathbf{e}}_{desc}]$'
    '\n'
    r'$\mathbf{c} \in \mathbb{R}^{2304}$',
    fs=9, color='#111827')
txt(ax, 9, Y_FUS+0.25,
    r'$\mathbf{g} = \sigma(\mathbf{W}_g \mathbf{c} + \mathbf{b}_g)$'
    '\n'
    r'$\mathbf{g} \in [0,1]^{768}$',
    fs=9, color='#111827')
txt(ax, 14.0, Y_FUS+0.25,
    r'$\mathbf{f} = \mathbf{g} \odot (\mathbf{W}_f \mathbf{c} + \mathbf{b}_f)$'
    '\n'
    r'$\mathbf{f} \in \mathbb{R}^{768}$',
    fs=9, color='#111827')

# Dividers between the three equations
for xd in [6.2, 11.8]:
    ax.plot([xd, xd], [Y_FUS-0.38, Y_FUS+0.60], color='#FECACA', lw=0.8)

# Labels beneath equations
for lbl, cx in [('(1) Concatenation', 4.0), ('(2) Gate', 9), ('(3) Hadamard Product', 14.0)]:
    txt(ax, cx, Y_FUS-0.52, lbl, fs=7.5, color='#9CA3AF', style='italic')

# h_s arrow joining fusion from the left
ax.annotate('', xy=(1.2, Y_FUS), xytext=(3.5, Y_ATT+0.65),
            arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.3,
                            connectionstyle='arc3,rad=0.0'), zorder=2)
ax.plot([1.2, 1.2], [Y_FUS, Y_FUS], color=TEAL, lw=1.3, zorder=2)
ax.annotate('', xy=(1.2, Y_FUS), xytext=(1.2, 16.82),
            arrowprops=dict(arrowstyle='->', color=TEAL, lw=0,), zorder=1)
# Simpler: draw a bent line manually
ax.plot([3.5, 3.5], [Y_EMB-0.31, Y_FUS+0.05], color=TEAL, lw=1.3, zorder=2,
        linestyle='--', dashes=(4, 3))
ax.annotate('', xy=(1.2, Y_FUS), xytext=(3.5, Y_FUS),
            arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.3), zorder=2)
ax.plot([1.2, 1.2], [Y_FUS, Y_FUS+0.05], color=TEAL, lw=0, zorder=1)

txt(ax, 3.5, 15.2, r'$\mathbf{h}_s$ bypass', fs=7, color=TEAL, style='italic')

arrow_v(ax, 9, Y_FUS - 1.05, Y_FUS - 1.52, color=RED)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 7 — Classifier  (y = 11.0)
# ══════════════════════════════════════════════════════════════════════════════
Y_CLS = 11.0

box(ax, 9, Y_CLS, 15.6, 1.86, SLATE_L, SLATE, lw=1.8, r=0.20)
txt(ax, 9, Y_CLS+0.62, 'Deep Classifier Head  (4 layers)', fs=11, bold=True, color=SLATE)

# Layer pills
CLS_COLS = ['#1D4ED8', '#15803D', '#6D28D9', '#B45309']
CLS_DIMS  = ['768', '384', '192', '96']
CLS_W = [('Linear(768→768)', '+ LN + GELU + Drop(0.30)'),
         ('Linear(768→384)', '+ LN + GELU + Drop(0.30)'),
         ('Linear(384→192)', '+ LN + GELU + Drop(0.30)'),
         ('Linear(192→96)',  '+ LN + GELU + Drop(0.15)')]
xs = [2.0, 6.0, 10.0, 14.0]
for i, (xl, (la, lb)) in enumerate(zip(xs, CLS_W)):
    box(ax, xl, Y_CLS-0.05, 3.5, 0.78, 'white', CLS_COLS[i], lw=1.3, r=0.12)
    txt(ax, xl, Y_CLS+0.18, la, fs=8, bold=True, color=CLS_COLS[i])
    txt(ax, xl, Y_CLS-0.12, lb, fs=7.2, color='#374151')
    # Arrow between pills
    if i < 3:
        ax.annotate('', xy=(xs[i+1]-1.75, Y_CLS-0.05),
                    xytext=(xl+1.75, Y_CLS-0.05),
                    arrowprops=dict(arrowstyle='->', color='#9CA3AF', lw=1.1))

arrow_v(ax, 9, Y_CLS - 0.93, Y_CLS - 1.40, color=SLATE)

# ══════════════════════════════════════════════════════════════════════════════
# Output layer  (y = 9.35)
# ══════════════════════════════════════════════════════════════════════════════
Y_OUT = 9.35
box(ax, 9, Y_OUT, 4.5, 0.58, SLATE_L, SLATE, lw=1.2)
txt(ax, 9, Y_OUT, 'Linear(96 → 1)', fs=9, color=SLATE)

arrow_v(ax, 9, Y_OUT - 0.29, Y_OUT - 0.76, color=SLATE)

# ══════════════════════════════════════════════════════════════════════════════
# Final output box  (y = 8.28)
# ══════════════════════════════════════════════════════════════════════════════
Y_PRED = 8.28
box(ax, 9, Y_PRED, 6.5, 0.80, '#F0FFF4', '#15803D', lw=2.2, r=0.2)
txt(ax, 9, Y_PRED+0.12, r'Bioactivity Score  $\hat{y}$  ∈  (0, 1)', fs=11, bold=True, color='#15803D')
txt(ax, 9, Y_PRED-0.18, r'$\hat{y} > 0.5$  →  Active  |  BCEWithLogitsLoss', fs=8, color='#374151')

# ══════════════════════════════════════════════════════════════════════════════
# Summary info panel (bottom, clean and compact)
# ══════════════════════════════════════════════════════════════════════════════
Y_SUMM = 7.0
ax.plot([0.8, 17.2], [Y_SUMM+0.45, Y_SUMM+0.45], color='#D1D5DB', lw=0.8)

params = [
    ('MoLFormer-XL', 'ibm/MoLFormer-XL-both-10pct\nUnfrozen: last 12 layers', BLUE),
    ('Attention', 'H = 8 heads\nd_k = 96 per head\nDropout = 0.15', AMBER),
    ('Gated Fusion', 'W_g, W_f ∈ ℝ^(768×2304)\nGate: σ(W_g·c)\nOutput: 768-dim', RED),
    ('Classifier', '4-layer MLP\n768→768→384→192→96→1\nDropout: 0.30 / 0.15', SLATE),
]
col_w = 4.0
for i, (title, details, col) in enumerate(params):
    cx = 1.5 + i * col_w + col_w/2
    box(ax, cx, Y_SUMM - 0.40, col_w - 0.3, 1.55, 'white', col, lw=1.2, r=0.14)
    txt(ax, cx, Y_SUMM + 0.08, title, fs=9, bold=True, color=col)
    ax.plot([cx - col_w/2 + 0.15, cx + col_w/2 - 0.15],
            [Y_SUMM - 0.14, Y_SUMM - 0.14], color='#E5E7EB', lw=0.7)
    for j, line in enumerate(details.split('\n')):
        txt(ax, cx, Y_SUMM - 0.38 - j*0.33, line, fs=7.8, color='#374151')

# ── figure caption
txt(ax, 9, 0.35,
    'Figure 1.  Architecture of FineTunedBERTaECFP.  '
    'Three complementary molecular representations are encoded independently, '
    'cross-attended against the SMILES context, and integrated via gated fusion.',
    fs=8, color='#4B5563', style='italic')

plt.tight_layout(pad=0.5)
out = r'E:\ML\BioActivity\classification\DL\FineTunedBERTaECFP_Architecture_v2.png'
plt.savefig(out, dpi=220, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f'Saved: {out}')
