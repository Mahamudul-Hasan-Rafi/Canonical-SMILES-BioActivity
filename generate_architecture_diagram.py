#!/usr/bin/env python3
"""
Publication-quality architecture diagram for FineTunedBERTaECFP model.
Designed for Q1 journal submission (300 DPI, vector-ready).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size':   9,
    'figure.dpi':  150,
})

# Palette (accessible + print-safe)
CLR = {
    'smiles':   '#1A5276',   # dark navy
    'smiles_l': '#AED6F1',   # light blue
    'ecfp':     '#1D6A36',   # dark green
    'ecfp_l':   '#A9DFBF',   # light green
    'desc':     '#784212',   # dark amber
    'desc_l':   '#FAD7A0',   # light amber
    'attn':     '#4A235A',   # dark purple
    'attn_l':   '#D7BDE2',   # light purple
    'fuse':     '#7B241C',   # dark red
    'fuse_l':   '#F1948A',   # light red
    'cls':      '#0E6655',   # dark teal
    'cls_l':    '#A2D9CE',   # light teal
    'out':      '#515A5A',   # dark steel
    'out_l':    '#CCD1D1',   # light steel
    'bg':       '#F2F3F4',   # near-white background
    'arrow':    '#2C3E50',
    'skip':     '#7F8C8D',
    'white':    '#FFFFFF',
}

FIG_W, FIG_H = 20, 27


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def rbox(ax, cx, cy, w, h, lines,
         bg='#1A5276', fg='white',
         title_size=9.0, body_size=8.0,
         alpha=0.95, zorder=5,
         border_color=None):
    """Rounded rectangle with multiple text lines."""
    bc = border_color or CLR['white']
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.12',
        fc=bg, ec=bc, lw=2.2,
        alpha=alpha, zorder=zorder,
    )
    # Soft drop-shadow (offset white patch)
    shadow = FancyBboxPatch(
        (cx - w / 2 + 0.06, cy - h / 2 - 0.06), w, h,
        boxstyle='round,pad=0.12',
        fc='#B0B0B0', ec='none',
        alpha=0.25, zorder=zorder - 1,
    )
    ax.add_patch(shadow)
    ax.add_patch(patch)

    if isinstance(lines, str):
        lines = [lines]
    n = len(lines)
    spacing = min(h / (n + 0.6), 0.55)
    for i, line in enumerate(lines):
        dy = ((n - 1) / 2 - i) * spacing
        is_title = (i == 0)
        is_eq = any(c in line for c in ['∈', '→', '=', '√', 'ℝ', 'ᵀ', 'σ', '∑'])
        ax.text(
            cx, cy + dy, line,
            ha='center', va='center',
            fontsize=title_size if is_title else body_size,
            color=fg,
            weight='bold' if is_title else 'normal',
            style='italic' if (is_eq and not is_title) else 'normal',
            zorder=zorder + 1,
            multialignment='center',
        )


def arr(ax, x1, y1, x2, y2,
        color=None, lw=2.0,
        label='', ldx=0.18, ldy=0.0,
        rad=0.0, headsize=16, ls='-',
        zorder=6):
    """Annotated arrow."""
    color = color or CLR['arrow']
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='->, head_width=0.3, head_length=0.18',
            color=color, lw=lw, linestyle=ls,
            mutation_scale=headsize,
            connectionstyle=f'arc3,rad={rad}',
        ),
        zorder=zorder,
    )
    if label:
        mx = (x1 + x2) / 2 + ldx
        my = (y1 + y2) / 2 + ldy
        ax.text(mx, my, label, fontsize=7.5, color=color,
                style='italic', ha='left', va='center', zorder=zorder + 1,
                bbox=dict(fc=CLR['bg'], ec='none', pad=1.5, alpha=0.8))


def band(ax, cx, cy, w, h, color, alpha=0.08, label='', zorder=2):
    """Light background band to group rows."""
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.15',
        fc=color, ec=color, lw=0.0, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(p)
    if label:
        ax.text(0.4, cy, label,
                ha='left', va='center', fontsize=7,
                color=color, weight='bold', zorder=zorder + 1,
                rotation=90)


def dim_badge(ax, cx, cy, text, color):
    """Small dimension badge pill."""
    ax.text(cx, cy, text,
            ha='center', va='center', fontsize=6.8,
            color=CLR['white'], zorder=9,
            bbox=dict(fc=color, ec=CLR['white'],
                      boxstyle='round,pad=0.25', lw=1.2, alpha=0.92))


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
ax.set_facecolor(CLR['bg'])
fig.patch.set_facecolor(CLR['bg'])

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(FIG_W / 2, FIG_H - 0.55,
        'FineTunedBERTaECFP: Multimodal Bioactivity Prediction Architecture',
        ha='center', va='center', fontsize=14, color='#1A1A1A',
        weight='bold', zorder=10)
ax.text(FIG_W / 2, FIG_H - 1.10,
        'MoLFormer-XL (SMILES)  ⊕  ECFP Fingerprint  ⊕  Physicochemical Descriptors  →  Binary Bioactivity',
        ha='center', va='center', fontsize=9, color='#4D4D4D',
        style='italic', zorder=10)

# ── Column x-centres ─────────────────────────────────────────────────────────
xS = 3.8    # SMILES
xE = 9.8    # ECFP
xD = 16.2   # Descriptors
xC = 11.0   # convergence centre (post-attention, fusion, classifier)

# ── Row y-centres ─────────────────────────────────────────────────────────────
y_inp  = 24.5   # inputs
y_enc  = 22.0   # encoders
y_rep  = 19.6   # representations
y_qkv  = 18.5   # Q/K/V labels
y_attn = 16.8   # attention module centre
y_res  = 14.4   # post-attention residual
y_fuse = 12.2   # bilinear fusion
y_clf  = 9.5    # classifier
y_out  = 7.0    # output
y_leg  = 1.6    # legend

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND BANDS
# ─────────────────────────────────────────────────────────────────────────────
band(ax, FIG_W/2, y_inp,  FIG_W - 0.6, 1.5,  '#333333', alpha=0.05, label='INPUT')
band(ax, FIG_W/2, y_enc,  FIG_W - 0.6, 2.8,  '#333333', alpha=0.05, label='ENCODING')
band(ax, FIG_W/2, y_rep,  FIG_W - 0.6, 1.5,  '#333333', alpha=0.05, label='REPR.')
band(ax, FIG_W/2, y_attn, FIG_W - 0.6, 3.2,  CLR['attn'], alpha=0.08, label='CROSS-ATTN')
band(ax, FIG_W/2, y_res,  FIG_W - 0.6, 1.5,  CLR['attn'], alpha=0.06, label='RESIDUAL')
band(ax, FIG_W/2, y_fuse, FIG_W - 0.6, 1.5,  CLR['fuse'], alpha=0.08, label='FUSION')
band(ax, FIG_W/2, y_clf,  FIG_W - 0.6, 3.2,  CLR['cls'],  alpha=0.08, label='CLASSIF.')
band(ax, FIG_W/2, y_out,  FIG_W - 0.6, 1.4,  CLR['out'],  alpha=0.08, label='OUTPUT')

# ─────────────────────────────────────────────────────────────────────────────
# 1.  INPUT LAYER
# ─────────────────────────────────────────────────────────────────────────────
rbox(ax, xS, y_inp, 5.8, 0.9,
     ['SMILES Sequence', 'x = [x₁, x₂, …, xₙ]  |  Max Length = 512'],
     bg=CLR['smiles'], title_size=9.5, body_size=8)

rbox(ax, xE, y_inp, 5.0, 0.9,
     ['Morgan Fingerprint  (ECFP)', 'f ∈ {0,1}¹⁰²⁴,  radius r = 2'],
     bg=CLR['ecfp'], title_size=9.5, body_size=8)

rbox(ax, xD, y_inp, 5.0, 0.9,
     ['Physicochemical Descriptors', 'd ∈ ℝ⁷  (MW, LogP, TPSA, …)'],
     bg=CLR['desc'], title_size=9.5, body_size=8)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  ENCODING LAYER
# ─────────────────────────────────────────────────────────────────────────────
rbox(ax, xS, y_enc, 5.8, 2.3,
     ['MoLFormer-XL (IBM)',
      '12-Layer Transformer  |  d_model = 768',
      'Last 5 layers selectively fine-tuned',
      'H = {h₁, …, h₁₂}  ∈  ℝⁿˣ⁷⁶⁸'],
     bg=CLR['smiles'], title_size=9.5, body_size=8)

rbox(ax, xE, y_enc, 5.0, 2.3,
     ['ECFP Projector  (2-layer MLP)',
      'Linear(1024 → 512) + LN + GELU',
      'Dropout(0.15)',
      'Linear(512 → 768) + LN + GELU + Dropout(0.30)'],
     bg=CLR['ecfp'], title_size=9.5, body_size=8)

rbox(ax, xD, y_enc, 5.0, 2.3,
     ['Descriptor Projector  (3-layer MLP)',
      'Linear(7 → 256) + LN + GELU + Dropout(0.15)',
      'Linear(256 → 512) + LN + GELU + Dropout(0.15)',
      'Linear(512 → 768) + LN + GELU + Dropout(0.30)'],
     bg=CLR['desc'], title_size=9.5, body_size=8)

# arrows: inputs → encoders
arr(ax, xS, y_inp - 0.45, xS, y_enc + 1.15, color=CLR['smiles'])
arr(ax, xE, y_inp - 0.45, xE, y_enc + 1.15, color=CLR['ecfp'])
arr(ax, xD, y_inp - 0.45, xD, y_enc + 1.15, color=CLR['desc'])

# ─────────────────────────────────────────────────────────────────────────────
# 3.  REPRESENTATION LAYER
# ─────────────────────────────────────────────────────────────────────────────
rbox(ax, xS, y_rep, 5.8, 1.1,
     ['Pooled SMILES Embedding',
      'h_SMILES = (h_CLS + h_mean + h_max) / 3  ∈  ℝ⁷⁶⁸'],
     bg=CLR['smiles'], title_size=9, body_size=8)

rbox(ax, xE, y_rep, 5.0, 1.1,
     ['ECFP Embedding',
      'h_ECFP  ∈  ℝ⁷⁶⁸'],
     bg=CLR['ecfp'], title_size=9, body_size=8.5)

rbox(ax, xD, y_rep, 5.0, 1.1,
     ['Descriptor Embedding',
      'h_desc  ∈  ℝ⁷⁶⁸'],
     bg=CLR['desc'], title_size=9, body_size=8.5)

# arrows: encoders → representations
arr(ax, xS, y_enc - 1.15, xS, y_rep + 0.55, color=CLR['smiles'])
arr(ax, xE, y_enc - 1.15, xE, y_rep + 0.55, color=CLR['ecfp'])
arr(ax, xD, y_enc - 1.15, xD, y_rep + 0.55, color=CLR['desc'])

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Q / K / V LABELS + CROSS-MODAL ATTENTION MODULE
# ─────────────────────────────────────────────────────────────────────────────
# K/V label under SMILES
ax.text(xS, y_qkv, 'K = V = h_SMILES',
        ha='center', va='center', fontsize=8, style='italic',
        color=CLR['smiles'], weight='bold',
        bbox=dict(fc=CLR['smiles_l'], ec=CLR['smiles'], boxstyle='round,pad=0.25', lw=1.5, alpha=0.9))

# Q label (spanning ECFP + Desc columns)
ax.text((xE + xD) / 2, y_qkv,
        'Q = stack(h_ECFP, h_desc)  ∈  ℝᴮˣ²ˣ⁷⁶⁸',
        ha='center', va='center', fontsize=8, style='italic',
        color=CLR['attn'], weight='bold',
        bbox=dict(fc=CLR['attn_l'], ec=CLR['attn'], boxstyle='round,pad=0.25', lw=1.5, alpha=0.9))

# Attention module
rbox(ax, xC, y_attn, 17.0, 2.4,
     ['Cross-Modal Multi-Head Attention   (H = 8 heads,  d_model = 768,  d_k = 96)',
      'headᵢ = softmax( Q Wᵢᴿ · (K Wᵢᴷ)ᵀ / √d_k ) · V Wᵢᵛ          (i = 1, …, H)',
      'MultiHead(Q, K, V) = Concat(head₁, …, headₕ) Wᴼ   ∈  ℝᴮˣ²ˣ⁷⁶⁸',
      'Key insight: structural encodings (ECFP, desc) query the SMILES context'],
     bg=CLR['attn'], title_size=9.5, body_size=8.3)

# arrows: representations → attention
arr(ax, xS, y_rep - 0.55, xC - 7.5, y_attn + 1.2,
    color=CLR['smiles'], lw=2.2, rad=0.2, label='K, V', ldx=0.1, ldy=0.25)
arr(ax, xE, y_rep - 0.55, xC + 0.5, y_attn + 1.2,
    color=CLR['ecfp'], lw=2.2, rad=0.05, label='Q₁', ldx=0.1, ldy=0.0)
arr(ax, xD, y_rep - 0.55, xC + 4.0, y_attn + 1.2,
    color=CLR['desc'], lw=2.2, rad=-0.2, label='Q₂', ldx=0.1, ldy=0.0)

# arrow: rep labels → attention labels
arr(ax, xS, y_rep - 0.55, xS, y_qkv + 0.2, color=CLR['smiles'], lw=1.5)
arr(ax, xE, y_rep - 0.55, xE, y_qkv + 0.2, color=CLR['ecfp'], lw=1.5)
arr(ax, xD, y_rep - 0.55, xD, y_qkv + 0.2, color=CLR['desc'], lw=1.5)
arr(ax, xS, y_qkv - 0.2, xC - 7.5, y_attn + 1.2, color=CLR['smiles'], lw=1.8, rad=0.15)
arr(ax, xE, y_qkv - 0.2, xC + 0.0, y_attn + 1.2, color=CLR['ecfp'], lw=1.8)
arr(ax, xD, y_qkv - 0.2, xC + 4.5, y_attn + 1.2, color=CLR['desc'], lw=1.8, rad=-0.1)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  POST-ATTENTION RESIDUAL (LayerNorm + residual connections)
# ─────────────────────────────────────────────────────────────────────────────
rbox(ax, xC - 3.2, y_res, 6.2, 1.0,
     ['h_ECFP_attn  =  LayerNorm( h_ECFP + attn[:,0] )   ∈  ℝ⁷⁶⁸'],
     bg=CLR['ecfp'], title_size=8.5, body_size=8)

rbox(ax, xC + 3.6, y_res, 6.2, 1.0,
     ['h_desc_attn  =  LayerNorm( h_desc + attn[:,1] )   ∈  ℝ⁷⁶⁸'],
     bg=CLR['desc'], title_size=8.5, body_size=8)

# arrows: attention → residual boxes  (attended output)
arr(ax, xC - 0.5, y_attn - 1.2, xC - 3.2, y_res + 0.5,
    color=CLR['ecfp'], lw=2, rad=0.2, label='attn[:,0]', ldx=0.12, ldy=0.15)
arr(ax, xC + 0.5, y_attn - 1.2, xC + 3.6, y_res + 0.5,
    color=CLR['desc'], lw=2, rad=-0.2, label='attn[:,1]', ldx=0.12, ldy=0.15)

# skip-connection arrows (dashed): h_ECFP / h_desc direct to residual
arr(ax, xE, y_rep - 0.55, xC - 4.0, y_res + 0.5,
    color=CLR['ecfp'], lw=1.4, rad=0.35, ls='dashed',
    label='residual', ldx=0.1, ldy=0.2)
arr(ax, xD, y_rep - 0.55, xC + 5.5, y_res + 0.5,
    color=CLR['desc'], lw=1.4, rad=-0.35, ls='dashed',
    label='residual', ldx=0.1, ldy=0.2)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  GATED FUSION
# ─────────────────────────────────────────────────────────────────────────────
rbox(ax, xC, y_fuse, 15.0, 1.7,
     ['Gated Fusion Module   (W_g, W_f ∈ ℝ⁷⁶⁸ˣ²³⁰⁴)',
      'z = Concat( h_SMILES, h_ECFP_attn, h_desc_attn )   ∈  ℝ²³⁰⁴',
      'g = σ( W_g·z + b_g )  ∈  ℝ⁷⁶⁸,     h_fused = g ⊙ ( W_f·z + b_f )  ∈  ℝ⁷⁶⁸'],
     bg=CLR['fuse'], title_size=9.5, body_size=8.3)

# arrows: residual boxes → fusion
arr(ax, xC - 3.2, y_res - 0.5, xC - 2.5, y_fuse + 0.65,
    color=CLR['ecfp'], lw=2, rad=0.1)
arr(ax, xC + 3.6, y_res - 0.5, xC + 2.5, y_fuse + 0.65,
    color=CLR['desc'], lw=2, rad=-0.1)

# h_SMILES → fusion (skip connection from representation layer)
arr(ax, xS, y_rep - 0.55, xC - 6.5, y_fuse + 0.65,
    color=CLR['smiles'], lw=2.0, rad=0.28, ls='dashed',
    label='h_SMILES  (skip)', ldx=0.12, ldy=0.15)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  CLASSIFICATION HEAD
# ─────────────────────────────────────────────────────────────────────────────
rbox(ax, xC, y_clf, 15.0, 2.9,
     ['Classification Head   (N = 4 layers,  p_dropout = 0.30)',
      'Layer 1:  Linear(768 → 768) + LayerNorm + GELU + Dropout(0.30)',
      'Layer 2:  Linear(768 → 384) + LayerNorm + GELU + Dropout(0.30)',
      'Layer 3:  Linear(384 → 192) + LayerNorm + GELU + Dropout(0.30)',
      'Layer 4:  Linear(192 →  96) + LayerNorm + GELU + Dropout(0.15)',
      'Output:   Linear( 96 →   1)  →   ŷ  ∈  ℝ'],
     bg=CLR['cls'], title_size=9.5, body_size=8.3)

arr(ax, xC, y_fuse - 0.85, xC, y_clf + 1.4,
    color=CLR['fuse'], lw=2.2,
    label='h_fused  ∈  ℝ⁷⁶⁸', ldx=0.15, ldy=0)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
rbox(ax, xC, y_out, 10.5, 1.2,
     ['Bioactivity Prediction',
      'P(active | molecule) = σ(ŷ)  ∈  [0, 1]   →   Threshold: 0.5'],
     bg=CLR['out'], fg='white', title_size=9.5, body_size=8.5)

arr(ax, xC, y_clf - 1.45, xC, y_out + 0.6,
    color=CLR['cls'], lw=2.2)

# Performance badge
rbox(ax, xC + 7.2, y_out, 4.5, 1.2,
     ['Best Performance',
      'AUROC: 95.39%   |   Bal. Acc.: 92.01%'],
     bg='#1C2833', fg='white', title_size=8.8, body_size=8)

# ─────────────────────────────────────────────────────────────────────────────
# 9.  DIMENSION BADGES
# ─────────────────────────────────────────────────────────────────────────────
dim_badge(ax, xS + 0.8, (y_rep + y_attn) / 2, '768-d', CLR['smiles'])
dim_badge(ax, xE + 0.8, y_enc - 0.5,           '768-d', CLR['ecfp'])
dim_badge(ax, xD + 0.8, y_enc - 0.5,           '768-d', CLR['desc'])
dim_badge(ax, xC + 1.0, y_fuse - 0.9,          '768-d', CLR['fuse'])

# ─────────────────────────────────────────────────────────────────────────────
# 10.  LEGEND
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    (CLR['smiles'], 'SMILES / MoLFormer-XL'),
    (CLR['ecfp'],   'ECFP Fingerprint Stream'),
    (CLR['desc'],   'Descriptor Stream'),
    (CLR['attn'],   'Cross-Modal Attention'),
    (CLR['fuse'],   'Gated Fusion'),
    (CLR['cls'],    'Classification Head'),
]

leg_x0 = 0.8
leg_y  = y_leg + 0.4
ax.text(FIG_W / 2, y_leg + 1.2,
        'Module Colour Legend', ha='center', va='center',
        fontsize=9, color='#2C3E50', weight='bold')

for i, (c, lab) in enumerate(legend_items):
    cx = leg_x0 + i * 3.25
    patch = FancyBboxPatch(
        (cx, leg_y - 0.25), 0.7, 0.5,
        boxstyle='round,pad=0.05', fc=c, ec='white', lw=1.4, alpha=0.93, zorder=6)
    ax.add_patch(patch)
    ax.text(cx + 0.85, leg_y, lab, fontsize=7.8,
            ha='left', va='center', color='#2C3E50', zorder=7)

# Training footnote
ax.text(FIG_W / 2, y_leg - 0.5,
        'Training: RMSprop  +  OneCycleLR  |  Regularisation: Dropout(0.30), '
        'Weight Decay(1.24×10⁻⁴), Grad. Clipping(1.0), Grad. Accum.(×2)  '
        '|  Optuna TPE  (50 trials)',
        ha='center', va='center', fontsize=7.5,
        color='#566573', style='italic', zorder=7)

# dashed = skip connection note
ax.text(FIG_W - 0.4, (y_rep + y_fuse) / 2,
        '╌╌  skip\n      connection',
        ha='right', va='center', fontsize=7, color=CLR['skip'], style='italic')

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.1)
OUT_PNG = r'E:\ML\BioActivity\architecture_diagram.png'
OUT_PDF = r'E:\ML\BioActivity\architecture_diagram.pdf'
plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor=CLR['bg'])
plt.savefig(OUT_PDF, bbox_inches='tight', facecolor=CLR['bg'])
print(f"Saved:\n  {OUT_PNG}\n  {OUT_PDF}")
