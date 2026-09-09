#!/usr/bin/env python3
"""
SSM Validation Plots — Publication Quality
Standard format for Journal of Biomechanics / Medical Image Analysis / Annals BME

Run:    python3 ssm_plots.py
Output: ssm_fig1_compactness.pdf/png
        ssm_fig2_generalization.pdf/png
        ssm_fig3_scree.pdf/png
        ssm_fig4_combined.pdf/png        ← main paper figure
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    # Font
    'font.family':         'sans-serif',
    'font.sans-serif':     ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size':           9,
    'axes.labelsize':      10,
    'axes.titlesize':      10,
    'axes.titleweight':    'bold',
    'axes.titlepad':       8,
    'xtick.labelsize':     8.5,
    'ytick.labelsize':     8.5,
    'legend.fontsize':     8.5,
    # Axes
    'axes.linewidth':      0.8,
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'axes.grid':           True,
    'grid.color':          '#E0E0E0',
    'grid.linewidth':      0.6,
    'grid.linestyle':      '-',
    'axes.axisbelow':      True,
    # Ticks
    'xtick.direction':     'out',
    'ytick.direction':     'out',
    'xtick.major.width':   0.8,
    'ytick.major.width':   0.8,
    'xtick.major.size':    4,
    'ytick.major.size':    4,
    # Lines
    'lines.linewidth':     1.8,
    # Save
    'savefig.dpi':         300,
    'savefig.bbox':        'tight',
    'savefig.pad_inches':  0.05,
    'pdf.fonttype':        42,
    'ps.fonttype':         42,
})

# Palette (colorblind-safe: blue / orange / grey)
C_BLUE   = '#1F77B4'
C_ORANGE = '#FF7F0E'
C_RED    = '#D62728'
C_GREY   = '#7F7F7F'
C_LBLUE  = '#AEC7E8'

# ── Data ──────────────────────────────────────────────────────────────────────
# Compactness: (mode index, cumulative variance %)
MODES_K  = np.array([1,  2,     3,     5,     10,    15,    20,    24])
CUMUL_V  = np.array([46.60, 56.96, 64.07, 74.31, 89.47, 96.00, 99.26, 100.0])

# Modes 1–10: individual variance %, shape-space σ (mm)
MODES_10   = np.arange(1, 11)
INDIV_V    = np.array([46.60, 10.36, 7.11, 5.57, 4.67,
                        4.59,  3.42,  2.64, 2.42, 2.08])
SIGMA_MM   = np.array([283.258, 133.531, 110.633, 97.953, 89.688,
                         88.922,  76.716,  67.409,  64.612, 59.915])
N_VERTS    = 8001
SIGMA_VTEX = SIGMA_MM / np.sqrt(N_VERTS)   # mean per-vertex displacement (mm)

# Generalization: LOO error per specimen (mm)
GEN = np.array([0.652, 0.529, 0.397, 0.404, 0.950,
                1.033, 0.998, 1.067, 0.953, 0.901,
                0.972, 0.888, 0.911, 0.828, 1.370,
                1.482, 0.796, 0.877, 0.811, 0.836,
                1.187, 1.098, 0.808, 0.775])

SPEC = 1.288   # specificity mean (mm, 50 samples)

N = len(GEN)
SPEC_IDS = np.arange(1, N + 1)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: smooth compactness curve via linear interpolation on dense grid
# ══════════════════════════════════════════════════════════════════════════════
_modes_dense  = np.arange(1, 25)
_cumul_dense  = np.interp(_modes_dense, MODES_K, CUMUL_V)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Compactness
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.0, 3.6))

ax.plot(_modes_dense, _cumul_dense,
        color=C_BLUE, linewidth=2.0, marker='o', markersize=4,
        markerfacecolor='white', markeredgewidth=1.4, zorder=4,
        label='Cumulative variance')

# Threshold reference lines
ax.axhline(90, color=C_RED,    linestyle='--', linewidth=1.1, zorder=3,
           label='90% (11 modes)')
ax.axhline(95, color=C_ORANGE, linestyle=':',  linewidth=1.3, zorder=3,
           label='95% (14 modes)')
ax.axvline(11, color=C_RED,    linestyle='--', linewidth=0.8, alpha=0.45, zorder=2)
ax.axvline(14, color=C_ORANGE, linestyle=':',  linewidth=1.0, alpha=0.45, zorder=2)

# Annotations
ax.text(11.3, 87.5, '11', fontsize=8, color=C_RED, va='top')
ax.text(14.3, 92.5, '14', fontsize=8, color=C_ORANGE, va='top')

ax.set_xlabel('Number of Modes')
ax.set_ylabel('Cumulative Variance Explained (%)')
ax.set_title('Compactness')
ax.set_xlim(0.5, 24.5)
ax.set_ylim(35, 104)
ax.set_xticks([1, 5, 10, 15, 20, 24])
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
ax.legend(frameon=False, loc='lower right')

plt.tight_layout()
plt.savefig('ssm_fig1_compactness.pdf')
plt.savefig('ssm_fig1_compactness.png')
plt.close()
print('[saved] ssm_fig1_compactness')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Generalization  (bar chart + box inset)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8),
                          gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.25})

# ── Left: per-specimen bars ──────────────────────────────────────────────────
ax = axes[0]
bar_colors = [C_RED if e > 1.0 else C_BLUE for e in GEN]
ax.bar(SPEC_IDS, GEN, color=bar_colors, edgecolor='white',
       linewidth=0.4, width=0.75, zorder=3)

mean_v = GEN.mean();  std_v = GEN.std()
ax.axhline(mean_v, color='black', linewidth=1.6, zorder=4,
           label=f'Mean = {mean_v:.3f} mm')
ax.axhline(mean_v + std_v, color=C_GREY, linestyle='--', linewidth=1.0, zorder=4,
           label=f'± SD ({std_v:.3f} mm)')
ax.axhline(mean_v - std_v, color=C_GREY, linestyle='--', linewidth=1.0, zorder=4)
ax.axhline(1.0, color=C_RED, linestyle=':', linewidth=1.0, alpha=0.7, zorder=4,
           label='1 mm reference')

ax.set_xlabel('Specimen')
ax.set_ylabel('Reconstruction Error (mm)')
ax.set_title('Generalization — Leave-One-Out Error')
ax.set_xticks(SPEC_IDS)
ax.set_xticklabels([str(i) for i in SPEC_IDS], fontsize=7.5)
ax.set_ylim(0, GEN.max() * 1.30)
ax.legend(frameon=False, loc='upper left', ncol=2, fontsize=8)

# Color legend patches
from matplotlib.patches import Patch
ax.legend(
    handles=[
        Patch(color=C_BLUE, label='Error < 1 mm'),
        Patch(color=C_RED,  label='Error ≥ 1 mm'),
        plt.Line2D([0],[0], color='black', lw=1.6, label=f'Mean = {mean_v:.3f} mm'),
        plt.Line2D([0],[0], color=C_GREY, lw=1.0, ls='--',
                   label=f'Mean ± SD  (SD={std_v:.3f} mm)'),
        plt.Line2D([0],[0], color=C_RED, lw=1.0, ls=':', label='1 mm reference'),
    ],
    frameon=False, loc='upper left', ncol=2, fontsize=8
)

# ── Right: box plot summary ───────────────────────────────────────────────────
ax2 = axes[1]
bp = ax2.boxplot(GEN, patch_artist=True, widths=0.45,
                 medianprops=dict(color='white', linewidth=2),
                 boxprops=dict(facecolor=C_LBLUE, edgecolor=C_BLUE, linewidth=1.0),
                 whiskerprops=dict(color=C_BLUE, linewidth=1.0),
                 capprops=dict(color=C_BLUE, linewidth=1.0),
                 flierprops=dict(marker='o', markerfacecolor=C_RED,
                                 markeredgecolor=C_RED, markersize=5))

ax2.axhline(mean_v, color='black', linewidth=1.4, linestyle='--', zorder=5,
            label=f'Mean={mean_v:.2f}')
ax2.set_ylabel('Reconstruction Error (mm)')
ax2.set_title('Distribution')
ax2.set_xticks([])
ax2.set_xlim(0.3, 1.7)

# Annotate statistics
stats_txt = (f'Mean={mean_v:.3f}\n'
             f'SD  ={std_v:.3f}\n'
             f'Max ={GEN.max():.3f}\n'
             f'Min ={GEN.min():.3f}')
ax2.text(1.55, GEN.max() * 1.05, stats_txt, fontsize=7.5,
         va='top', ha='right', color='#333333',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   edgecolor='#CCCCCC', linewidth=0.6))

plt.tight_layout()
plt.savefig('ssm_fig2_generalization.pdf')
plt.savefig('ssm_fig2_generalization.png')
plt.close()
print('[saved] ssm_fig2_generalization')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Scree + per-vertex displacement
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8),
                                gridspec_kw={'wspace': 0.35})

# Left: variance % per mode
ax1.bar(MODES_10, INDIV_V, color=C_BLUE, edgecolor='white',
        linewidth=0.5, width=0.72, zorder=3)
for i, (x, v) in enumerate(zip(MODES_10, INDIV_V)):
    ax1.text(x, v + 0.4, f'{v:.1f}%', ha='center', va='bottom',
             fontsize=7.5, color='#333333')
ax1.set_xlabel('Mode')
ax1.set_ylabel('Variance Explained (%)')
ax1.set_title('(a)  Scree Plot')
ax1.set_xticks(MODES_10)
ax1.set_ylim(0, INDIV_V[0] * 1.18)

# Right: per-vertex displacement per 1σ
ax2.bar(MODES_10, SIGMA_VTEX, color=C_ORANGE, edgecolor='white',
        linewidth=0.5, width=0.72, zorder=3)
for i, (x, v) in enumerate(zip(MODES_10, SIGMA_VTEX)):
    ax2.text(x, v + 0.01, f'{v:.2f}', ha='center', va='bottom',
             fontsize=7.5, color='#333333')
ax2.set_xlabel('Mode')
ax2.set_ylabel('Mean Vertex Displacement per 1σ (mm)')
ax2.set_title('(b)  Shape Variation Magnitude per Mode')
ax2.set_xticks(MODES_10)

plt.tight_layout()
plt.savefig('ssm_fig3_scree.pdf')
plt.savefig('ssm_fig3_scree.png')
plt.close()
print('[saved] ssm_fig3_scree')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Combined (main paper figure, 2 × 2)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(11.0, 7.8))
gs  = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

# ── (a) Compactness ──────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, :2])
ax_a.plot(_modes_dense, _cumul_dense,
          color=C_BLUE, linewidth=2.0, marker='o', markersize=4.5,
          markerfacecolor='white', markeredgewidth=1.4, zorder=4)
ax_a.fill_between(_modes_dense, _cumul_dense, 35, alpha=0.08, color=C_BLUE)
ax_a.axhline(90, color=C_RED,    linestyle='--', linewidth=1.1, zorder=3,
             label='90% variance threshold (11 modes)')
ax_a.axhline(95, color=C_ORANGE, linestyle=':',  linewidth=1.3, zorder=3,
             label='95% variance threshold (14 modes)')
ax_a.axvline(11, color=C_RED,    linestyle='--', linewidth=0.8, alpha=0.4, zorder=2)
ax_a.axvline(14, color=C_ORANGE, linestyle=':',  linewidth=1.0, alpha=0.4, zorder=2)
ax_a.text(11.4, 86.5, '11 modes', fontsize=8, color=C_RED)
ax_a.text(14.4, 91.8, '14 modes', fontsize=8, color=C_ORANGE)
ax_a.set_xlabel('Number of Modes')
ax_a.set_ylabel('Cumulative Variance Explained (%)')
ax_a.set_title('(a)  Compactness')
ax_a.set_xlim(0.5, 24.5)
ax_a.set_ylim(35, 106)
ax_a.set_xticks([1, 5, 10, 15, 20, 24])
ax_a.legend(frameon=False, loc='lower right', fontsize=8)

# ── (b) Scree ────────────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 2])
ax_b.bar(MODES_10, INDIV_V, color=C_BLUE, edgecolor='white',
         linewidth=0.4, width=0.72, zorder=3)
for x, v in zip(MODES_10, INDIV_V):
    ax_b.text(x, v + 0.4, f'{v:.1f}', ha='center', va='bottom', fontsize=7, color='#333333')
ax_b.set_xlabel('Mode')
ax_b.set_ylabel('Variance Explained (%)')
ax_b.set_title('(b)  Scree Plot')
ax_b.set_xticks(MODES_10)
ax_b.set_ylim(0, INDIV_V[0] * 1.22)

# ── (c) Generalization bars ───────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, :2])
bar_colors = [C_RED if e > 1.0 else C_BLUE for e in GEN]
ax_c.bar(SPEC_IDS, GEN, color=bar_colors, edgecolor='white',
         linewidth=0.3, width=0.75, zorder=3)
ax_c.axhline(mean_v, color='black', linewidth=1.6, zorder=4)
ax_c.axhline(mean_v + std_v, color=C_GREY, linestyle='--', linewidth=0.9, zorder=4)
ax_c.axhline(mean_v - std_v, color=C_GREY, linestyle='--', linewidth=0.9, zorder=4)
ax_c.axhline(1.0, color=C_RED, linestyle=':', linewidth=1.0, alpha=0.6, zorder=4)
ax_c.text(24.6, mean_v,       f'{mean_v:.2f}', fontsize=7.5, va='center', color='black')
ax_c.text(24.6, mean_v+std_v, f'+SD',           fontsize=7,   va='center', color=C_GREY)
ax_c.text(24.6, 1.0,          '1 mm',           fontsize=7,   va='center', color=C_RED)
ax_c.set_xlabel('Specimen')
ax_c.set_ylabel('LOO Reconstruction Error (mm)')
ax_c.set_title('(c)  Generalization — Leave-One-Out Error per Specimen')
ax_c.set_xticks(SPEC_IDS)
ax_c.set_xticklabels([str(i) for i in SPEC_IDS], fontsize=7.5)
ax_c.set_xlim(0.3, 25.5)
ax_c.set_ylim(0, GEN.max() * 1.30)
ax_c.legend(
    handles=[Patch(color=C_BLUE, label='< 1 mm'), Patch(color=C_RED, label='≥ 1 mm')],
    frameon=False, loc='upper left', fontsize=8
)

# ── (d) Summary panel ────────────────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 2])
ax_d.axis('off')

# Draw summary table manually
rows = [
    ('DATASET',           None,                True),
    ('Specimens (N)',      '24',               False),
    ('Subjects',          '12 paired L/R',     False),
    ('Vertices (ref)',    '8,001',             False),
    ('SSM rank',          '24',               False),
    ('',                  None,               False),
    ('COMPACTNESS',       None,                True),
    ('Mode 1 variance',   '46.60 %',          False),
    ('Modes @ 90 %',      '11',               False),
    ('Modes @ 95 %',      '14',               False),
    ('',                  None,               False),
    ('GENERALIZATION',    None,                True),
    ('Mean LOO error',    f'{mean_v:.3f} mm', False),
    ('SD',                f'{std_v:.3f} mm',  False),
    ('Max LOO error',     f'{GEN.max():.3f} mm', False),
    ('',                  None,               False),
    ('SPECIFICITY',       None,                True),
    ('Mean (50 samples)', f'{SPEC:.3f} mm',   False),
]

y = 0.97
dy_header = 0.062
dy_row    = 0.053
for label, val, is_hdr in rows:
    if not label:
        y -= 0.025; continue
    fw  = 'bold' if is_hdr else 'normal'
    clr = C_BLUE  if is_hdr else '#222222'
    ax_d.text(0.05, y, label, transform=ax_d.transAxes,
              fontsize=9 if is_hdr else 8.5,
              fontweight=fw, color=clr, va='top')
    if val:
        ax_d.text(0.97, y, val, transform=ax_d.transAxes,
                  fontsize=8.5, ha='right', va='top', color='#111111')
    y -= dy_header if is_hdr else dy_row

# Border
from matplotlib.patches import FancyBboxPatch
ax_d.add_patch(FancyBboxPatch(
    (0.02, 0.01), 0.96, 0.97,
    boxstyle='round,pad=0.01', transform=ax_d.transAxes,
    facecolor='#FAFAFA', edgecolor='#CCCCCC', linewidth=0.8, zorder=0))
ax_d.set_title('(d)  Validation Summary', pad=6)

plt.savefig('ssm_fig4_combined.pdf')
plt.savefig('ssm_fig4_combined.png', dpi=300)
plt.close()
print('[saved] ssm_fig4_combined')

print('\nDone. Output files:')
for f in ['ssm_fig1_compactness', 'ssm_fig2_generalization',
          'ssm_fig3_scree', 'ssm_fig4_combined']:
    print(f'  {f}.pdf   {f}.png')
