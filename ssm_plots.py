#!/usr/bin/env python3
"""
Publication-quality SSM validation plots — Scapula Statistical Shape Model
Run:  python3 ssm_plots.py
Saves: ssm_compactness.pdf/png  ssm_scree.pdf/png
       ssm_generalization.pdf/png  ssm_modes_sigma.pdf/png
       ssm_validation_combined.pdf/png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Publication style (IEEE / Journal of Biomechanics / MedIA) ────────────────
matplotlib.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'font.size':          10,
    'axes.labelsize':     10,
    'axes.titlesize':     11,
    'axes.titleweight':   'bold',
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'axes.linewidth':     0.8,
    'lines.linewidth':    1.8,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'pdf.fonttype':       42,   # embed fonts in PDF
    'ps.fonttype':        42,
})

BLUE       = '#2166AC'
LIGHT_BLUE = '#4393C3'
RED        = '#D6604D'
ORANGE     = '#F4A582'
GREY       = '#888888'

# ── Raw data from SSM validation output ───────────────────────────────────────
N_SPECIMENS = 24
N_VERTICES  = 8001   # after decimation

# Compactness: (mode, cumulative variance %)
compactness_pts = np.array([
    [1,  46.60],
    [2,  56.96],
    [3,  64.07],
    [5,  74.31],
    [10, 89.47],
    [15, 96.00],
    [20, 99.26],
    [24, 100.00],
])

# Individual mode variance % and shape-space σ (mm) for modes 1–10
modes_10    = np.arange(1, 11)
indiv_var   = np.array([46.60, 10.36, 7.11, 5.57, 4.67,
                         4.59,  3.42, 2.64, 2.42, 2.08])
sigma_shape = np.array([283.258, 133.531, 110.633, 97.953, 89.688,
                          88.922,  76.716,  67.409,  64.612, 59.915])
# Average per-vertex displacement for 1 σ move along each mode
sigma_vertex = sigma_shape / np.sqrt(N_VERTICES)   # mm/vertex

# Generalization: leave-one-out reconstruction error per specimen (mm)
gen_errors = np.array([
    0.652, 0.529, 0.397, 0.404, 0.950,
    1.033, 0.998, 1.067, 0.953, 0.901,
    0.972, 0.888, 0.911, 0.828, 1.370,
    1.482, 0.796, 0.877, 0.811, 0.836,
    1.187, 1.098, 0.808, 0.775,
])
gen_mean = gen_errors.mean()
gen_std  = gen_errors.std()
gen_max  = gen_errors.max()

# Specificity
specificity_mean = 1.288   # mean over 50 random samples


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Compactness curve
# ══════════════════════════════════════════════════════════════════════════════
def plot_compactness(ax, fontsize_annot=8):
    mx = compactness_pts[:, 0]
    cv = compactness_pts[:, 1]

    # linear interpolation between known points for a clean curve
    modes_fine  = np.linspace(1, 24, 500)
    cumul_fine  = np.interp(modes_fine, mx, cv)

    ax.fill_between(modes_fine, cumul_fine, alpha=0.12, color=BLUE)
    ax.plot(modes_fine, cumul_fine, color=BLUE, linewidth=2, zorder=3, label='Cumulative variance')
    ax.scatter(mx, cv, color=BLUE, s=35, zorder=4)

    # threshold lines
    ax.axhline(90, color=RED,    linestyle='--', linewidth=1.0, label='90% (11 modes)')
    ax.axhline(95, color=ORANGE, linestyle=':',  linewidth=1.2, label='95% (14 modes)')
    ax.axvline(11, color=RED,    linestyle='--', linewidth=0.7, alpha=0.5)
    ax.axvline(14, color=ORANGE, linestyle=':',  linewidth=0.9, alpha=0.5)

    ax.annotate('11 modes\n@90%', xy=(11, 90), xytext=(12.5, 84),
                fontsize=fontsize_annot, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))
    ax.annotate('14 modes\n@95%', xy=(14, 95), xytext=(16, 89),
                fontsize=fontsize_annot, color='#B35806',
                arrowprops=dict(arrowstyle='->', color='#B35806', lw=0.8))

    ax.set_xlabel('Number of Modes')
    ax.set_ylabel('Cumulative Variance Explained (%)')
    ax.set_xlim(1, 24)
    ax.set_ylim(40, 103)
    ax.set_xticks([1, 5, 10, 15, 20, 24])
    ax.legend(frameon=False, loc='lower right', fontsize=fontsize_annot)
    ax.grid(axis='y', linewidth=0.4, alpha=0.35, color=GREY)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g%%'))


fig1, ax = plt.subplots(figsize=(5.5, 3.8))
plot_compactness(ax)
ax.set_title('SSM Compactness')
fig1.tight_layout()
fig1.savefig('ssm_compactness.pdf'); fig1.savefig('ssm_compactness.png')
print('[OK] ssm_compactness')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Scree plot
# ══════════════════════════════════════════════════════════════════════════════
def plot_scree(ax):
    colors = [BLUE if v >= 5 else LIGHT_BLUE for v in indiv_var]
    bars = ax.bar(modes_10, indiv_var, color=colors, edgecolor='white',
                  linewidth=0.5, width=0.75)
    for bar, val in zip(bars, indiv_var):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=7.5, color='#333333')
    ax.set_xlabel('Mode')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_xticks(modes_10)
    ax.set_xlim(0.4, 10.6)
    ax.grid(axis='y', linewidth=0.4, alpha=0.35, color=GREY)
    legend_elems = [
        mpatches.Patch(color=BLUE,       label='≥ 5%'),
        mpatches.Patch(color=LIGHT_BLUE, label='< 5%'),
    ]
    ax.legend(handles=legend_elems, frameon=False, fontsize=8)


fig2, ax = plt.subplots(figsize=(5.5, 3.8))
plot_scree(ax)
ax.set_title('Scree Plot — Individual Mode Variance')
fig2.tight_layout()
fig2.savefig('ssm_scree.pdf'); fig2.savefig('ssm_scree.png')
print('[OK] ssm_scree')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Generalization (LOO per specimen)
# ══════════════════════════════════════════════════════════════════════════════
def plot_generalization(ax, fontsize_annot=8):
    specimens = np.arange(1, N_SPECIMENS + 1)
    colors = [RED if e > 1.0 else LIGHT_BLUE for e in gen_errors]

    ax.bar(specimens, gen_errors, color=colors, edgecolor='white',
           linewidth=0.4, width=0.75, zorder=2)

    ax.axhline(gen_mean, color='black', linestyle='-', linewidth=1.8,
               label=f'Mean = {gen_mean:.3f} mm', zorder=3)
    ax.axhline(gen_mean + gen_std, color=GREY, linestyle='--', linewidth=1.0,
               label=f'Mean ± SD  (SD = {gen_std:.3f} mm)', zorder=3)
    ax.axhline(gen_mean - gen_std, color=GREY, linestyle='--', linewidth=1.0, zorder=3)
    ax.axhline(1.0, color=RED, linestyle=':', linewidth=1.0, alpha=0.7,
               label='1 mm reference', zorder=3)

    # Annotate max
    worst = int(np.argmax(gen_errors))
    ax.annotate(f'Max={gen_max:.3f} mm\n(sp.{worst+1})',
                xy=(worst + 1, gen_max), xytext=(worst - 3, gen_max + 0.05),
                fontsize=fontsize_annot, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))

    ax.set_xlabel('Specimen')
    ax.set_ylabel('LOO Reconstruction Error (mm)')
    ax.set_xticks(specimens)
    ax.set_xticklabels([str(i) for i in specimens], fontsize=7.5)
    ax.set_ylim(0, gen_max * 1.25)
    ax.grid(axis='y', linewidth=0.4, alpha=0.35, color=GREY)

    legend_elems = [
        mpatches.Patch(color=LIGHT_BLUE, label='< 1 mm'),
        mpatches.Patch(color=RED,        label='≥ 1 mm'),
        plt.Line2D([0],[0], color='black',  linewidth=1.8, label=f'Mean = {gen_mean:.3f} mm'),
        plt.Line2D([0],[0], color=GREY,     linestyle='--', linewidth=1.0,
                   label=f'± SD ({gen_std:.3f} mm)'),
        plt.Line2D([0],[0], color=RED,      linestyle=':',  linewidth=1.0, label='1 mm ref.'),
    ]
    ax.legend(handles=legend_elems, frameon=False, loc='upper left',
              ncol=2, fontsize=fontsize_annot)


fig3, ax = plt.subplots(figsize=(8.0, 3.8))
plot_generalization(ax)
ax.set_title('SSM Generalization — Leave-One-Out Reconstruction Error')
fig3.tight_layout()
fig3.savefig('ssm_generalization.pdf'); fig3.savefig('ssm_generalization.png')
print('[OK] ssm_generalization')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Mode shape amplitudes (σ per mode)
# ══════════════════════════════════════════════════════════════════════════════
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(9, 3.8))

# Left: shape-space σ (mm)
ax4a.bar(modes_10, sigma_shape, color=BLUE, edgecolor='white', linewidth=0.5, width=0.75)
ax4a.set_xlabel('Mode')
ax4a.set_ylabel('Shape-space σ (mm)')
ax4a.set_title('(a) Shape-Space Standard Deviation per Mode')
ax4a.set_xticks(modes_10)
ax4a.grid(axis='y', linewidth=0.4, alpha=0.35, color=GREY)
for i, (bar, val) in enumerate(zip(ax4a.patches, sigma_shape)):
    ax4a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
              f'{val:.0f}', ha='center', va='bottom', fontsize=7.5)

# Right: per-vertex mean displacement
ax4b.bar(modes_10, sigma_vertex, color=LIGHT_BLUE, edgecolor='white', linewidth=0.5, width=0.75)
ax4b.set_xlabel('Mode')
ax4b.set_ylabel('Mean Vertex Displacement per 1σ (mm)')
ax4b.set_title('(b) Per-Vertex Shape Variation per Mode')
ax4b.set_xticks(modes_10)
ax4b.grid(axis='y', linewidth=0.4, alpha=0.35, color=GREY)
for i, (bar, val) in enumerate(zip(ax4b.patches, sigma_vertex)):
    ax4b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
              f'{val:.2f}', ha='center', va='bottom', fontsize=7.5)

fig4.tight_layout()
fig4.savefig('ssm_modes_sigma.pdf'); fig4.savefig('ssm_modes_sigma.png')
print('[OK] ssm_modes_sigma')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Combined validation figure (research paper main figure)
# ══════════════════════════════════════════════════════════════════════════════
fig5 = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 3, figure=fig5, hspace=0.50, wspace=0.38)

# (a) Compactness
ax5a = fig5.add_subplot(gs[0, :2])
plot_compactness(ax5a, fontsize_annot=8)
ax5a.set_title('(a) Compactness')

# (b) Scree
ax5b = fig5.add_subplot(gs[0, 2])
plot_scree(ax5b)
ax5b.set_title('(b) Scree Plot')

# (c) Generalization
ax5c = fig5.add_subplot(gs[1, :2])
plot_generalization(ax5c, fontsize_annot=8)
ax5c.set_title('(c) Generalization (Leave-One-Out)')

# (d) Summary table
ax5d = fig5.add_subplot(gs[1, 2])
ax5d.axis('off')
ax5d.add_patch(mpatches.FancyBboxPatch(
    (0.03, 0.02), 0.94, 0.96,
    boxstyle='round,pad=0.01',
    facecolor='#F5F5F5', edgecolor='#CCCCCC',
    linewidth=0.8, transform=ax5d.transAxes, zorder=0))

rows = [
    ('Dataset', None),
    ('  Specimens (N)', '24'),
    ('  Subjects', '12 (paired L/R)'),
    ('  Vertices (ref)', '8,001'),
    ('  SSM rank', '24'),
    ('', None),
    ('Compactness', None),
    ('  Modes @ 90% var.', '11'),
    ('  Modes @ 95% var.', '14'),
    ('  Mode 1 variance', '46.60%'),
    ('', None),
    ('Generalization (LOO)', None),
    ('  Mean error', f'{gen_mean:.3f} mm'),
    ('  SD', f'{gen_std:.3f} mm'),
    ('  Max error', f'{gen_max:.3f} mm'),
    ('', None),
    ('Specificity', None),
    ('  Mean (50 samples)', f'{specificity_mean:.3f} mm'),
]

y = 0.96
for label, val in rows:
    if not label:
        y -= 0.032; continue
    is_header = (val is None)
    weight = 'bold' if is_header else 'normal'
    color  = BLUE    if is_header else '#222222'
    ax5d.text(0.06, y, label, transform=ax5d.transAxes,
              fontsize=9, fontweight=weight, color=color, va='top')
    if val is not None:
        ax5d.text(0.96, y, val, transform=ax5d.transAxes,
                  fontsize=9, ha='right', va='top', color='#222222')
    y -= 0.055 if is_header else 0.052

ax5d.set_title('(d) Validation Summary', pad=5)

fig5.savefig('ssm_validation_combined.pdf')
fig5.savefig('ssm_validation_combined.png', dpi=300)
print('[OK] ssm_validation_combined')

print('\nAll plots saved. Files:')
for f in ['ssm_compactness', 'ssm_scree', 'ssm_generalization',
          'ssm_modes_sigma', 'ssm_validation_combined']:
    print(f'  {f}.pdf   {f}.png')
