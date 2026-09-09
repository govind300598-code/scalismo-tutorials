#!/usr/bin/env python3
"""
SSM Validation Plots — Publication Quality
Standard format for Journal of Biomechanics / Medical Image Analysis / Annals BME

Run:    python3 ssm_plots.py
Output:
  ssm_fig_validation_3panel.pdf/png   ← main paper figure (compactness + gen + spec)
  ssm_fig_metrics_table.pdf/png       ← per-specimen RMSE / HD95 / HD / Chamfer table
  ssm_fig_scree.pdf/png               ← scree plot (individual variance per mode)

NOTE: Update REGQ_* and GEN_* arrays below with values from the new terminal output
      after running the pipeline with the updated SSMValidation.scala.
      Run once and copy values from the [REGISTRATION QUALITY] and [GENERALIZATION]
      tables printed to stdout.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, FancyBboxPatch
import matplotlib.patheffects as pe

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
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
    'axes.linewidth':      0.8,
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'axes.grid':           True,
    'grid.color':          '#E0E0E0',
    'grid.linewidth':      0.6,
    'grid.linestyle':      '-',
    'axes.axisbelow':      True,
    'xtick.direction':     'out',
    'ytick.direction':     'out',
    'xtick.major.width':   0.8,
    'ytick.major.width':   0.8,
    'xtick.major.size':    4,
    'ytick.major.size':    4,
    'lines.linewidth':     1.8,
    'savefig.dpi':         300,
    'savefig.bbox':        'tight',
    'savefig.pad_inches':  0.05,
    'pdf.fonttype':        42,
    'ps.fonttype':         42,
})

C_BLUE   = '#1F77B4'
C_ORANGE = '#FF7F0E'
C_RED    = '#D62728'
C_GREY   = '#7F7F7F'
C_LBLUE  = '#AEC7E8'
C_GREEN  = '#2CA02C'

# ── DATA ─────────────────────────────────────────────────────────────────────
# Compactness
MODES_K  = np.array([1,  2,    3,    5,    10,   15,   20,   24])
CUMUL_V  = np.array([46.60, 56.96, 64.07, 74.31, 89.47, 96.00, 99.26, 100.0])

# Scree (individual variance % for first 10 modes)
MODES_10 = np.arange(1, 11)
INDIV_V  = np.array([46.60, 10.36, 7.11, 5.57, 4.67,
                      4.59,  3.42,  2.64, 2.42, 2.08])

# Generalization — LOO MSD per specimen (mm)
GEN_MSD  = np.array([0.652, 0.529, 0.397, 0.404, 0.950,
                      1.033, 0.998, 1.067, 0.953, 0.901,
                      0.972, 0.888, 0.911, 0.828, 1.370,
                      1.482, 0.796, 0.877, 0.811, 0.836,
                      1.187, 1.098, 0.808, 0.775])

# ── UPDATE THESE after re-running with new SSMValidation.scala ──────────────
# Copy values from "[REGISTRATION QUALITY]" table in terminal output
# (metrics between each registered mesh and the SSM mean shape)
REGQ_MSD  = np.array([0.000]*24)   # ← replace with actual values
REGQ_RMSE = np.array([0.000]*24)   # ← replace with actual values
REGQ_HD95 = np.array([0.000]*24)   # ← replace with actual values
REGQ_HD   = np.array([0.000]*24)   # ← replace with actual values

# Copy values from "[GENERALIZATION]" table (LOO full stats)
GEN_RMSE  = np.array([0.000]*24)   # ← replace with actual values
GEN_HD95  = np.array([0.000]*24)   # ← replace with actual values
GEN_HD    = np.array([0.000]*24)   # ← replace with actual values
# ─────────────────────────────────────────────────────────────────────────────

SPEC = 1.288   # specificity mean (mm, 50 samples)
SPEC_SD  = 0.0   # ← update with SD from terminal output

N = len(GEN_MSD)
SPEC_IDS = np.arange(1, N + 1)

_modes_dense = np.arange(1, 25)
_cumul_dense = np.interp(_modes_dense, MODES_K, CUMUL_V)


# ══════════════════════════════════════════════════════════════════════════════
# FIG A — 3-panel: Compactness + Generalization + Specificity
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(13.0, 4.4))
gs  = GridSpec(1, 3, figure=fig, wspace=0.38,
               left=0.06, right=0.97, top=0.88, bottom=0.16)

# ── Panel (a): Compactness ────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0])
ax_a.plot(_modes_dense, _cumul_dense,
          color=C_BLUE, linewidth=2.0, marker='o', markersize=4,
          markerfacecolor='white', markeredgewidth=1.5, zorder=4)
ax_a.fill_between(_modes_dense, _cumul_dense, 35, alpha=0.09, color=C_BLUE)

modes90 = int(_modes_dense[np.searchsorted(_cumul_dense, 90.0)])
modes95 = int(_modes_dense[np.searchsorted(_cumul_dense, 95.0)])

ax_a.axhline(90, color=C_RED,    linestyle='--', linewidth=1.1, zorder=3,
             label=f'90%  ({modes90} modes)')
ax_a.axhline(95, color=C_ORANGE, linestyle=':',  linewidth=1.3, zorder=3,
             label=f'95%  ({modes95} modes)')
ax_a.axvline(modes90, color=C_RED,    linestyle='--', linewidth=0.8, alpha=0.4)
ax_a.axvline(modes95, color=C_ORANGE, linestyle=':',  linewidth=1.0, alpha=0.4)

ax_a.text(modes90 + 0.4, 86.0, str(modes90), fontsize=8, color=C_RED, va='top')
ax_a.text(modes95 + 0.4, 91.5, str(modes95), fontsize=8, color=C_ORANGE, va='top')

ax_a.set_xlabel('Number of Modes', fontsize=10)
ax_a.set_ylabel('Cumulative Variance Explained (%)', fontsize=10)
ax_a.set_title('(a)  Compactness')
ax_a.set_xlim(0.5, 24.5)
ax_a.set_ylim(35, 106)
ax_a.set_xticks([1, 5, 10, 15, 20, 24])
ax_a.legend(frameon=False, loc='lower right', fontsize=8)

# ── Panel (b): Generalization — LOO reconstruction error ─────────────────────
ax_b = fig.add_subplot(gs[1])
mean_v = GEN_MSD.mean()
std_v  = GEN_MSD.std()

bar_colors = [C_RED if e > 1.0 else C_BLUE for e in GEN_MSD]
ax_b.bar(SPEC_IDS, GEN_MSD, color=bar_colors, edgecolor='white',
         linewidth=0.3, width=0.78, zorder=3)

ax_b.axhline(mean_v, color='black', linewidth=1.8, zorder=5)
ax_b.axhline(mean_v + std_v, color=C_GREY, linestyle='--', linewidth=1.0, zorder=4)
ax_b.axhline(mean_v - std_v, color=C_GREY, linestyle='--', linewidth=1.0, zorder=4)
ax_b.axhline(1.0, color=C_RED, linestyle=':', linewidth=1.0, alpha=0.65, zorder=4)

ax_b.text(N + 0.6, mean_v,       f'{mean_v:.2f}', fontsize=7.5, va='center')
ax_b.text(N + 0.6, mean_v+std_v, '+SD',            fontsize=7,   va='center', color=C_GREY)
ax_b.text(N + 0.6, 1.0,          '1mm',            fontsize=7,   va='center', color=C_RED)

ax_b.set_xlabel('Specimen', fontsize=10)
ax_b.set_ylabel('LOO Reconstruction Error (mm)', fontsize=10)
ax_b.set_title('(b)  Generalization')
ax_b.set_xticks(SPEC_IDS[::4])
ax_b.set_xlim(0.3, N + 1.8)
ax_b.set_ylim(0, GEN_MSD.max() * 1.32)

ax_b.legend(
    handles=[Patch(color=C_BLUE, label='< 1 mm'),
             Patch(color=C_RED,  label='≥ 1 mm')],
    frameon=False, loc='upper left', fontsize=8
)

# Stats box
stats_txt = (f'Mean = {mean_v:.3f} mm\n'
             f'SD   = {std_v:.3f} mm\n'
             f'Min  = {GEN_MSD.min():.3f} mm\n'
             f'Max  = {GEN_MSD.max():.3f} mm')
ax_b.text(0.97, 0.97, stats_txt, transform=ax_b.transAxes,
          fontsize=7.5, va='top', ha='right', color='#333333',
          bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                    edgecolor='#BBBBBB', linewidth=0.7))

# ── Panel (c): Specificity ───────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[2])

# Single bar + error bar
bar = ax_c.bar([1], [SPEC], yerr=[[0], [SPEC_SD]] if SPEC_SD > 0 else None,
               color=C_GREEN, edgecolor='white', width=0.5, zorder=3,
               error_kw=dict(ecolor='black', capsize=6, linewidth=1.2))

# Reference: typical registration error / clinical threshold
ax_c.axhline(1.0, color=C_RED, linestyle='--', linewidth=1.1, alpha=0.7,
             label='1 mm clinical threshold')
ax_c.axhline(GEN_MSD.mean(), color=C_BLUE, linestyle=':', linewidth=1.1, alpha=0.7,
             label=f'Gen. mean = {mean_v:.2f} mm')

ax_c.text(1, SPEC + max(SPEC_SD, 0.03) + 0.04,
          f'{SPEC:.3f} mm', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax_c.set_xticks([1])
ax_c.set_xticklabels(['Specificity\n(50 samples)'], fontsize=9)
ax_c.set_ylabel('Distance to Nearest Training Shape (mm)', fontsize=10)
ax_c.set_title('(c)  Specificity')
ax_c.set_xlim(0.3, 1.7)
ax_c.set_ylim(0, max(SPEC * 1.55, 1.6))
ax_c.legend(frameon=False, loc='upper right', fontsize=8)

# Annotation box
note = (f'N = 50 random\nSSM samples\nSpec > Gen:\n'
        f'{SPEC:.3f} > {mean_v:.3f} mm')
ax_c.text(0.04, 0.97, note, transform=ax_c.transAxes,
          fontsize=8, va='top', ha='left', color='#444444',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#F7F7F7',
                    edgecolor='#CCCCCC', linewidth=0.6))

plt.savefig('ssm_fig_validation_3panel.pdf')
plt.savefig('ssm_fig_validation_3panel.png', dpi=300)
plt.close()
print('[saved] ssm_fig_validation_3panel')


# ══════════════════════════════════════════════════════════════════════════════
# FIG B — Comprehensive surface-distance metrics table + per-specimen plot
# Shows: Chamfer (MSD), RMSE, HD95, Hausdorff for both Reg Quality and Gen LOO
# NOTE: REGQ_* and GEN_RMSE/HD95/HD arrays must be filled with real values first.
# ══════════════════════════════════════════════════════════════════════════════

# Check if we have real data or placeholders
has_regq = REGQ_MSD.sum() > 0
has_gen_full = GEN_RMSE.sum() > 0

if not has_regq and not has_gen_full:
    # ── Fallback: show only generalization MSD (what we have) ────────────────
    fig, ax = plt.subplots(figsize=(10.0, 5.0))

    metrics = {'Chamfer / MSD (mm)': GEN_MSD}
    x = np.arange(N)

    ax.bar(x + 1, GEN_MSD, color=C_BLUE, edgecolor='white',
           linewidth=0.3, width=0.7, zorder=3,
           label='Generalization LOO — Chamfer/MSD')
    ax.axhline(GEN_MSD.mean(), color='black', linewidth=1.5, zorder=5,
               label=f'Mean = {GEN_MSD.mean():.3f} mm')
    ax.axhline(1.0, color=C_RED, linestyle='--', linewidth=1.1, alpha=0.7,
               label='1 mm reference')

    ax.set_xlabel('Specimen')
    ax.set_ylabel('Surface Distance (mm)')
    ax.set_title('Surface Distance Metrics — Generalization LOO\n'
                 '(Re-run pipeline with updated SSMValidation.scala for full RMSE / HD95 / HD data)',
                 fontsize=9)
    ax.set_xticks(np.arange(1, N+1))
    ax.set_xticklabels([str(i) for i in range(1, N+1)], fontsize=8)
    ax.legend(frameon=False, loc='upper left', fontsize=8.5)
    ax.set_ylim(0, GEN_MSD.max() * 1.35)

    ax.text(0.99, 0.99,
            '⚠ RMSE, HD95, Hausdorff columns require\n'
            '   updated SSMValidation.scala output.\n'
            '   Fill REGQ_* and GEN_RMSE/HD95/HD arrays\n'
            '   at the top of this script after re-running.',
            transform=ax.transAxes, fontsize=8, va='top', ha='right',
            color='#B04000',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8F0',
                      edgecolor='#E0A060', linewidth=0.8))

    plt.tight_layout()
    plt.savefig('ssm_fig_metrics_table.pdf')
    plt.savefig('ssm_fig_metrics_table.png', dpi=300)
    plt.close()
    print('[saved] ssm_fig_metrics_table  (placeholder — re-run pipeline for full data)')

else:
    # ── Full metrics figure: 4-metric grouped bars per specimen ──────────────
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.0),
                              gridspec_kw={'hspace': 0.50, 'wspace': 0.30})

    def metric_panel(ax, regq, gen, label, ylim_factor=1.4,
                     ref_line=None, ref_label=None):
        x = np.arange(N)
        w = 0.38
        b1 = ax.bar(x + 1 - w/2, regq, width=w, color=C_BLUE,
                    edgecolor='white', linewidth=0.3, zorder=3,
                    label='Reg. Quality (vs mean)')
        b2 = ax.bar(x + 1 + w/2, gen,  width=w, color=C_ORANGE,
                    edgecolor='white', linewidth=0.3, zorder=3,
                    label='Generalization (LOO)')
        ax.axhline(regq.mean(), color=C_BLUE,   linestyle='--',
                   linewidth=1.2, alpha=0.7, zorder=4,
                   label=f'RegQ mean = {regq.mean():.3f}')
        ax.axhline(gen.mean(),  color=C_ORANGE, linestyle='--',
                   linewidth=1.2, alpha=0.7, zorder=4,
                   label=f'Gen mean  = {gen.mean():.3f}')
        if ref_line:
            ax.axhline(ref_line, color=C_RED, linestyle=':', linewidth=1.1,
                       alpha=0.65, zorder=4, label=ref_label)
        ax.set_xlabel('Specimen', fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticks(np.arange(1, N+1)[::4])
        all_vals = np.concatenate([regq, gen])
        ax.set_ylim(0, all_vals.max() * ylim_factor)
        ax.set_xlim(0.3, N + 0.7)
        ax.legend(frameon=False, loc='upper left', fontsize=7.5, ncol=2)

    metric_panel(axes[0, 0], REGQ_MSD,  GEN_MSD,  'Chamfer / MSD (mm)', ref_line=1.0, ref_label='1 mm ref')
    metric_panel(axes[0, 1], REGQ_RMSE, GEN_RMSE, 'RMSE (mm)',          ref_line=1.0, ref_label='1 mm ref')
    metric_panel(axes[1, 0], REGQ_HD95, GEN_HD95, 'HD95 (mm)')
    metric_panel(axes[1, 1], REGQ_HD,   GEN_HD,   'Hausdorff HD (mm)')

    titles = ['(a)  Chamfer / MSD — symmetric mean surface distance',
              '(b)  RMSE — root-mean-square surface error',
              '(c)  HD95 — 95th-percentile Hausdorff distance',
              '(d)  Hausdorff — maximum surface distance']
    for ax, t in zip(axes.flat, titles):
        ax.set_title(t, fontsize=9)

    fig.suptitle('Per-Specimen Surface-Distance Metrics\n'
                 'Registration Quality (vs SSM mean shape)  vs  Generalization (LOO)',
                 fontsize=11, fontweight='bold', y=1.01)

    plt.savefig('ssm_fig_metrics_table.pdf', bbox_inches='tight')
    plt.savefig('ssm_fig_metrics_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('[saved] ssm_fig_metrics_table  (full 4-metric figure)')


# ══════════════════════════════════════════════════════════════════════════════
# FIG C — Scree plot (individual variance per mode)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 3.8))

ax.bar(MODES_10, INDIV_V, color=C_BLUE, edgecolor='white',
       linewidth=0.5, width=0.72, zorder=3)
for x, v in zip(MODES_10, INDIV_V):
    ax.text(x, v + 0.5, f'{v:.1f}%', ha='center', va='bottom',
            fontsize=8, color='#333333')

ax.set_xlabel('Principal Mode')
ax.set_ylabel('Variance Explained (%)')
ax.set_title('Scree Plot — Individual Variance per Mode')
ax.set_xticks(MODES_10)
ax.set_ylim(0, INDIV_V[0] * 1.20)

# Add cumulative line on twin axis
ax2 = ax.twinx()
ax2.spines['top'].set_visible(False)
cumul10 = np.cumsum(INDIV_V)
ax2.plot(MODES_10, cumul10, color=C_ORANGE, linewidth=1.8,
         marker='s', markersize=4, markerfacecolor='white',
         markeredgewidth=1.2, zorder=5)
ax2.set_ylabel('Cumulative Variance (%)', color=C_ORANGE, fontsize=10)
ax2.tick_params(axis='y', labelcolor=C_ORANGE)
ax2.set_ylim(0, 115)
ax2.spines['right'].set_edgecolor(C_ORANGE)

from matplotlib.lines import Line2D
handles = [
    plt.Rectangle((0,0),1,1, color=C_BLUE,   label='Individual variance %'),
    Line2D([0],[0], color=C_ORANGE, marker='s', markersize=4,
           markerfacecolor='white', linewidth=1.8, label='Cumulative variance %'),
]
ax.legend(handles=handles, frameon=False, loc='upper right', fontsize=8.5)

plt.tight_layout()
plt.savefig('ssm_fig_scree.pdf')
plt.savefig('ssm_fig_scree.png', dpi=300)
plt.close()
print('[saved] ssm_fig_scree')


print('\nDone. Output files:')
for f in ['ssm_fig_validation_3panel', 'ssm_fig_metrics_table', 'ssm_fig_scree']:
    print(f'  {f}.pdf   {f}.png')
print()
print('To get the full metrics table, re-run the Scala pipeline then fill in')
print('REGQ_MSD, REGQ_RMSE, REGQ_HD95, REGQ_HD, GEN_RMSE, GEN_HD95, GEN_HD')
print('from the [REGISTRATION QUALITY] and [GENERALIZATION] tables in terminal output.')
