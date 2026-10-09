"""Figures for the speed-up report (cost growth, speed-up by lever, accuracy against the expensive model)."""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nh3hia import paths
from analysis.validation import speedup_report as R
from analysis.validation.speedup_run import OUT

FIG = os.path.join(OUT, 'figures', 'report')
INK, GRID = '#1f2933', '#d8dde3'
C = {'cube': '#b5483a', 'hybrid': '#8a6d3b', 'r12': '#2b6cb0', 'r14': '#2f855a', 'fast': '#2b6cb0', 'old': '#9aa5b1'}
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
                     'ytick.color': INK, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False})


def cost_growth():
    c = pd.read_csv(os.path.join(paths.PRODUCTION, 'output_mesh_nlte', 'comparison.csv'))
    meshes = [('cube', 'Cube\n377 pts'), ('radial12x140', 'Radial 12×140\n935 pts'),
              ('hybrid', 'Hybrid\n1297 pts'), ('radial14x260', 'Radial 14×260\n1702 pts')]
    base = c[c.set == 'cube'].set_index('point').wall_s
    med, ros = [], []
    for s, _ in meshes:
        x = c[c.set == s].set_index('point').wall_s.reindex(base.index)
        med.append((x / base).median()); ros.append(x.sum() / base.sum())
    gold_h = pd.read_csv(paths.LUT_GOLD_CSV).wall_s.sum() / 3600.0          # 634 job-hours on the cube
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
    a = ax[0]
    cols = [C['cube'], C['r12'], C['hybrid'], C['r14']]
    w = [c[c.set == s].set_index('point').wall_s.reindex(base.index).median() for s, _ in meshes]
    a.bar(range(4), w, color=cols)
    for i, v in enumerate(w):
        a.text(i, v * 1.03, f"{v:,.0f} s\n{med[i]:.1f}× cube", ha='center', va='bottom', fontsize=9.5)
    a.set_xticks(range(4)); a.set_xticklabels([m[1] for m in meshes], fontsize=9.5)
    a.set_ylabel('median wall time per model [s]\n(11 nodes, original solver settings)'); a.set_ylim(0, max(w) * 1.3)
    a.set_title('One model, four seed meshes')
    b = ax[1]
    days = lambda ratio: gold_h * ratio / 3 / 24                         # 3 processes
    rows = [('Cube, current gold table', days(1), days(1), C['cube']),
            ('Radial 12×140, original settings', days(ros[1]), days(med[1]), C['r12']),
            ('Radial 14×260, original settings', days(ros[3]), days(med[3]), C['r14'])]
    for i, (lab, lo, hi, col) in enumerate(rows):
        b.barh(i, lo, color=col, alpha=0.9)
        if hi > lo * 1.02:
            b.barh(i, hi - lo, left=lo, color=col, alpha=0.35)
        b.text(max(lo, hi) + 1.2, i, f"{lo:.0f} d" if hi <= lo * 1.02 else f"{lo:.0f}–{hi:.0f} d", va='center', fontsize=10)
    b.barh(3, 44 / 24, color='#1f2933')
    b.text(44 / 24 + 1.2, 3, '1.8 d (92 % done, projected end)', va='center', fontsize=10)
    b.set_yticks(range(4)); b.set_yticklabels([r[0] for r in rows] + ['Radial 12×140, this work'], fontsize=10)
    b.invert_yaxis(); b.set_xlim(0, 125)
    b.set_xlabel('wall time for the full 2464-model table on 3 processes [days]')
    b.set_title('Rebuilding the whole table')
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'f1_cost_growth.png'), dpi=150); plt.close(fig)


def stages():
    V, G = R.load_variants(), R.load_gold()
    names = [('close', 'sum nearby\nlines only'), ('close-lev36', '+ 36 levels'), ('close-lev36-nq12', '+ nquads 12'),
             ('close-lev36-nq12-prec4', '+ stop at 1e-4')]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    sp, mx = [], []
    for nm, _ in names:
        c = R.compare(V[nm], G)
        sp.append((c.wall_ref / c.wall).values)
        c = c[c.ok_ref]
        mx.append(c[['d_R_01_MAIN', 'd_R_10_MAIN', 'd_R_12_MAIN', 'd_R_21_MAIN', 'd_R_22_MAIN']].abs().max(axis=1).values)
    a = ax[0]
    bp = a.boxplot(sp, positions=range(4), widths=0.55, showfliers=True, patch_artist=True,
                   medianprops=dict(color=INK, lw=2), flierprops=dict(marker='o', ms=3, mfc='none', mec=INK))
    for p in bp['boxes']:
        p.set_facecolor('#c6d8ee'); p.set_edgecolor(INK)
    for i, v in enumerate(sp):
        a.text(i, np.median(v) * 1.18, f"{np.median(v):.1f}×", ha='center', fontsize=10, fontweight='bold')
    a.set_yscale('log'); a.set_xticks(range(4)); a.set_xticklabels([n[1] for n in names], fontsize=9.5)
    a.set_ylabel('speed-up over the original solver\n(per node, 48 nodes)'); a.set_title('Cumulative speed-up'); a.grid(axis='y', color=GRID)
    b = ax[1]
    bp = b.boxplot(mx, positions=range(4), widths=0.55, showfliers=True, patch_artist=True,
                   medianprops=dict(color=INK, lw=2), flierprops=dict(marker='o', ms=3, mfc='none', mec=INK))
    for p in bp['boxes']:
        p.set_facecolor('#f0d9d5'); p.set_edgecolor(INK)
    b.axhline(0.5, color=C['cube'], ls='--', lw=1.2); b.text(3.45, 0.55, 'acceptance 0.5 %', color=C['cube'], ha='right', fontsize=9.5)
    b.set_yscale('symlog', linthresh=0.01); b.set_ylim(0, 6)
    b.set_xticks(range(4)); b.set_xticklabels([n[1] for n in names], fontsize=9.5)
    b.set_ylabel('largest ratio change per node [%]\n(max of the five satellite and (2,2) ratios)')
    b.set_title('Change in the answer, same nodes'); b.grid(axis='y', color=GRID)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'f2_stages.png'), dpi=150); plt.close(fig)


def accuracy():
    V, G = R.load_variants(), R.load_gold()
    lo = R.compare(V['close-lev36-nq12-prec4'], G); lo = lo[lo.log_N_dv <= 15.8 + 1e-9]
    hi = R.compare(V['close-nq12-prec4'], G)
    c = pd.concat([lo, hi]); c = c[c.ok_ref]
    cols = [('d_R_01_MAIN', r'$F_1\!:0\to1$'), ('d_R_10_MAIN', r'$F_1\!:1\to0$'), ('d_R_12_MAIN', r'$F_1\!:1\to2$'),
            ('d_R_21_MAIN', r'$F_1\!:2\to1$'), ('d_R_22_MAIN', '(2,2)/(1,1)'), ('d_R_21_11', '(2,1)/(1,1)'),
            ('d_A_MAIN', r'$T_B$(main)')]
    fig, a = plt.subplots(figsize=(11.5, 4.6))
    rng = np.random.default_rng(3)
    for i, (k, lab) in enumerate(cols):
        v = c[k].dropna().values
        a.scatter(np.full(len(v), i) + rng.uniform(-0.18, 0.18, len(v)), v, s=22, color=C['fast'], alpha=0.7, zorder=3)
        a.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, lw=2, zorder=4)
        a.text(i, 0.93, f"max {np.abs(v).max():.2f}%", ha='center', fontsize=9, transform=a.get_xaxis_transform())
    a.axhspan(-0.5, 0.5, color='#e6f2e8', zorder=0); a.axhline(0, color=INK, lw=0.6)
    a.set_xticks(range(len(cols))); a.set_xticklabels([x[1] for x in cols])
    a.set_ylim(-1.0, 1.0); a.set_ylabel('fast profile minus original solver [%]')
    a.set_title(f'{len(c)} validation nodes, final profile (green band = acceptance ±0.5 %; (2,1) ±1 % only where detected)')
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'f3_accuracy.png'), dpi=150); plt.close(fig)
    return c


def truncation():
    V, G = R.load_variants(), R.load_gold()
    c = R.compare(V['close-lev36'], G); c = c[c.ok_ref]
    fig, a = plt.subplots(figsize=(6.4, 4.2))
    mx = c[['d_R_01_MAIN', 'd_R_10_MAIN', 'd_R_12_MAIN', 'd_R_21_MAIN', 'd_A_MAIN']].abs().max(axis=1)
    sc = a.scatter(c.log_N_dv, mx.clip(lower=1e-3), c=c.T_cloud, cmap='viridis', s=36, edgecolor='white', lw=0.4)
    a.set_yscale('log'); a.axhline(0.5, color=C['cube'], ls='--', lw=1.2)
    a.set_xlabel(r'log $N/\Delta v$'); a.set_ylabel('largest change vs 48 levels [%]')
    a.set_title('36-level truncation fails at high column'); a.axvline(15.8, color='0.5', ls=':')
    a.text(15.82, 2.0e-3, 'safe region\nends here', fontsize=8.5, color='0.35', ha='left')
    plt.colorbar(sc, label=r'$T_k$ [K]'); plt.tight_layout(); plt.savefig(os.path.join(FIG, 'f4_truncation.png'), dpi=150); plt.close(fig)


if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    cost_growth(); stages(); accuracy(); truncation(); print('saved', FIG)
