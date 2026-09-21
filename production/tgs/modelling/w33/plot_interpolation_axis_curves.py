"""Representative leave-one-out interpolation curves along the T_cloud and
log_n_H2 axes, at a few chosen slices -- same style as this project's earlier
per-axis visualizations (viz_interp_temp_axis.py / the log_Ndv-axis check):
true curve (black) vs LOO-predicted points (red x) with %-error annotated at
each held-out point, one panel per ratio.

Complements plot_interpolation_axis_completeness.py's pooled histograms by
showing what the error actually looks like on real curves, not just its
distribution.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_lut_gold import LOG_N_AXIS, T_AXIS, LOG_NDV_AXIS
from plot_interpolation_axis_completeness import load, PC_CM, DEFAULT_LUT

RATIOS = ['R_01_MAIN', 'R_10_MAIN', 'R_21_MAIN', 'R_12_MAIN']


def loo_curve(g, axis_col):
    """Given a slice (fixed other two axes) sorted along axis_col, return
    x, {ratio: true_y}, and per-interior-point LOO predictions with % error."""
    g = g.sort_values(axis_col)
    x = g[axis_col].values
    y = {r: g[r].values for r in RATIOS}
    amps = {a: np.log10(g[a].values.clip(min=1e-12)) for a in ['A_01', 'A_10', 'A_MAIN', 'A_21', 'A_12']}
    num_amp = {'R_01_MAIN': 'A_01', 'R_10_MAIN': 'A_10', 'R_21_MAIN': 'A_21', 'R_12_MAIN': 'A_12'}
    loo_x, loo_pred, loo_err = {r: [] for r in RATIOS}, {r: [] for r in RATIOS}, {r: [] for r in RATIOS}
    for k in range(1, len(x) - 1):
        xk = np.delete(x, k)
        pred_amp = {a: 10 ** np.interp(x[k], xk, np.delete(amps[a], k)) for a in amps}
        for r in RATIOS:
            pv = pred_amp[num_amp[r]] / pred_amp['A_MAIN']
            tv = y[r][k]
            loo_x[r].append(x[k])
            loo_pred[r].append(pv)
            loo_err[r].append(abs(pv - tv) / tv * 100 if tv > 1e-8 else np.nan)
    return x, y, loo_x, loo_pred, loo_err


def plot_slices(df, axis_col, slice_cols, slices, xlabel, title, out_path):
    n_rows = len(slices)
    fig, axes = plt.subplots(n_rows, len(RATIOS), figsize=(3.6 * len(RATIOS), 2.9 * n_rows), sharex='col')
    axes = np.atleast_2d(axes)
    for i, sv in enumerate(slices):
        mask = np.all([np.isclose(df[c], v) for c, v in zip(slice_cols, sv)], axis=0)
        g = df[mask]
        label = ', '.join(f'{c}={v:g}' for c, v in zip(slice_cols, sv))
        if len(g) < 3:
            for j in range(len(RATIOS)):
                axes[i, j].text(0.5, 0.5, f'insufficient data\n({len(g)} pts)\n{label}',
                                ha='center', va='center', transform=axes[i, j].transAxes, fontsize=8)
            continue
        x, y, loo_x, loo_pred, loo_err = loo_curve(g, axis_col)
        for j, r in enumerate(RATIOS):
            ax = axes[i, j]
            ax.plot(x, y[r], 'o-', color='black', markersize=6, zorder=3, label='true')
            if loo_x[r]:
                ax.scatter(loo_x[r], loo_pred[r], marker='x', s=80, color='crimson', zorder=4,
                           label='leave-one-out pred.')
                for xk, pk, ek, tk in zip(loo_x[r], loo_pred[r], loo_err[r], np.interp(loo_x[r], x, y[r])):
                    ax.annotate(f'{ek:.1f}%', (xk, max(pk, tk)), textcoords='offset points',
                                xytext=(0, 7), fontsize=7, ha='center', color='crimson')
            if i == 0:
                ax.set_title(r, fontsize=11)
            if j == 0:
                ax.set_ylabel(label, fontsize=9)
            if i == n_rows - 1:
                ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc='best')
    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(out_path, dpi=160)
    print('saved', out_path)


def pick_best_slices(df, axis_col, axis_vals, slice_cols, n=3):
    """Slices (fixed other-two-axis values) with the most points along axis_col,
    spread across the range of the FIRST slice_col for variety."""
    cov = df.groupby(slice_cols)[axis_col].nunique().reset_index(name='n')
    cov = cov.sort_values('n', ascending=False)
    full = cov[cov.n >= len(axis_vals) - 1]
    if len(full) == 0:
        full = cov
    full = full.sort_values(slice_cols[0])
    idx = np.linspace(0, len(full) - 1, n).round().astype(int)
    return [tuple(full.iloc[i][slice_cols]) for i in idx]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--lut', default=DEFAULT_LUT)
    ap.add_argument('--outdir', default='/home/yasho379/magritte_rebuilt/production/output_lut_gold/results/')
    a = ap.parse_args()

    df = load(a.lut)
    print(f"{len(df)} converged rows (radius mask: off)")

    T_slices = pick_best_slices(df, 'T_cloud', T_AXIS, ['log_n_H2', 'log_N_dv'], n=4)
    print('T-axis slices chosen (log_n_H2, log_N_dv):', T_slices)
    plot_slices(df, 'T_cloud', ['log_n_H2', 'log_N_dv'], T_slices, 'T_cloud [K]',
                'Gold LUT: leave-one-out along T_cloud, representative (log_n, log_Ndv) slices',
                os.path.join(a.outdir, 'interp_curves_Taxis.png'))

    n_slices = pick_best_slices(df, 'log_n_H2', LOG_N_AXIS, ['T_cloud', 'log_N_dv'], n=4)
    print('log_n-axis slices chosen (T_cloud, log_N_dv):', n_slices)
    plot_slices(df, 'log_n_H2', ['T_cloud', 'log_N_dv'], n_slices, 'log(n_H2)',
                'Gold LUT: leave-one-out along log(n_H2), representative (T, log_Ndv) slices',
                os.path.join(a.outdir, 'interp_curves_naxis.png'))


if __name__ == '__main__':
    main()
