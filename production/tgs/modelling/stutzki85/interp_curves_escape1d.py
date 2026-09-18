"""Representative leave-one-out interpolation curves for the escape1d coarse
grid, along the T_cloud and log_n_H2 axes at a few chosen slices -- same
style and purpose as w33/plot_interpolation_axis_curves.py: true curve
(black) vs LOO-predicted points (red x) with %-error annotated, one panel
per ratio. Complements interpolation_completeness_escape1d.py's pooled
histograms by showing what the error looks like on real curves, including
where a curve runs through or near the masing threshold.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_W33 = os.path.join(os.path.dirname(_HERE), 'w33')
sys.path.insert(0, _HERE)
sys.path.insert(0, _W33)
from build_lut_gold import LOG_N_AXIS, T_AXIS, LOG_NDV_AXIS
from build_escape_grid import out_csv_for, outdir_for
from interpolation_completeness_escape1d import load, AMPS, RATIOS as _NUM_RATIOS

RATIOS = [name for name, _ in _NUM_RATIOS]
NUM_AMP = dict(_NUM_RATIOS)


def loo_curve(g, axis_col):
    """Given a slice (fixed other two axes) sorted along axis_col, return
    x, {ratio: true_y}, and per-interior-point LOO predictions with % error."""
    g = g.sort_values(axis_col)
    x = g[axis_col].values
    y = {r: g[r].values for r in RATIOS}
    amps = {a: np.log10(g[a].values.clip(min=1e-12)) for a in AMPS}
    loo_x, loo_pred, loo_err = {r: [] for r in RATIOS}, {r: [] for r in RATIOS}, {r: [] for r in RATIOS}
    for k in range(1, len(x) - 1):
        xk = np.delete(x, k)
        pred_amp = {a: 10 ** np.interp(x[k], xk, np.delete(amps[a], k)) for a in amps}
        for r in RATIOS:
            pv = pred_amp[NUM_AMP[r]] / pred_amp['T_B_main']
            tv = y[r][k]
            loo_x[r].append(x[k])
            loo_pred[r].append(pv)
            loo_err[r].append(abs(pv - tv) / abs(tv) * 100 if abs(tv) > 1e-8 else np.nan)
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
        n_maser = int(g['any_maser'].sum())
        for j, r in enumerate(RATIOS):
            ax = axes[i, j]
            ax.plot(x, y[r], 'o-', color='black', markersize=6, zorder=3, label='true')
            if loo_x[r]:
                ax.scatter(loo_x[r], loo_pred[r], marker='x', s=80, color='crimson', zorder=4,
                           label='leave-one-out pred.')
                for xk, pk, ek, tk in zip(loo_x[r], loo_pred[r], loo_err[r], np.interp(loo_x[r], x, y[r])):
                    ax.annotate(f'{ek:.0f}%', (xk, max(pk, tk)), textcoords='offset points',
                                xytext=(0, 7), fontsize=7, ha='center', color='crimson')
            if i == 0:
                ax.set_title(r, fontsize=11)
            if j == 0:
                ax.set_ylabel(label + (f'\n({n_maser}/{len(g)} masing)' if n_maser else ''), fontsize=8)
            if i == n_rows - 1:
                ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc='best')
    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(out_path, dpi=160)
    print('saved', out_path)


def pick_best_slices(df, axis_col, axis_vals, slice_cols, n=4):
    """Slices (fixed other-two-axis values) with the most points along
    axis_col, spread across the range of the FIRST slice_col for variety."""
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
    ap.add_argument('--rates', choices=['loreau', 'original', 'full_original'], default='loreau')
    ap.add_argument('--grid', default=None, help='override the grid path implied by --rates')
    ap.add_argument('--outdir', default=None, help='override the output directory implied by --rates')
    a = ap.parse_args()

    grid_path = a.grid or out_csv_for(a.rates, fine=False)
    outdir = a.outdir or outdir_for(a.rates)

    df = load(grid_path)
    print(f"{len(df)} converged rows ({df['any_maser'].mean():.1%} masing)")

    T_slices = pick_best_slices(df, 'T_cloud', T_AXIS, ['log_n_H2', 'log_N_dv'], n=4)
    print('T-axis slices chosen (log_n_H2, log_N_dv):', T_slices)
    plot_slices(df, 'T_cloud', ['log_n_H2', 'log_N_dv'], T_slices, 'T_cloud [K]',
                f'escape1d ({a.rates} rates): leave-one-out along T_cloud, representative (log_n, log_Ndv) slices',
                os.path.join(outdir, f'interp_curves_escape1d_Taxis_{a.rates}.png'))

    n_slices = pick_best_slices(df, 'log_n_H2', LOG_N_AXIS, ['T_cloud', 'log_N_dv'], n=4)
    print('log_n-axis slices chosen (T_cloud, log_N_dv):', n_slices)
    plot_slices(df, 'log_n_H2', ['T_cloud', 'log_N_dv'], n_slices, 'log(n_H2)',
                f'escape1d ({a.rates} rates): leave-one-out along log(n_H2), representative (T, log_Ndv) slices',
                os.path.join(outdir, f'interp_curves_escape1d_naxis_{a.rates}.png'))


if __name__ == '__main__':
    main()
