"""Reproduce Stutzki & Winnewisser (1985) Fig. 4a-c using the 1D
escape-probability model's own coarse grid (nh3hia/escape1d/grid.py, no
--fine -- the exact gold LUT axes, so 18/24/30/36 K land exactly on the
grid, matching analysis/figures/fig4_magritte.py's temperature choice).

This is the most literal possible reproduction of Fig. 4: same physical
method (escape-probability closure) as Stutzki's own, only the collision
rates are updated (Loreau et al. 2023) and the hyperfine redistribution is
exact rather than statistical.

guard_masers=False means some grid points have extreme or negative outer
T_B values near the masing threshold (see magritte-quirks-2026-09-18.md
Sec 4.2); griddata's linear interpolation will show this as streaking or
saturated colour near those points rather than smooth contours -- left
visible deliberately rather than clipped away, since where the panels look
noisy is itself informative about where the closure is unstable.

Usage: python3 -m analysis.figures.fig4_escape1d [--grid path] [--temps 18,24,30,36]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from nh3hia.escape1d.grid import out_csv_for, outdir_for

FIG4_TEMPS = [18.0, 24.0, 30.0, 36.0]

# (column, panel title, cmap) -- matches Fig. 4a (outer), 4b (inner), 4c ((2,2)/(1,1))
PANELS = [
    ('R_01', r'$T_B(F_1=0\to1)/T_B(\Delta F_1=0)$ [outer]', 'viridis'),
    ('R_10', r'$T_B(F_1=1\to0)/T_B(\Delta F_1=0)$ [outer]', 'viridis'),
    ('R_12', r'$T_B(F_1=1\to2)/T_B(\Delta F_1=0)$ [inner]', 'plasma'),
    ('R_21', r'$T_B(F_1=2\to1)/T_B(\Delta F_1=0)$ [inner]', 'plasma'),
    ('R_2211', r'$T_B(2,2;\Delta F_1=0)/T_B(1,1;\Delta F_1=0)$', 'cividis'),
]


def load_grid(path):
    df = pd.read_csv(path)
    df = df[df['converged']]
    return df


def contour_panel(fig, ax, df_T, col, title, cmap, is_ratio=True, vmin=None, vmax=None,
                   draw_colorbar=True):
    """2nd-98th percentile colour clipping, matching the house convention
    from reproduce_raw_fig4_full.py: clipped BEFORE griddata interpolation
    (not just at display time), so an extreme outlier near the masing
    threshold can't drag the interpolated field's gradient toward it between
    itself and its neighbours -- only the clip boundary shows a step. Ratio
    panels always include the LTE/tau=0 floor (0.0) in the colour range even
    if the 2nd percentile sits above it, so the no-anomaly baseline stays
    visible.

    By default each panel gets its own colorbar (own percentile range,
    computed from that panel's own data). Pass explicit vmin/vmax (e.g. a
    percentile range pooled across a whole figure's panels) to put multiple
    panels on one shared colour scale instead -- draw_colorbar=False then
    skips this panel's own colorbar so the caller can add one shared one."""
    x = df_T['log_n_H2'].values
    y = df_T['log_N_dv'].values
    raw = df_T[col].values
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(raw)
    x, y, raw = x[valid], y[valid], raw[valid]
    if len(x) < 8:
        ax.text(0.5, 0.5, f'insufficient data\n({len(x)} points)',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        return
    if vmin is None or vmax is None:
        lo, hi = np.percentile(raw, [2, 98])
        if is_ratio:
            lo = min(lo, 0.0)
    else:
        lo, hi = vmin, vmax
    z = np.clip(raw, lo, hi)
    gx = np.linspace(x.min(), x.max(), 200)
    gy = np.linspace(y.min(), y.max(), 200)
    GX, GY = np.meshgrid(gx, gy)
    GZ = griddata(np.column_stack([x, y]), z, (GX, GY), method='linear')
    cf = ax.contourf(GX, GY, GZ, levels=20, cmap=cmap, vmin=lo, vmax=hi)
    cs = ax.contour(GX, GY, GZ, levels=10, colors='k', linewidths=0.4, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')
    ax.scatter(x, y, s=4, c='k', alpha=0.15)
    if draw_colorbar:
        cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=6)
    n_maser = int(df_T['any_maser'].values[valid].sum())
    ax.set_title(title + (f'\n({n_maser}/{len(x)} masing)' if n_maser else ''), fontsize=9)
    return cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rates', choices=['loreau', 'full_original'], default='loreau')
    ap.add_argument('--grid', default=None, help='override the grid path implied by --rates')
    ap.add_argument('--out', default=None, help='override the output path implied by --rates')
    ap.add_argument('--temps', default=None, help='comma-separated T_cloud list; default is the 4 Stutzki panel temps')
    a = ap.parse_args()

    global FIG4_TEMPS
    if a.temps:
        FIG4_TEMPS = [float(x) for x in a.temps.split(',')]

    grid_path = a.grid or out_csv_for(a.rates, fine=False)
    out_path = a.out or os.path.join(outdir_for(a.rates), f'fig4_reproduction_escape1d_{a.rates}.png')
    a.out = out_path
    df = load_grid(grid_path)
    print(f"{len(df)} converged rows ({df['any_maser'].mean():.1%} masing)")

    available_T = [T for T in FIG4_TEMPS if len(df[np.isclose(df['T_cloud'], T)]) >= 8]
    if not available_T:
        raise SystemExit(f"No requested temperature has >=8 converged rows.")
    print(f"Temperatures with enough data: {available_T}")

    n_rows = len(FIG4_TEMPS)
    n_cols = len(PANELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows),
                              sharex=True, sharey=True)

    for i, T in enumerate(FIG4_TEMPS):
        df_T = df[np.isclose(df['T_cloud'], T)]
        for j, (col, title, cmap) in enumerate(PANELS):
            ax = axes[i, j]
            contour_panel(fig, ax, df_T, col, title if i == 0 else '', cmap, is_ratio=True)
            if j == 0:
                ax.set_ylabel(f'T={T:.0f}K\n' + r'log($N_{NH_3}/\Delta v$)', fontsize=8)
            if i == n_rows - 1:
                ax.set_xlabel(r'log($n_{H_2}$)', fontsize=8)

    plt.suptitle('Reproduction of Stutzki & Winnewisser (1985) Fig. 4a-c\n'
                  '1D escape-probability model, Loreau et al. (2023) rates, guard_masers=False',
                  fontsize=12)
    plt.tight_layout()
    plt.savefig(a.out, dpi=200)
    print(f"saved {a.out}")


if __name__ == '__main__':
    main()
