"""Fig. 4-style contour panels for R_10_MAIN (the F_1=1->0 outer satellite,
'R_10' in escape1d's own naming) only, across the three new Delta_v sweep
rungs (0.2, 0.5, 0.8 km/s; the reference 0.3 km/s rung is included as a
4th row for context), one figure per rate set.

Uses the FINE grids (nh3hia/escape1d/grid.py --fine, already computed for the
Delta_v sweep -- 70x50x36 points, far denser than the coarse 14x14x9 grid
analysis/figures/fig4_escape1d.py normally reads) directly -- no new compute,
and actually gives smoother contours than the coarse-grid original since
griddata doesn't care whether the input is a regular grid.

Reuses analysis/figures/fig4_escape1d.py's own contour_panel function so the
clipping/colour convention (2nd-98th percentile, clipped before
interpolation, LTE floor always in range) is identical to the existing
Fig. 4 reproduction elsewhere in this project.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nh3hia.escape1d.grid import out_csv_for, outdir_for
from analysis.figures.fig4_escape1d import contour_panel
from nh3hia import paths

TEMPS = [18.0, 24.0, 30.0, 36.0]
DVS = [0.2, 0.3, 0.5, 0.8]
COL = 'R_10'
TITLE = r'$T_B(F_1=1\to0)/T_B(\Delta F_1=0)$ [outer, R\_10\_MAIN]'
OUT_DIR = paths.PRODUCTION + "/output_escape1d/results/"


def load_grid(rates, dv):
    dv_kms = None if abs(dv - 0.3) < 1e-9 else dv
    path = out_csv_for(rates, fine=True, dv_kms=dv_kms)
    df = pd.read_csv(path)
    return df[df['converged']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rates', choices=['loreau', 'full_original'], default=None,
                    help='default: run both, one figure each')
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    rates_list = [a.rates] if a.rates else ['loreau', 'full_original']

    for rates in rates_list:
        dfs = {dv: load_grid(rates, dv) for dv in DVS}
        for dv, df in dfs.items():
            print(f"rates={rates} dv={dv}: {len(df)} converged rows ({df['any_maser'].mean():.1%} masing)")

        # snap each (dv, T_target) to the nearest available fine-grid T once,
        # and pool every panel's R_10 values to fix ONE shared colour scale
        # across the whole figure (2nd-98th percentile of the pooled data,
        # same floor-at-0.0 convention as contour_panel's own per-panel default)
        panels = {}
        pooled = []
        for dv in DVS:
            df = dfs[dv]
            T_axis = np.sort(df['T_cloud'].unique())
            for T_target in TEMPS:
                T_actual = T_axis[np.argmin(np.abs(T_axis - T_target))]
                df_T = df[np.isclose(df['T_cloud'], T_actual)]
                panels[(dv, T_target)] = (T_actual, df_T)
                vals = df_T[COL].values
                pooled.append(vals[np.isfinite(vals)])
        pooled = np.concatenate(pooled)
        vmin, vmax = np.percentile(pooled, [2, 98])
        vmin = min(vmin, 0.0)
        print(f"shared colour scale for {rates}: vmin={vmin:.3f} vmax={vmax:.3f} "
              f"(pooled 2nd-98th percentile over all {len(DVS)}x{len(TEMPS)} panels)")

        fig, axes = plt.subplots(len(DVS), len(TEMPS), figsize=(3.0 * len(TEMPS) + 1.0, 3.1 * len(DVS)),
                                  sharex=True, sharey=True)
        last_cf = None
        for i, dv in enumerate(DVS):
            for j, T_target in enumerate(TEMPS):
                ax = axes[i, j]
                T_actual, df_T = panels[(dv, T_target)]
                title = f'T={T_actual:.2f}K (target {T_target:.0f}K)' if i == 0 else ''
                cf = contour_panel(fig, ax, df_T, COL, title, 'viridis', is_ratio=True,
                                    vmin=vmin, vmax=vmax, draw_colorbar=False)
                if cf is not None:
                    last_cf = cf
                if j == 0:
                    ax.set_ylabel(rf'$\Delta v$={dv:.1f} km/s' + '\n' + r'log($N_{NH_3}/\Delta v$)', fontsize=8)
                if i == len(DVS) - 1:
                    ax.set_xlabel(r'log($n_{H_2}$)', fontsize=8)

        plt.suptitle(f'{TITLE}\nescape1d, {rates} rates, guard_masers=False -- '
                     r'$\Delta v$ sweep at fixed log($N_{NH_3}/\Delta v$) axis -- shared colour scale',
                     fontsize=12)
        plt.tight_layout(rect=(0, 0, 0.90, 0.96))

        cax = fig.add_axes((0.92, 0.15, 0.015, 0.7))
        cb = fig.colorbar(last_cf, cax=cax)
        cb.set_label(TITLE, fontsize=9)

        out_png = os.path.join(OUT_DIR, f'fig4_r10_dv_sweep_{rates}.png')
        plt.savefig(out_png, dpi=180)
        print(f"saved {out_png}\n")


if __name__ == '__main__':
    main()
