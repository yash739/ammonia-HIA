"""Reproduce Stutzki & Winnewisser (1985) Fig. 4a-c from the LUT.

Fig. 4 plots, in the log10(n_H2) vs log10(N_NH3/dv) plane, contours of the
relative intensity of each (1,1) satellite to the main line, and of the (2,2)
main-line-to-(1,1)-main-line ratio, in four panels stacked by kinetic
temperature (18, 24, 30, 36 K -- the LUT grid is deliberately spaced to land
exactly on these). We do not have digitized contour data from the printed
figure to overlay -- Table-based digitization was superseded by the
user-provided CSV for the ratio tables, and no equivalent exists for Fig. 4's
contour levels themselves -- so this reproduces our own model's contours at
his grid extent and temperatures, for comparison by eye against the published
panels (paper pages 18-19), per the plan (Paper I deliverable 3).

Usage: python3 reproduce_stutzki_fig4.py [--lut path/to/lut_dv0.30.csv]
Requires all four temperatures (18, 24, 30, 36 K) to have enough converged
rows in the LUT to interpolate a contour; panels for temperatures not yet
covered are skipped with a note, so this can be run against a partial build.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

DEFAULT_LUT = "/home/yasho379/magritte_rebuilt/production/output_lut_gold/results/lut_dv0.30.csv"
FIG4_TEMPS = [18.0, 24.0, 30.0, 36.0]

# (column, panel title, cmap) -- matches Fig. 4a (outer), 4b (inner), 4c ((2,2)/(1,1))
PANELS = [
    ('R_01_MAIN', r'$T_B(F_1=0\to1)/T_B(\Delta F_1=0)$ [outer]', 'viridis'),
    ('R_10_MAIN', r'$T_B(F_1=1\to0)/T_B(\Delta F_1=0)$ [outer]', 'viridis'),
    ('R_12_MAIN', r'$T_B(F_1=1\to2)/T_B(\Delta F_1=0)$ [inner]', 'plasma'),
    ('R_21_MAIN', r'$T_B(F_1=2\to1)/T_B(\Delta F_1=0)$ [inner]', 'plasma'),
    ('R_22_MAIN', r'$T_B(2,2;\Delta F_1=0)/T_B(1,1;\Delta F_1=0)$', 'cividis'),
]


def load_lut(path):
    df = pd.read_csv(path)
    df = df[df['Status'] == 'SUCCESS']
    df = df[df['convergence_ok']]
    # radius mask intentionally OFF for this run (user request)
    return df


def contour_panel(ax, df_T, col, title, cmap):
    x = df_T['log_n_H2'].values
    y = df_T['log_N_dv'].values
    z = df_T[col].values
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]
    if len(x) < 8:
        ax.text(0.5, 0.5, f'insufficient data\n({len(x)} points)',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        return
    gx = np.linspace(x.min(), x.max(), 200)
    gy = np.linspace(y.min(), y.max(), 200)
    GX, GY = np.meshgrid(gx, gy)
    GZ = griddata(np.column_stack([x, y]), z, (GX, GY), method='linear')
    cf = ax.contourf(GX, GY, GZ, levels=20, cmap=cmap)
    cs = ax.contour(GX, GY, GZ, levels=10, colors='k', linewidths=0.4, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')
    ax.scatter(x, y, s=4, c='k', alpha=0.15)
    ax.set_title(title, fontsize=10)
    return cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lut', default=DEFAULT_LUT)
    ap.add_argument('--out', default='fig4_reproduction.png')
    ap.add_argument('--temps', default=None, help='comma-separated T_cloud list; default is the 4 Stutzki panel temps')
    a = ap.parse_args()

    global FIG4_TEMPS
    if a.temps:
        FIG4_TEMPS = [float(x) for x in a.temps.split(',')]

    if not os.path.exists(a.lut) or os.path.getsize(a.lut) == 0:
        raise SystemExit(f"LUT at {a.lut} doesn't exist yet or has no rows written -- "
                          f"the build hasn't completed its first model yet. Try again later.")
    df = load_lut(a.lut)

    available_T = [T for T in FIG4_TEMPS if len(df[np.isclose(df['T_cloud'], T)]) >= 8]
    if not available_T:
        raise SystemExit(
            f"No Fig. 4 temperature (18/24/30/36 K) has >=8 converged rows yet "
            f"({len(df)} rows total in LUT so far). Re-run once the build progresses."
        )
    print(f"Temperatures with enough data: {available_T} "
          f"(missing: {[T for T in FIG4_TEMPS if T not in available_T]})")

    n_rows = len(FIG4_TEMPS)
    n_cols = len(PANELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows),
                              sharex=True, sharey=True)

    for i, T in enumerate(FIG4_TEMPS):
        df_T = df[np.isclose(df['T_cloud'], T)]
        for j, (col, title, cmap) in enumerate(PANELS):
            ax = axes[i, j]
            contour_panel(ax, df_T, col, title if i == 0 else '', cmap)
            if j == 0:
                ax.set_ylabel(f'T={T:.0f}K\n' + r'log($N_{NH_3}/\Delta v$)', fontsize=8)
            if i == n_rows - 1:
                ax.set_xlabel(r'log($n_{H_2}$)', fontsize=8)

    plt.suptitle('Reproduction of Stutzki & Winnewisser (1985) Fig. 4a-c '
                  '(our LUT, same grid extent/temperatures)', fontsize=12)
    plt.tight_layout()
    plt.savefig(a.out, dpi=200)
    print(f"saved {a.out}")


if __name__ == '__main__':
    main()
