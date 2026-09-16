"""Reproduce Stutzki & Winnewisser (1985) Figs. 5 and 6 from the LUT.

Fig. 5: the outer-satellite/main ratio vs the main line's optical depth,
for T_k in {18, 26, 36} K and n'_H2 in {10^3.5, 10^5.0, 10^7.0} cm^-3,
overplotted with the Eq. (11) no-anomaly reference curve.
Fig. 6: same, but for the intensity-averaged inner satellites.

The coarse LUT's T axis is the non-uniform {12,18,24,30,36,48} tuple (see
build_lut.T_VALUES) -- chosen so it lands on Stutzki's Fig. 4 temperatures,
not his Fig. 5/6 choice of 26 K. The second, offset grid (build_lut_offset.py)
added a real T=27 K rung across the full density axis specifically to close
this gap; used here as the middle panel (1 K off his 26 K, an even smaller
offset than the coarse grid's own 24 K substitute), stated rather than
silently glossed over (see plan, Paper I deliverable 4).

The Eq. (11) curve comes from stutzki_physics.eq11_thermal_ratio, derived
from the LTE (1,1) hyperfine intensities in Fig. 1 of the paper (11.1%
outer, 13.9% inner, 50.0% main) combined with the disc-averaged e(tau)
Eq. (10) already used throughout this pipeline's imaging path -- not the
plain single-sightline exponential approximation used by an earlier,
quarantined attempt at this plot.

Usage: python3 reproduce_stutzki_fig5_6.py [--lut path/to/lut_dv0.30.csv]
Runs against whatever subset of (T, n) the LUT has completed so far; missing
combinations are simply left out of the legend.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stutzki_physics import eq11_thermal_ratio

DEFAULT_LUT = "/home/yasho379/magritte_rebuilt/production/output_lut_coarse/results/lut_dv0.30.csv"
FIG56_TEMPS = [18.0, 27.0, 36.0]   # paper uses 26 K; offset grid now has a real 27 K rung (see docstring)
FIG56_DENSITIES_LOGN = [3.5, 5.0, 7.0]
LINESTYLES = {3.5: ':', 5.0: '--', 7.0: '-'}
COLORS = {18.0: 'tab:blue', 27.0: 'tab:green', 36.0: 'tab:red'}


PC_CM = 3.0857e18

# Low density + high column ("N/dv") implies a huge clump radius at fixed
# abundance (radius ~ N_target/(2*n*X)), pushing well past Stutzki's own
# ~0.01 pc clumps into a multi-pc diffuse-cloud regime this pipeline was
# never designed for. That corner (log_n<=4.5, log_N_dv>=15.35 in the
# current grid) shows non-convergent, physically nonsensical ratios
# (R_01_MAIN up to ~570) -- roughly half of it (by row count) hits the
# max_NLTE iteration cap outright. A 0.5 pc radius cut removes every
# instance of this (checked empirically: zero remaining rows with any
# satellite/main ratio > 1.5) while keeping 527/757 rows, so it is applied
# by default; pass --max-radius-pc 0 to disable.
def load_lut(path, max_radius_pc=0.5):
    df = pd.read_csv(path)
    df = df[df['Status'] == 'SUCCESS']
    df = df[df['convergence_ok']]
    df = df[df['tau_main'] > 0]
    if max_radius_pc:
        n_before = len(df)
        df = df[df['radius_sphere'] <= max_radius_pc * PC_CM]
        print(f"radius mask (<= {max_radius_pc} pc): kept {len(df)}/{n_before} rows")
    return df.sort_values('tau_main')


def plot_one(ax, df, ratio_col, title, ylabel, eq11_component):
    found_any = False
    for T in FIG56_TEMPS:
        df_T = df[np.isclose(df['T_cloud'], T)]
        if df_T.empty:
            continue
        for log_n in FIG56_DENSITIES_LOGN:
            sub = df_T[np.isclose(df_T['log_n_H2'], log_n)]
            if len(sub) < 2:
                continue
            found_any = True
            ax.plot(sub['tau_main'], sub[ratio_col],
                     color=COLORS[T], linestyle=LINESTYLES[log_n],
                     linewidth=1.6, alpha=0.85,
                     label=f"T={T:.0f}K, n'=$10^{{{log_n:.1f}}}$")

    tau_ref = np.logspace(-2, 2, 300)
    ax.plot(tau_ref, eq11_thermal_ratio(tau_ref, eq11_component),
            color='black', linewidth=2.2, label='Eq. (11), no anomaly')

    ax.set_xscale('log')
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r'$\tau(\Delta F_1=0)$')
    ax.set_ylabel(ylabel)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    if found_any:
        ax.legend(fontsize=6, ncol=2, loc='lower right')
    else:
        ax.text(0.5, 0.5, 'no matching (T, n) rows yet', ha='center', va='center',
                transform=ax.transAxes)
    return found_any


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lut', default=DEFAULT_LUT)
    ap.add_argument('--out5', default='fig5_reproduction.png')
    ap.add_argument('--out6', default='fig6_reproduction.png')
    ap.add_argument('--max-radius-pc', type=float, default=0.5,
                     help='drop rows with implied clump radius above this many pc '
                          '(0 disables the mask)')
    a = ap.parse_args()

    if not os.path.exists(a.lut) or os.path.getsize(a.lut) == 0:
        raise SystemExit(f"LUT at {a.lut} doesn't exist yet or has no rows written -- "
                          f"the build hasn't completed its first model yet. Try again later.")
    df = load_lut(a.lut, max_radius_pc=a.max_radius_pc)
    if df.empty:
        raise SystemExit("LUT has no converged SUCCESS rows yet.")

    # Fig. 5 -- outer satellites, one panel per component (0->1 and 1->0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ok1 = plot_one(axes[0], df, 'R_01_MAIN', r'$F_1=0\to1$ (outer)',
                   r'$T_B(F_1=0\to1)/T_B(\Delta F_1=0)$', 'outer')
    ok2 = plot_one(axes[1], df, 'R_10_MAIN', r'$F_1=1\to0$ (outer)',
                   r'$T_B(F_1=1\to0)/T_B(\Delta F_1=0)$', 'outer')
    plt.suptitle('Reproduction of Stutzki & Winnewisser (1985) Fig. 5 (outer satellites)')
    plt.tight_layout()
    plt.savefig(a.out5, dpi=200)
    print(f"saved {a.out5}" + ("" if (ok1 or ok2) else " (no data plotted yet)"))
    plt.close(fig)

    # Fig. 6 -- intensity-averaged inner satellites
    df = df.copy()
    df['Avg_Inner_Ratio'] = (df['R_12_MAIN'] + df['R_21_MAIN']) / 2.0
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ok3 = plot_one(ax, df, 'Avg_Inner_Ratio', 'Averaged inner satellites',
                   r'$[T_B(1{\to}2)+T_B(2{\to}1)]/2T_B(\Delta F_1=0)$', 'inner')
    plt.suptitle('Reproduction of Stutzki & Winnewisser (1985) Fig. 6 (inner satellites)')
    plt.tight_layout()
    plt.savefig(a.out6, dpi=200)
    print(f"saved {a.out6}" + ("" if ok3 else " (no data plotted yet)"))


if __name__ == '__main__':
    main()
