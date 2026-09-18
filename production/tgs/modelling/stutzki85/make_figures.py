"""Phase 6: reproduce Stutzki & Winnewisser (1985) Figs. 4-6 from the grids
computed by grid.py. Self-contained (no Magritte import) -- deliberately does
NOT reuse tgs/modelling/w33/stutzki_physics.eq11_thermal_ratio, which pulls in
nh3_NLTE_sphere -> magritte.*; e_escape here is the same Eq. (10) formula,
just kept local so this whole reproduction has zero Magritte dependency.

NaN cells/points (see nh3_escape_model.group_brightness_temperatures) show as
gaps in the contour panels and breaks in the Fig. 5/6 curves -- these mark
grid points where a hyperfine group came out genuinely population-inverted
(masing) in this solve, which Stutzki's own thermal escape-probability
closure (and this direct implementation of it) has no way to report a finite
brightness for. See README note below / the run's printed summary for how
much of the grid that affects and why it's expected with modern collision
rates.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from nh3_escape_model import e_escape

OUTER_LTE_RATIO = 0.111 / 0.500  # r, from Fig. 1's LTE hyperfine intensities
INNER_LTE_RATIO = 0.139 / 0.500

FIG4_PANELS = [
    ('R_10', r'$T_B(F_1{=}1{\to}0)/T_B(\Delta F_1{=}0)$'),
    ('R_01', r'$T_B(F_1{=}0{\to}1)/T_B(\Delta F_1{=}0)$'),
    ('R_12', r'$T_B(F_1{=}1{\to}2)/T_B(\Delta F_1{=}0)$'),
    ('R_21', r'$T_B(F_1{=}2{\to}1)/T_B(\Delta F_1{=}0)$'),
    ('R_2211', r'$T_B(2,2)/T_B(1,1)$'),
]


def eq11_ratio(tau_main, r):
    """Stutzki Eq. (11): the no-anomaly (equal T_ex) satellite/main ratio as
    a function of the main line's optical depth alone."""
    tau_main = np.asarray(tau_main, dtype=float)
    tau_sat = r * tau_main
    return (1.0 - e_escape(2.0 * tau_sat)) / (1.0 - e_escape(2.0 * tau_main))


def make_fig4(df, out_path):
    temps = sorted(df['T_k'].unique())
    fig, axes = plt.subplots(len(temps), len(FIG4_PANELS),
                              figsize=(3.3 * len(FIG4_PANELS), 3.0 * len(temps)),
                              sharex=True, sharey=True)
    for i, T in enumerate(temps):
        df_T = df[df['T_k'] == T]
        x = df_T['log_n_H2'].values
        y = df_T['log_N_dv'].values
        gx = np.linspace(x.min(), x.max(), 220)
        gy = np.linspace(y.min(), y.max(), 220)
        GX, GY = np.meshgrid(gx, gy)
        for j, (col, title) in enumerate(FIG4_PANELS):
            ax = axes[i, j]
            z = df_T[col].values
            valid = np.isfinite(z)
            if valid.sum() < 8:
                ax.text(0.5, 0.5, 'masing\neverywhere', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8)
            else:
                GZ = griddata(np.column_stack([x[valid], y[valid]]), z[valid],
                               (GX, GY), method='linear')
                cf = ax.contourf(GX, GY, GZ, levels=20, cmap='viridis')
                cs = ax.contour(GX, GY, GZ, levels=8, colors='k', linewidths=0.4, alpha=0.6)
                ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')
            if i == 0:
                ax.set_title(title, fontsize=10)
            if j == 0:
                ax.set_ylabel(f'$T_k$={T:.0f} K\n' + r'$\log(N_{NH_3}/\Delta v)$', fontsize=8)
            if i == len(temps) - 1:
                ax.set_xlabel(r'$\log(n_{H_2})$', fontsize=8)
    plt.suptitle('Reproduction of Stutzki & Winnewisser (1985) Fig. 4a-c\n'
                  '(escape-probability solve, Loreau et al. 2023 NH3-H2 rates -- '
                  'not Magritte; blank cells = population-inverted group, NaN)',
                  fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved {out_path}")


def make_fig5_6(df, out5_path, out6_path):
    temps = sorted(df['T_k'].unique())
    log_ns = sorted(df['log_n_H2'].unique())
    colors = dict(zip(temps, ['tab:blue', 'tab:green', 'tab:red']))
    styles = dict(zip(log_ns, [':', '--', '-']))

    def plot_panel(ax, ratio_col, r_lte, title, ylabel):
        for T in temps:
            for log_n in log_ns:
                sub = df[(df['T_k'] == T) & (df['log_n_H2'] == log_n)].sort_values('tau_main')
                if len(sub) < 2:
                    continue
                ax.plot(sub['tau_main'], sub[ratio_col], color=colors[T],
                         linestyle=styles[log_n], linewidth=1.6, alpha=0.85,
                         label=f"T={T:.0f}K, n=$10^{{{log_n:.1f}}}$")
        tau_ref = np.logspace(-1.5, 2, 300)
        ax.plot(tau_ref, eq11_ratio(tau_ref, r_lte), color='black', linewidth=2.2,
                 label='Eq. (11), no anomaly')
        ax.set_xscale('log')
        ax.set_ylim(0.0, 1.3)
        ax.set_xlabel(r'$\tau(\Delta F_1=0)$')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(True, which='both', ls='--', alpha=0.3)
        ax.legend(fontsize=6, ncol=2, loc='upper left')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    plot_panel(axes[0], 'R_10', OUTER_LTE_RATIO, r'$F_1=1\to0$ (outer)',
               r'$T_B(F_1{=}1{\to}0)/T_B(\Delta F_1{=}0)$')
    plot_panel(axes[1], 'R_01', OUTER_LTE_RATIO, r'$F_1=0\to1$ (outer)',
               r'$T_B(F_1{=}0{\to}1)/T_B(\Delta F_1{=}0)$')
    plt.suptitle('Reproduction of Stutzki & Winnewisser (1985) Fig. 5 (outer satellites)\n'
                  'gaps = population-inverted at that (T,n,N)')
    plt.tight_layout()
    plt.savefig(out5_path, dpi=200)
    plt.close(fig)
    print(f"saved {out5_path}")

    df = df.copy()
    df['Avg_Inner_Ratio'] = (df['R_12'] + df['R_21']) / 2.0
    fig, ax = plt.subplots(figsize=(6, 4.8))
    plot_panel(ax, 'Avg_Inner_Ratio', INNER_LTE_RATIO, 'Averaged inner satellites',
               r'$[T_B(1{\to}2)+T_B(2{\to}1)]/2T_B(\Delta F_1{=}0)$')
    plt.suptitle('Reproduction of Stutzki & Winnewisser (1985) Fig. 6 (inner satellites)')
    plt.tight_layout()
    plt.savefig(out6_path, dpi=200)
    plt.close(fig)
    print(f"saved {out6_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fig4-csv', default='/home/yasho379/magritte_rebuilt/scratch/output/output_stutzki85/stutzki85_fig4_grid.csv')
    ap.add_argument('--fig56-csv', default='/home/yasho379/magritte_rebuilt/scratch/output/output_stutzki85/stutzki85_fig56_curves.csv')
    ap.add_argument('--out-dir', default='/home/yasho379/magritte_rebuilt/scratch/output/output_stutzki85')
    a = ap.parse_args()

    df4 = pd.read_csv(a.fig4_csv)
    make_fig4(df4, os.path.join(a.out_dir, 'stutzki85_fig4.png'))

    df56 = pd.read_csv(a.fig56_csv)
    make_fig5_6(df56,
                os.path.join(a.out_dir, 'stutzki85_fig5.png'),
                os.path.join(a.out_dir, 'stutzki85_fig6.png'))
