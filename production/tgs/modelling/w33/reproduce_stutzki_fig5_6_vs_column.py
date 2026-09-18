"""Same satellite/main ratios as reproduce_stutzki_fig5_6.py, but plotted
against log(N_NH3/dv) -- a direct LUT grid axis -- instead of tau_main,
which is a derived, image-extracted quantity whose exact convention
(radial vs full-chord optical depth) is still being checked against
Stutzki's own Fig. 5/6 x-axis definition. This plot needs no such
convention and is a clean cross-check in its own right: it shows whether
the "early saturation" seen vs tau_main is a real feature of the physics
(and should also show up vs column) or an artifact of how tau_main is
read off the image.

No Eq. (11) reference curve is drawn here -- that curve is a function of
tau_main specifically (Stutzki's Eq. 8/10), not of column density, so it
has no natural counterpart on this axis.

Usage: python3 reproduce_stutzki_fig5_6_vs_column.py [--lut path]
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_LUT = "/home/yasho379/magritte_rebuilt/production/output_lut_coarse/results/lut_dv0.30.csv"
TEMPS = [18.0, 24.0, 36.0]
DENSITIES_LOGN = [3.5, 5.0, 7.0]
LINESTYLES = {3.5: ':', 5.0: '--', 7.0: '-'}
COLORS = {18.0: 'tab:blue', 24.0: 'tab:green', 36.0: 'tab:red'}


def load_lut(path):
    df = pd.read_csv(path)
    df = df[df['Status'] == 'SUCCESS']
    df = df[df['convergence_ok']]
    return df.sort_values('log_N_dv')


def plot_one(ax, df, ratio_col, title, ylabel):
    found_any = False
    for T in TEMPS:
        df_T = df[np.isclose(df['T_cloud'], T)]
        if df_T.empty:
            continue
        for log_n in DENSITIES_LOGN:
            sub = df_T[np.isclose(df_T['log_n_H2'], log_n)]
            if len(sub) < 2:
                continue
            found_any = True
            ax.plot(sub['log_N_dv'], sub[ratio_col],
                    color=COLORS[T], linestyle=LINESTYLES[log_n],
                    marker='o', markersize=4, linewidth=1.6, alpha=0.9,
                    label=f"T={T:.0f}K, n'=$10^{{{log_n:.1f}}}$")
    ax.set_ylim(0.0, max(1.05, ax.get_ylim()[1]))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r'log($N_{NH_3}/\Delta v$) [cm$^{-2}$ s km$^{-1}$]')
    ax.set_ylabel(ylabel)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    if found_any:
        ax.legend(fontsize=7, ncol=2, loc='best')
    else:
        ax.text(0.5, 0.5, 'no matching (T, n) rows yet', ha='center', va='center',
                transform=ax.transAxes)
    return found_any


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lut', default=DEFAULT_LUT)
    ap.add_argument('--out5', default='fig5_vs_column.png')
    ap.add_argument('--out6', default='fig6_vs_column.png')
    a = ap.parse_args()

    if not os.path.exists(a.lut) or os.path.getsize(a.lut) == 0:
        raise SystemExit(f"LUT at {a.lut} doesn't exist yet or has no rows written.")
    df = load_lut(a.lut)
    if df.empty:
        raise SystemExit("LUT has no converged SUCCESS rows yet.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ok1 = plot_one(axes[0], df, 'R_01_MAIN', r'$F_1=0\to1$ (outer)',
                   r'$T_B(F_1=0\to1)/T_B(\Delta F_1=0)$')
    ok2 = plot_one(axes[1], df, 'R_10_MAIN', r'$F_1=1\to0$ (outer)',
                   r'$T_B(F_1=1\to0)/T_B(\Delta F_1=0)$')
    plt.suptitle('Outer satellite ratios vs. column density (cf. Fig. 5)')
    plt.tight_layout()
    plt.savefig(a.out5, dpi=200)
    print(f"saved {a.out5}" + ("" if (ok1 or ok2) else " (no data plotted yet)"))
    plt.close(fig)

    df = df.copy()
    df['Avg_Inner_Ratio'] = (df['R_12_MAIN'] + df['R_21_MAIN']) / 2.0
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ok3 = plot_one(ax, df, 'Avg_Inner_Ratio', 'Averaged inner satellites',
                   r'$[T_B(1{\to}2)+T_B(2{\to}1)]/2T_B(\Delta F_1=0)$')
    plt.suptitle('Inner satellite ratio vs. column density (cf. Fig. 6)')
    plt.tight_layout()
    plt.savefig(a.out6, dpi=200)
    print(f"saved {a.out6}" + ("" if ok3 else " (no data plotted yet)"))


if __name__ == '__main__':
    main()


def main_all_ratios(lut_path, out_path):
    """All six stored ratio columns vs. log(N_NH3/dv) in one grid, same
    (T, n') style as the Fig. 5/6-style plots above."""
    df = load_lut(lut_path)
    panels = [
        ('R_01_MAIN', r'$F_1=0\to1$ (outer)', r'$T_B(0\to1)/T_B(\Delta F_1=0)$'),
        ('R_10_MAIN', r'$F_1=1\to0$ (outer)', r'$T_B(1\to0)/T_B(\Delta F_1=0)$'),
        ('R_12_MAIN', r'$F_1=1\to2$ (inner)', r'$T_B(1\to2)/T_B(\Delta F_1=0)$'),
        ('R_21_MAIN', r'$F_1=2\to1$ (inner)', r'$T_B(2\to1)/T_B(\Delta F_1=0)$'),
        ('R_22_MAIN', r'$(2,2)$ main / $(1,1)$ main', r'$T_B(2,2)/T_B(1,1)$'),
        ('R_21_11', r'$(2,1)$ / $(1,1)$ main', r'$T_B(2,1)/T_B(1,1)$'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (col, title, ylabel) in zip(axes.flat, panels):
        plot_one(ax, df, col, title, ylabel)
    plt.suptitle('All stored ratios vs. column density N(NH$_3$)/$\\Delta v$', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"saved {out_path}")


if __name__ == '__main_all__':
    pass
