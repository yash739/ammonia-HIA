"""Stutzki & Winnewisser (1985) Figs. 5 and 6 equivalents, plotted directly.

x = central-chord optical depth of the (1,1) main line (the paper's
tau(Delta F1=0) axis), y = disc-averaged satellite/main ratios. The no-anomaly
reference is Eq. (11) computed analytically: eq11_thermal_ratio takes the
RADIAL optical depth, so it is evaluated at tau_chord / 2.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nh3hia.stutzki.physics import eq11_thermal_ratio

COLORS = {3.5: 'tab:red', 5.0: 'tab:blue', 7.0: 'tab:green'}
PANELS = [
    ('outer', lambda g, e: g[f'disc_{e}_R01'], r'$T_B(F_1{=}0\to1)\,/\,T_B(\Delta F_1{=}0)$'),
    ('outer', lambda g, e: g[f'disc_{e}_R10'], r'$T_B(F_1{=}1\to0)\,/\,T_B(\Delta F_1{=}0)$'),
    ('inner', lambda g, e: 0.5 * (g[f'disc_{e}_R12'] + g[f'disc_{e}_R21']),
     r'$[T_B(1{\to}2)+T_B(2{\to}1)]\,/\,2T_B(\Delta F_1{=}0)$'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', nargs='+', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--meshes', default=None)
    ap.add_argument('--est', choices=('fit', 'pk'), default='fit')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    df = pd.concat([pd.read_csv(c) for c in a.csv])
    df = df[(df.Status == 'SUCCESS') & (df.tau11_chord > 0)]
    if a.meshes:
        df = df[df.mesh.isin(a.meshes.split(','))]
    T = sorted(df.T_cloud.unique())

    tau = np.logspace(-2, 2, 400)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, (kind, fn, ylabel) in zip(axes, PANELS):
        ax.plot(tau, eq11_thermal_ratio(tau / 2.0, kind), 'k-', lw=1.6, label='Eq. (11), LTE / no anomaly')
        for (mesh, log_n), g in df.groupby(['mesh', 'log_n_H2']):
            g = g.sort_values('tau11_chord')
            y = fn(g, a.est).values
            x = g.tau11_chord.values
            conv = g.final_convergence.fillna(100).values >= 90
            c = COLORS.get(log_n, 'k')
            ls = '-' if mesh.startswith('radial') else '--'
            ax.plot(x, y, ls=ls, color=c, lw=1.8, label=f'{mesh}, n$_{{H_2}}$=10$^{{{log_n:g}}}$ cm$^{{-3}}$')
            ax.scatter(x[conv], y[conv], color=c, s=22, zorder=5)
            ax.scatter(x[~conv], y[~conv], facecolors='none', edgecolors=c, s=40, zorder=5)
        ax.set_xscale('log')
        ax.set_xlim(0.03, 100)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel(r'$\tau(\Delta F_1{=}0)$  (central chord, (1,1) main line)')
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', alpha=0.25)
    axes[0].legend(fontsize=8, loc='upper left')
    axes[0].set_title('Fig. 5 upper')
    axes[1].set_title('Fig. 5 lower')
    axes[2].set_title('Fig. 6')
    fig.suptitle(f"T$_k$ = {', '.join(f'{t:g}' for t in T)} K, $\\Delta v$ = 0.3 km/s  {a.label}  "
                 f"(disc-averaged spectrum, {a.est} estimator; hollow = convergence < 90%)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    plt.savefig(a.out, dpi=150)
    print('saved', a.out)


if __name__ == '__main__':
    main()
