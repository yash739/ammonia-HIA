"""Magritte-only Stutzki & Winnewisser (1985) Figs. 5/6 from the expanded
fig56 grid (8 densities x T = 18/26/36 K), in two forms:

1. Stutzki's layout on clean axes: one panel per (T_k, satellite), densities
   overlaid, x = central-chord tau(1,1) (Magritte's post-fix tau_main, the
   same quantity as his x-axis), Eq. (11) evaluated at tau/2 (it takes his
   radial tau -- see mesh-comparison-and-fig56-2026-09-16.md sec 1).
2. The same data drawn onto the scanned page of his figure at T = 18 K, his
   three densities only (the page calibration in plot_fig56_overlay.py covers
   his T = 18 K curves), reusing plot_fig56_overlay.panel.

Known limitation carried on both: the grid uses the cube mesh, which
under-reports the central optical depth by roughly a factor of two
(magritte-quirks-2026-09-16.md sec 2), so Magritte points sit left of where
the true tau would put them -- visible as the inner satellites (almost
anomaly-free) running ahead of Eq. (11).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from compare_fig5_6_direct import load_magritte, eq11_on_chord_axis, OUTDIR, TEMPS
import plot_fig56_overlay as ov

LOG_NS = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
STUTZKI_NS = {3.5, 5.0, 7.0}
OBSERVABLES = [
    (r'$F_1=0\to1$ (outer)', 'R_01_MAIN', 'outer'),
    (r'$F_1=1\to0$ (outer)', 'R_10_MAIN', 'outer'),
    ('inner satellites (averaged)', 'Avg_Inner_Ratio', 'inner'),
]


def clean_axes_figure(mag):
    cmap = plt.get_cmap('viridis')
    colors = {n: cmap(i / (len(LOG_NS) - 1)) for i, n in enumerate(LOG_NS)}
    tau_ref = np.logspace(-2.2, 1.6, 300)
    fig, axes = plt.subplots(len(TEMPS), len(OBSERVABLES), figsize=(13.5, 11.5), sharex=True, sharey=True)
    for i, T in enumerate(TEMPS):
        for j, (title, col, comp) in enumerate(OBSERVABLES):
            ax = axes[i, j]
            ax.plot(tau_ref, eq11_on_chord_axis(tau_ref, comp), color='black', ls='--', lw=1.2)
            for n in LOG_NS:
                g = mag[np.isclose(mag['T_cloud'], T) & np.isclose(mag['log_n_H2'], n)].sort_values('tau_chord')
                ax.plot(g['tau_chord'], g[col], color=colors[n], lw=2.0 if n in STUTZKI_NS else 1.0,
                        marker='o', ms=3.5, mec='white', mew=0.4)
            ax.set_xscale('log')
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, which='both', ls='--', alpha=0.2)
            if i == 0:
                ax.set_title(title, fontsize=11)
            if j == 0:
                ax.set_ylabel(f'$T_k$ = {T:.0f} K\nsatellite / main', fontsize=10)
            if i == len(TEMPS) - 1:
                ax.set_xlabel(r'central-chord $\tau(\Delta F_1=0)$', fontsize=10)
    handles = [Line2D([], [], color=colors[n], lw=2.0 if n in STUTZKI_NS else 1.0, marker='o', ms=3.5,
                      label=f"log n' = {n:.1f}" + (' (Stutzki)' if n in STUTZKI_NS else ''))
               for n in LOG_NS] + [Line2D([], [], color='black', ls='--', label='Eq. (11), no anomaly')]
    fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle('Stutzki & Winnewisser (1985) Figs. 5-6 reproduced with Magritte 3D NLTE '
                 '(Loreau et al. 2023 rates, cube mesh), 8 densities', y=1.04, fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(OUTDIR, 'fig5_6_magritte_8dens.png')
    plt.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)


def scanned_page_figure(mag):
    g = mag[np.isclose(mag['T_cloud'], 18.0) & mag['log_n_H2'].isin(sorted(STUTZKI_NS))].copy()
    df = pd.DataFrame({
        'mesh': 'cube', 'mode': 'nlte', 'log_n_H2': g['log_n_H2'].values,
        'tau11_chord': g['tau_chord'].values, 'final_convergence': g['final_convergence'].values,
        'disc_fit_R01': g['R_01_MAIN'].values, 'disc_fit_R10': g['R_10_MAIN'].values,
        'disc_fit_R12': g['R_12_MAIN'].values, 'disc_fit_R21': g['R_21_MAIN'].values,
    })
    page = ov.page_image()
    r01 = lambda d, e: d[f'disc_{e}_R01'].values
    r10 = lambda d, e: d[f'disc_{e}_R10'].values
    rin = lambda d, e: 0.5 * (d[f'disc_{e}_R12'].values + d[f'disc_{e}_R21'].values)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.5))
    ov.panel(axes[0], page, ov.FIG5_UPPER, df, r01, 'Fig. 5 upper: F1=0->1 / main', 'fit')
    ov.panel(axes[1], page, ov.FIG5_LOWER, df, r10, 'Fig. 5 lower: F1=1->0 / main', 'fit')
    ov.panel(axes[2], page, ov.FIG6_T18, df, rin, 'Fig. 6, T=18 K: inner average / main', 'fit')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=10)
    fig.suptitle("Stutzki & Winnewisser (1985), scanned (black), vs Magritte 3D NLTE at T_k = 18 K "
                 "(red 10^3.5, blue 10^5, green 10^7 cm^-3; hollow = convergence < 90%)\n"
                 "x = central-chord tau(1,1). His T = 18 K curves -- Fig. 5: thick solid 10^3.5, dotted 10^5, "
                 "dashed 10^7; Fig. 6: dashed 10^3.5, dotted 10^5, solid 10^7", fontsize=10)
    plt.tight_layout(rect=(0, 0.07, 1, 0.94))
    out = os.path.join(OUTDIR, 'fig5_6_magritte_T18_on_scanned_page.png')
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print('saved', out)


if __name__ == '__main__':
    mag = load_magritte()
    clean_axes_figure(mag)
    scanned_page_figure(mag)
