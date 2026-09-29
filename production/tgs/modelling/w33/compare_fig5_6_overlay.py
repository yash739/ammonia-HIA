"""Stutzki & Winnewisser (1985) Figs. 5/6 in their own layout: satellite/main
ratio vs main-line optical depth, one panel per (T_k, observable), with every
density overlaid as a colour-coded curve -- now 8 densities (log n' = 3.5 to
7.0 in 0.5-dex steps) after the fig56 grid expansion, rather than his three.

Per panel: Magritte 3D NLTE (points, output_lut_fig56), escape1d at the same
parameters (lines, stutzki85/run_fig5_6_grid.py), and the Eq. (11) LTE
no-anomaly floor (dashed black). One figure per escape1d rate set.

compare_fig5_6_direct.py keeps the one-panel-per-(T, n) view with residuals
for Stutzki's three densities; this script is the multi-density overview,
and also writes the Magritte-vs-escape1d residual table for all 8.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from stutzki_physics import eq11_thermal_ratio
from compare_fig5_6_direct import (MAGRITTE_CSV, ESCAPE1D_CSV, OUTDIR, TEMPS,
                                   load_magritte, load_escape1d, interp_residual, summarize_residual)

LOG_NS = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
STUTZKI_NS = {3.5, 5.0, 7.0}
RATES_LABEL = {'full_original': "Stutzki's own rates", 'loreau': 'Loreau et al. (2023) rates'}

# (key, panel title, magritte col, escape1d col, eq11 component)
OBSERVABLES = [
    ('outer_01', r'$F_1=0\to1$ (outer)', 'R_01_MAIN', 'R_01', 'outer'),
    ('outer_10', r'$F_1=1\to0$ (outer)', 'R_10_MAIN', 'R_10', 'outer'),
    ('inner_avg', 'inner satellites (averaged)', 'Avg_Inner_Ratio', 'Avg_Inner_Ratio', 'inner'),
]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    mag = load_magritte()
    cmap = plt.get_cmap('viridis')
    colors = {n: cmap(i / (len(LOG_NS) - 1)) for i, n in enumerate(LOG_NS)}
    tau_ref = np.logspace(-2.2, 1.3, 300)
    resid_rows = []

    for rates, rates_label in RATES_LABEL.items():
        esc = load_escape1d(rates)
        fig, axes = plt.subplots(len(TEMPS), len(OBSERVABLES), figsize=(13.5, 11.5),
                                 sharex=True, sharey=True)
        for i, T in enumerate(TEMPS):
            for j, (okey, otitle, mcol, ecol, eq11c) in enumerate(OBSERVABLES):
                ax = axes[i, j]
                ax.plot(tau_ref, eq11_thermal_ratio(tau_ref, eq11c), color='black', ls='--', lw=1.2, alpha=0.8)
                for n in LOG_NS:
                    c = colors[n]
                    e = esc[np.isclose(esc['T_k'], T) & np.isclose(esc['log_n_H2'], n)].sort_values('tau_main')
                    mm = mag[np.isclose(mag['T_cloud'], T) & np.isclose(mag['log_n_H2'], n)].sort_values('tau_main')
                    lw = 2.0 if n in STUTZKI_NS else 1.1
                    ax.plot(e['tau_main'], e[ecol], color=c, lw=lw, alpha=0.9)
                    ax.plot(mm['tau_main'], mm[mcol], ls='none', marker='o', ms=4,
                            mfc=c, mec='white', mew=0.5)
                    r = interp_residual(mm['tau_main'].values, mm[mcol].values,
                                        e['tau_main'].values, e[ecol].values)
                    med, mx, nov = summarize_residual(r)
                    resid_rows.append(dict(rates=rates, observable=okey, T_k=T, log_n_H2=n,
                                           n_overlap=nov, median_abs_resid=med, max_abs_resid=mx))
                ax.set_xscale('log')
                ax.set_ylim(0.0, 1.05)
                ax.grid(True, which='both', ls='--', alpha=0.2)
                if i == 0:
                    ax.set_title(otitle, fontsize=11)
                if j == 0:
                    ax.set_ylabel(f'$T_k$ = {T:.0f} K\nsatellite / main', fontsize=10)
                if i == len(TEMPS) - 1:
                    ax.set_xlabel(r'$\tau(\Delta F_1=0)$', fontsize=10)

        handles = [Line2D([], [], color=colors[n], lw=2.0 if n in STUTZKI_NS else 1.1,
                          label=f"log n' = {n:.1f}" + (' (Stutzki)' if n in STUTZKI_NS else ''))
                   for n in LOG_NS]
        handles += [Line2D([], [], color='gray', lw=1.5, label=f'escape1d, {rates_label}'),
                    Line2D([], [], color='gray', ls='none', marker='o', label='Magritte 3D NLTE'),
                    Line2D([], [], color='black', ls='--', label='Eq. (11), no anomaly')]
        fig.legend(handles=handles, loc='upper center', ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(f"Stutzki & Winnewisser (1985) Figs. 5-6, 8 densities: Magritte 3D NLTE vs "
                     f"escape1d ({rates_label})", y=1.05, fontsize=12)
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        out_png = os.path.join(OUTDIR, f'fig5_6_overlay_{rates}.png')
        plt.savefig(out_png, dpi=170, bbox_inches='tight')
        plt.close(fig)
        print(f"saved {out_png}")

    res = pd.DataFrame(resid_rows)
    out_csv = os.path.join(OUTDIR, 'residual_summary_8dens.csv')
    res.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")
    pd.set_option('display.width', 200)
    print(res.pivot_table(index=['rates', 'observable'], columns='log_n_H2',
                          values='median_abs_resid', aggfunc='median').round(3).to_string())


if __name__ == '__main__':
    main()
