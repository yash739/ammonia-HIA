"""Difference map: escape-probability model minus corrected Magritte 3D RT,
evaluated at the EXACT (n_H2, N_NH3) points Magritte has data for (a paired
comparison, no interpolation needed on the escape-probability side), shown
as a scatter-based heatmap in the (log n_H2, log N_NH3/dv) plane -- the same
axes as the Fig. 4 reproductions.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import nh3_escape_model as m
from compare_magritte import load_corrected_magritte, MAGRITTE_CSV

OUT_DIR = '/home/yasho379/magritte_rebuilt/scratch/output/output_stutzki85'
DV_CLUMP = 0.3


def compute_paired(T_k=36.0):
    df = load_corrected_magritte(T_k)
    df['log_N_dv'] = np.log10(df['N_NH3']) - np.log10(DV_CLUMP)

    model = m.NH3Model()
    Cmat = model.collision_matrix(T_k)

    # sort by (n, N) so warm-starting is effective
    df = df.sort_values(['log_n_H2', 'log_N_dv']).reset_index(drop=True)
    rows = []
    x0 = None
    prev_n = None
    for _, r in df.iterrows():
        if prev_n is not None and not np.isclose(r['log_n_H2'], prev_n):
            x0 = None  # new density column, restart warm-start chain
        out = m.run_one(model, Cmat, T_k=T_k, n_H2=10 ** r['log_n_H2'],
                         log_N_dv=r['log_N_dv'], x0=x0)
        x0 = out['x']
        prev_n = r['log_n_H2']
        rows.append(dict(log_n_H2=r['log_n_H2'], log_N_dv=r['log_N_dv'],
                          tau_main_escape=out['tau_main'],
                          R_01_escape=out['R_01'], R_10_escape=out['R_10'],
                          R_12_escape=out['R_12'], R_21_escape=out['R_21'],
                          R_01_magritte=r['R_01_MAIN_fixed'], R_10_magritte=r['R_10_MAIN_fixed'],
                          R_12_magritte=r['R_12_MAIN_fixed'], R_21_magritte=r['R_21_MAIN_fixed'],
                          tau_main_magritte=r['tau_main']))
    return pd.DataFrame(rows)


def plot_diff_heatmap(df, out_path):
    panels = [('R_01', r'$F_1=0\to1$'), ('R_10', r'$F_1=1\to0$'),
              ('R_12', r'$F_1=1\to2$'), ('R_21', r'$F_1=2\to1$')]
    fig, axes = plt.subplots(2, len(panels), figsize=(4.2 * len(panels), 8))

    for j, (col, title) in enumerate(panels):
        esc = df[f'{col}_escape'].values
        mag = df[f'{col}_magritte'].values
        diff = esc - mag
        valid = np.isfinite(diff)

        ax = axes[0, j]
        vmax = np.nanpercentile(np.abs(diff[valid]), 95)
        sc = ax.scatter(df['log_n_H2'][valid], df['log_N_dv'][valid], c=diff[valid],
                          cmap='RdBu_r', vmin=-vmax, vmax=vmax, s=22, edgecolors='none')
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(f'{title}: escape $-$ Magritte', fontsize=10)
        ax.set_xlabel(r'$\log(n_{H_2})$')
        if j == 0:
            ax.set_ylabel(r'$\log(N_{NH_3}/\Delta v)$')

        n_masked = (~np.isfinite(esc)).sum()

        ax2 = axes[1, j]
        log_n = df['log_n_H2'].values
        n_list = sorted(set(np.round(log_n, 2)))
        colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(n_list)))
        tau_m = df['tau_main_magritte'].values
        for log_n_val, color in zip(n_list, colors):
            m_ = valid & np.isclose(log_n, log_n_val, atol=1e-2)
            ax2.scatter(tau_m[m_], diff[m_], color=color, s=10,
                        label=f'$n=10^{{{log_n_val:.1f}}}$')
        ax2.axhline(0, color='k', lw=0.8, ls='--')
        ax2.set_xscale('log')
        ax2.set_xlabel(r'$\tau_{main}$ (Magritte)')
        ax2.set_ylabel('escape $-$ Magritte')
        ax2.set_title(f'{title}: difference vs $\\tau_{{main}}$', fontsize=9)
        ax2.grid(alpha=0.3)
        if j == len(panels) - 1:
            ax2.legend(fontsize=5, ncol=2, loc='upper left')

        print(f'{col}: {valid.sum()}/{len(df)} finite, masked (maser) at escape side: {n_masked}, '
              f'mean diff {np.nanmean(diff[valid]):+.3f}, RMS diff {np.sqrt(np.nanmean(diff[valid]**2)):.3f}')

    plt.suptitle('Escape-probability model minus corrected Magritte 3D RT (T=36K)\n'
                  'top row: difference map in (n, N) plane; bottom row: same difference vs optical depth',
                  fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f'saved {out_path}')


if __name__ == '__main__':
    df = compute_paired(36.0)
    df.to_csv(f'{OUT_DIR}/compare_paired_T36.csv', index=False)
    plot_diff_heatmap(df, f'{OUT_DIR}/compare_diff_heatmap_T36.png')
