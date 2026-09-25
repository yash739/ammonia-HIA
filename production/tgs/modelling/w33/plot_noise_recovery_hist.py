"""Histograms of recovered log n_H2 per noisy realization for the gold-LUT
noise-injection recovery test (noise_recovery_lut.py), plus branch
fractions -- the aggregate median/scatter in noise_recovery_summary_*.csv
cannot distinguish a broad unimodal spread from a split between two density
branches, which is exactly the question.

Branch classification per realization: 'at truth' if |recovered - true|
<= 0.5 dex (the pass tolerance), 'high branch' if more than 0.5 dex above,
'low branch' if more than 0.5 dex below.

Usage: python3 plot_noise_recovery_hist.py [--mode holdout|onnode]
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = "/home/yasho379/magritte_rebuilt/production/output_lut_gold/results/"
TOL = 0.5


def classify(rec, true):
    d = rec - true
    return pd.Series({'at_truth': np.mean(np.abs(d) <= TOL),
                      'high_branch': np.mean(d > TOL),
                      'low_branch': np.mean(d < -TOL)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['holdout', 'onnode'], default='holdout')
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(RES, f'noise_recovery_realizations_{a.mode}.csv'))
    df = df.dropna(subset=['raw_log_n'])

    rows = []
    for (label, snr), g in df.groupby(['label', 'snr']):
        t = g['log_n_true'].iloc[0]
        for kind in ('raw', 'filt'):
            c = classify(g[f'{kind}_log_n'].values, t)
            rows.append(dict(label=label, log_n_true=t, snr=snr, estimator=kind, n=len(g), **c))
    frac = pd.DataFrame(rows).sort_values(['log_n_true', 'snr', 'estimator'], ascending=[True, False, True])
    out_csv = os.path.join(RES, f'noise_recovery_branch_fractions_{a.mode}.csv')
    frac.to_csv(out_csv, index=False)
    pd.set_option('display.width', 200)
    print(frac.round(3).to_string(index=False))
    print(f"saved {out_csv}")

    truths = sorted(df['log_n_true'].unique())
    snrs = sorted(df['snr'].unique(), reverse=True)
    bins = np.arange(3.5, 7.5 + 0.1, 0.1)
    fig, axes = plt.subplots(len(truths), len(snrs), figsize=(3.0 * len(snrs), 2.6 * len(truths)),
                             sharex=True, sharey=True)
    for i, t in enumerate(truths):
        for j, snr in enumerate(snrs):
            ax = axes[i, j]
            g = df[(df['log_n_true'] == t) & (df['snr'] == snr)]
            ax.hist(g['raw_log_n'], bins=bins, color='tab:blue', alpha=0.55, label='raw $\\chi^2$ best')
            ax.hist(g['filt_log_n'], bins=bins, histtype='step', color='tab:red', lw=1.5,
                    label='after physical filter')
            ax.axvspan(t - TOL, t + TOL, color='k', alpha=0.08)
            ax.axvline(t, color='k', ls='--', lw=1)
            f = frac[(frac.log_n_true == t) & (frac.snr == snr) & (frac.estimator == 'raw')].iloc[0]
            ax.text(0.03, 0.95, f"at truth {f.at_truth:.0%}\nhigh {f.high_branch:.0%}  low {f.low_branch:.0%}",
                    transform=ax.transAxes, va='top', fontsize=7)
            if i == 0:
                ax.set_title(f'S/N = {snr}', fontsize=10)
            if j == 0:
                ax.set_ylabel(f'truth log n = {t}\nrealizations', fontsize=9)
            if i == len(truths) - 1:
                ax.set_xlabel(r'recovered $\log n_{\rm H_2}$', fontsize=9)
    axes[0, 0].legend(fontsize=7, loc='center left')
    plt.suptitle(f'Noise-injection recovery against the gold LUT ({a.mode}), '
                 r'$T_k$=24 K, $\log(N/\Delta v)$=14.9 -- grey band = $\pm$0.5 dex of truth',
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = os.path.join(RES, f'noise_recovery_hist_{a.mode}.png')
    plt.savefig(out_png, dpi=170)
    print(f"saved {out_png}")


if __name__ == '__main__':
    main()
