"""Paper I, Work item 2a deliverable: compare the 23-position Stutzki
retrieval across the Delta_v = 0.2/0.3/0.5/0.8 km/s escape1d fine grids,
for one rate set at a time. No new compute -- reads the four already-
computed stutzki1985_retrieval/summary.csv files.

Zhou et al. (2020)'s prediction: HIA_OS (and, by extension, the fitted
anomaly-implied density needed to reproduce it) should weaken toward the
LTE/no-anomaly value as the assumed clump linewidth broadens toward the
observed blended value -- i.e. the retrieved log_n_H2 should trend DOWN
and chi2 (fit to observed hyperfine ratios assuming a narrow intrinsic
clump) should trend UP as Delta_v widens, if the observed linewidths are
closer to the broad/ensemble regime than to Stutzki's own 0.3 km/s
assumption.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from build_escape_grid import outdir_for

DVS = ['0.20', '0.30', '0.50', '0.80']
OUT_DIR = "/home/yasho379/magritte_rebuilt/production/output_escape1d/results/"


def summary_path(rates, dv):
    dv_kms = None if dv == '0.30' else float(dv)
    return os.path.join(outdir_for(rates, dv_kms=dv_kms), 'stutzki1985_retrieval', 'summary.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rates', choices=['loreau', 'full_original'], default='loreau')
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    dfs = {}
    for dv in DVS:
        p = summary_path(a.rates, dv)
        if not os.path.exists(p):
            print(f"MISSING: {p} -- skipping dv={dv}")
            continue
        dfs[dv] = pd.read_csv(p)

    print(f"=== rates={a.rates}: summary stats per Delta_v rung ===")
    for dv, df in dfs.items():
        print(f"  dv={dv}  median chi2={df.chi2.median():6.3f}  median log_n_H2={df.log_n_H2_fit.median():.3f}  "
              f"physically_consistent={100*df.physically_consistent.mean():5.1f}%  "
              f"any_maser={100*df.any_maser_at_fit.mean():5.1f}%")
    print()

    idcols = [c for c in ['region', 'pos'] if c in dfs['0.30'].columns]
    merged = dfs['0.30'][idcols].copy()
    for dv, df in dfs.items():
        tag = dv.replace('.', '')
        merged[f'logn_{tag}'] = df['log_n_H2_fit'].values
        merged[f'chi2_{tag}'] = df['chi2'].values
        merged[f'consistent_{tag}'] = df['physically_consistent'].values

    logn_cols = [f'logn_{dv.replace(".", "")}' for dv in DVS if dv in dfs]
    chi2_cols = [f'chi2_{dv.replace(".", "")}' for dv in DVS if dv in dfs]

    print("=== per-position log_n_H2_fit across Delta_v ===")
    print(merged[idcols + logn_cols].to_string(index=False))
    print()
    print("=== per-position chi2 across Delta_v ===")
    print(merged[idcols + chi2_cols].to_string(index=False))
    print()

    n_moved = (merged[logn_cols].max(axis=1) - merged[logn_cols].min(axis=1) > 0.02).sum()
    print(f"positions where log_n_H2_fit varies by >0.02 dex across the Delta_v sweep: "
          f"{n_moved}/{len(merged)}")

    # trend check: does median chi2 rise and median log_n fall with Delta_v (Zhou-quenching direction)?
    dv_vals = [float(dv) for dv in DVS if dv in dfs]
    med_chi2 = [dfs[dv].chi2.median() for dv in DVS if dv in dfs]
    med_logn = [dfs[dv].log_n_H2_fit.median() for dv in DVS if dv in dfs]
    print(f"\nmedian chi2 vs Delta_v: {list(zip(dv_vals, [round(x,3) for x in med_chi2]))}")
    print(f"median log_n vs Delta_v: {list(zip(dv_vals, [round(x,3) for x in med_logn]))}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    for _, row in merged.iterrows():
        ax.plot(dv_vals, [row[c] for c in logn_cols], marker='o', ms=3, alpha=0.5, lw=1)
    ax.plot(dv_vals, med_logn, marker='o', ms=8, color='black', lw=2.5, label='median')
    ax.set_xlabel(r'assumed clump $\Delta v$ [km/s]')
    ax.set_ylabel(r'retrieved $\log n_{\rm H_2}$')
    ax.set_title('Retrieved density vs. assumed linewidth')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for _, row in merged.iterrows():
        ax.plot(dv_vals, [row[c] for c in chi2_cols], marker='o', ms=3, alpha=0.5, lw=1)
    ax.plot(dv_vals, med_chi2, marker='o', ms=8, color='black', lw=2.5, label='median')
    ax.set_xlabel(r'assumed clump $\Delta v$ [km/s]')
    ax.set_ylabel(r'$\chi^2$')
    ax.set_yscale('log')
    ax.set_title(r'Fit $\chi^2$ vs. assumed linewidth')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    consist_cols = [f'consistent_{dv.replace(".", "")}' for dv in DVS if dv in dfs]
    frac_consistent = [merged[c].mean() * 100 for c in consist_cols]
    ax.plot(dv_vals, frac_consistent, marker='o', ms=8, color='tab:purple', lw=2.5)
    ax.set_xlabel(r'assumed clump $\Delta v$ [km/s]')
    ax.set_ylabel('physically_consistent [%]')
    ax.set_ylim(0, 105)
    ax.set_title('Physical-consistency fraction vs. linewidth')
    ax.grid(alpha=0.3)

    plt.suptitle(f'Escape1d Delta_v sweep ({a.rates} rates): 23-position Stutzki retrieval', fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out_png = os.path.join(OUT_DIR, f'dv_sweep_comparison_{a.rates}.png')
    plt.savefig(out_png, dpi=160)
    print(f"\nsaved {out_png}")

    out_csv = os.path.join(OUT_DIR, f'dv_sweep_comparison_{a.rates}.csv')
    merged.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")


if __name__ == '__main__':
    main()
