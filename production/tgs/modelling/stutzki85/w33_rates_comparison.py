"""Direct comparison at W33: escape1d + Loreau rates vs escape1d + Stutzki's
own (full_original, 95% Stutzki-sourced) rates. Holds the radiative-transfer
method fixed (1D escape-probability) and swaps only the collisional rates --
the complementary test to the gold-vs-escape1d (method-swap, rates-fixed)
comparison already in the paper (sec:escape1d).

Reads the two W33 retrieval summaries already produced by
retrieve_w33_escape1d.py --rates {loreau,full_original} -- no new compute.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOREAU_CSV = "/home/yasho379/magritte_rebuilt/production/output_escape1d/results_loreau_rates/w33_retrieval/summary.csv"
FULL_ORIG_CSV = "/home/yasho379/magritte_rebuilt/production/output_escape1d/results_full_original_rates/w33_retrieval/summary.csv"
OUT_PNG = "/home/yasho379/magritte_rebuilt/production/output_escape1d/results/w33_rates_comparison.png"


def main():
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    lor = pd.read_csv(LOREAU_CSV)
    fo = pd.read_csv(FULL_ORIG_CSV)
    lor = lor[lor.fittable].set_index('source')
    fo = fo[fo.fittable].set_index('source')
    sources = list(lor.index)

    print(f"{'source':8s} {'quantity':22s} {'Loreau':>10s} {'orig.rates':>10s} {'observed':>10s}")
    for src in sources:
        print(f"{src:8s} {'log n_H2':22s} {lor.loc[src,'log_n_H2_fit']:10.3f} {fo.loc[src,'log_n_H2_fit']:10.3f} {'':>10s}")
        print(f"{src:8s} {'T_k [K]':22s} {lor.loc[src,'T_k_fit']:10.2f} {fo.loc[src,'T_k_fit']:10.2f} {'':>10s}")
        print(f"{src:8s} {'chi2':22s} {lor.loc[src,'chi2']:10.3f} {fo.loc[src,'chi2']:10.3f} {'':>10s}")
        print(f"{src:8s} {'eta_f':22s} {lor.loc[src,'eta_f']:10.3f} {fo.loc[src,'eta_f']:10.3f} {'':>10s}")
        print(f"{src:8s} {'(2,1)/(1,1) predicted':22s} {lor.loc[src,'R21_11_predicted']:10.4f} "
              f"{fo.loc[src,'R21_11_predicted']:10.4f} {lor.loc[src,'R21_11_observed']:10.4f}")
        print()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, src in zip(axes, sources):
        labels = ['Loreau\nrates', "Stutzki's own\nrates (95%)", 'Tursun+22\nobserved']
        vals = [lor.loc[src, 'R21_11_predicted'], fo.loc[src, 'R21_11_predicted'], lor.loc[src, 'R21_11_observed']]
        colors = ['tab:orange', 'tab:blue', 'black']
        ax.bar(labels, vals, color=colors, alpha=0.85)
        for i, v in enumerate(vals):
            ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        ax.set_title(f'{src}\n'
                     f'Loreau: $\\log n$={lor.loc[src,"log_n_H2_fit"]:.2f}, $T$={lor.loc[src,"T_k_fit"]:.1f}K, '
                     f'$\\chi^2$={lor.loc[src,"chi2"]:.3f}\n'
                     f"Stutzki: $\\log n$={fo.loc[src,'log_n_H2_fit']:.2f}, $T$={fo.loc[src,'T_k_fit']:.1f}K, "
                     f"$\\chi^2$={fo.loc[src,'chi2']:.3f}", fontsize=8.5)
        ax.set_ylabel(r'$T_B(2,1)/T_B(1,1)$', fontsize=9)
        ax.grid(alpha=0.2, axis='y')
    plt.suptitle('W33: escape-probability model, Loreau vs. Stutzki\'s own (full_original) rates\n'
                 '(same radiative-transfer method throughout -- only the collisional rates differ)',
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.88))
    plt.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PNG}")


if __name__ == '__main__':
    main()
