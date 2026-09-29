"""Paper I, Work item 3: the real Fig. 5/6 comparison -- not against Eq. (11)
(the LTE no-anomaly floor, a different thing entirely), but against direct
runs of Stutzki & Winnewisser (1985)'s OWN escape-probability method, at his
own exact Fig. 5/6 parameters (T_k in {18,26,36} K, n'_H2 in
{10^3.5,10^5.0,10^7.0} cm^-3, N_NH3 in 10^13.7-10^15.1 cm^-2 at
Delta v = 0.3 km/s).

Four curves per panel, one panel per (T_k, n'_H2) combination (9 panels,
matching his own figure's own layout):
  (a) escape1d + full_original (95% Stutzki-sourced) rates -- a faithful
      stand-in for his own published curves, run at his exact parameters.
  (b) escape1d + Loreau et al. (2023) rates -- the modern hyperfine-
      resolved NH3-H2 rates used throughout the rest of this project's
      escape1d work (and what Magritte's own 3D NLTE leg uses internally).
      Same method as (a), different rates -- isolates the rates' own
      contribution, complementary to the gold-vs-escape1d method-swap
      comparison already in the paper.
  (c) the Magritte 3D NLTE "fig56" grid -- full non-LTE radiative transfer
      at the SAME (T_k, n'_H2, N/dv) grid points, with the tau_main and
      model.write() fixes already applied (w33/build_lut_fig56.py).
  (d) Eq. (11), the LTE no-anomaly reference (for context only -- (a),(b)
      and (c) all include the trapping anomaly; Eq. 11 does not).

(stutzki85/run_fig5_6_grid.py --rates {loreau,full_original} computes (a)
and (b); both already run.)

Three quantitative residuals reported per panel (interpolating each denser
escape1d curve onto Magritte's own sparser tau_main points in log-tau
space): full_original-vs-Magritte and loreau-vs-Magritte (method-fixed
rates-vs-Magritte, i.e. how much either rate choice differs from the full
3D treatment), and loreau-vs-full_original (rates-only, method fixed --
the complementary test).

Caveat carried from reproduce_stutzki_fig5_6.py: the n'_H2=10^3.5 branch
implies large clump radii at high column (>0.15 pc rising toward ~10 pc at
the highest log_N_dv point, which also failed the Magritte convergence gate
and is excluded) -- well outside Stutzki's own ~0.01 pc clump picture. Kept
in rather than radius-masked here (unlike the general-grid script) because
the point of this comparison is the radiative-transfer/rates treatments at
IDENTICAL (T,n,N/dv), not enforcing physical plausibility of the implied
geometry -- but it means the n'=3.5 panels should be read as a mathematical
comparison, not a claim about real sub-arcsecond clumps that dense.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_STUTZKI85 = os.path.join(os.path.dirname(_HERE), 'stutzki85')
sys.path.insert(0, _HERE)
sys.path.insert(0, _STUTZKI85)

from stutzki_physics import eq11_thermal_ratio

MAGRITTE_CSV = "/home/yasho379/magritte_rebuilt/production/output_lut_fig56/results/lut_dv0.30.csv"
ESCAPE1D_CSV = {
    'full_original': "/home/yasho379/magritte_rebuilt/production/output_escape1d/results_full_original_rates/fig56_curves.csv",
    'loreau': "/home/yasho379/magritte_rebuilt/production/output_escape1d/results_loreau_rates/fig56_curves.csv",
}
OUTDIR = "/home/yasho379/magritte_rebuilt/production/output_lut_fig56/results/fig5_6_direct_comparison/"

TEMPS = [18.0, 26.0, 36.0]
LOG_NS = [3.5, 5.0, 7.0]

ESCAPE1D_STYLE = {
    'full_original': dict(color='tab:blue', lw=2.0, label="escape1d, Stutzki's own rates"),
    'loreau': dict(color='tab:green', lw=2.0, ls=(0, (4, 1.5)), label='escape1d, Loreau rates'),
}

# (panel key, panel title, magritte ratio col, escape1d ratio col, eq11 component, y label)
OBSERVABLES = [
    ('outer_01', r'$F_1=0\to1$ (outer)', 'R_01_MAIN', 'R_01', 'outer',
     r'$T_B(F_1{=}0{\to}1)/T_B(\Delta F_1{=}0)$'),
    ('outer_10', r'$F_1=1\to0$ (outer)', 'R_10_MAIN', 'R_10', 'outer',
     r'$T_B(F_1{=}1{\to}0)/T_B(\Delta F_1{=}0)$'),
    ('inner_avg', 'Averaged inner satellites', None, None, 'inner',
     r'$[T_B(1{\to}2)+T_B(2{\to}1)]/2T_B(\Delta F_1{=}0)$'),
]


def load_magritte():
    df = pd.read_csv(MAGRITTE_CSV)
    df = df[(df['Status'] == 'SUCCESS') & df['convergence_ok'] & (df['tau_main'] > 0)]
    df['Avg_Inner_Ratio'] = (df['R_12_MAIN'] + df['R_21_MAIN']) / 2.0
    # Magritte's tau_main (post-fix) is the central-pixel (1,1) optical depth,
    # i.e. the CENTRAL-CHORD tau -- the same quantity as Stutzki's Fig. 5/6
    # x-axis (important_notes/mesh-comparison-and-fig56-2026-09-16.md sec 1).
    df['tau_chord'] = df['tau_main']
    return df


def eq11_on_chord_axis(tau_chord, component):
    """eq11_thermal_ratio takes Stutzki's RADIAL tau (his tau_G, with 2 tau
    in the emergent intensity), so on a central-chord axis evaluate it at
    tau_chord / 2."""
    return eq11_thermal_ratio(np.asarray(tau_chord) / 2.0, component)


def load_escape1d(rates):
    df = pd.read_csv(ESCAPE1D_CSV[rates])
    df = df[df['converged'] & (df['tau_main'] > 0)]
    df['Avg_Inner_Ratio'] = (df['R_12'] + df['R_21']) / 2.0
    # escape1d's tau_main is Stutzki's radial tau_G; its central chord is 2x.
    df['tau_chord'] = 2.0 * df['tau_main']
    return df


def interp_residual(ref_tau, ref_ratio, other_tau, other_ratio):
    """Interpolate the (denser) `other` curve onto `ref`'s own tau points,
    in log-tau space, and return per-point residuals (ref - other)."""
    order = np.argsort(other_tau)
    other_tau_s, other_ratio_s = other_tau[order], other_ratio[order]
    log_other_tau = np.log10(other_tau_s)
    in_range = (ref_tau >= other_tau_s.min()) & (ref_tau <= other_tau_s.max())
    if in_range.sum() == 0:
        return np.array([])
    other_at_ref = np.interp(np.log10(ref_tau[in_range]), log_other_tau, other_ratio_s)
    return ref_ratio[in_range] - other_at_ref


def summarize_residual(resid):
    if len(resid) == 0:
        return np.nan, np.nan, 0
    return np.median(np.abs(resid)), np.max(np.abs(resid)), len(resid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=OUTDIR)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    mag = load_magritte()
    esc = {rates: load_escape1d(rates) for rates in ESCAPE1D_CSV}
    print(f"Magritte fig56 grid: {len(mag)} converged rows")
    for rates, df in esc.items():
        print(f"escape1d {rates} fig56 curves: {len(df)} converged rows")

    tau_ref = np.logspace(-2.2, 1.6, 300)
    summary_rows = []

    for obs_key, obs_title, mag_col_fixed, esc_col_fixed, eq11_comp, ylabel in OBSERVABLES:
        mag_col = mag_col_fixed or 'Avg_Inner_Ratio'
        esc_col = esc_col_fixed or 'Avg_Inner_Ratio'

        fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)
        for i, T in enumerate(TEMPS):
            for j, log_n in enumerate(LOG_NS):
                ax = axes[i, j]
                mag_sub = mag[np.isclose(mag['T_cloud'], T) & np.isclose(mag['log_n_H2'], log_n)].sort_values('tau_chord')
                esc_sub = {rates: df[np.isclose(df['T_k'], T) & np.isclose(df['log_n_H2'], log_n)].sort_values('tau_chord')
                           for rates, df in esc.items()}

                first_panel = (i == 0 and j == 0)
                for rates, sub in esc_sub.items():
                    style = dict(ESCAPE1D_STYLE[rates])
                    if not first_panel:
                        style['label'] = None
                    ax.plot(sub['tau_chord'], sub[esc_col], **style)
                ax.plot(mag_sub['tau_chord'], mag_sub[mag_col], color='tab:red', lw=1.6,
                        marker='o', ms=4, label='Magritte 3D NLTE' if first_panel else None)
                ax.plot(tau_ref, eq11_on_chord_axis(tau_ref, eq11_comp), color='black', lw=1.3,
                        ls=':', alpha=0.7, label='Eq. (11), no anomaly' if first_panel else None)

                resid_fo = interp_residual(mag_sub['tau_chord'].values, mag_sub[mag_col].values,
                                            esc_sub['full_original']['tau_chord'].values,
                                            esc_sub['full_original'][esc_col].values)
                resid_lo = interp_residual(mag_sub['tau_chord'].values, mag_sub[mag_col].values,
                                            esc_sub['loreau']['tau_chord'].values,
                                            esc_sub['loreau'][esc_col].values)
                resid_rates = interp_residual(esc_sub['loreau']['tau_chord'].values, esc_sub['loreau'][esc_col].values,
                                               esc_sub['full_original']['tau_chord'].values,
                                               esc_sub['full_original'][esc_col].values)

                med_fo, max_fo, n_fo = summarize_residual(resid_fo)
                med_lo, max_lo, n_lo = summarize_residual(resid_lo)
                med_rates, max_rates, n_rates = summarize_residual(resid_rates)

                ax.set_title(f"T={T:.0f}K, n'=$10^{{{log_n:.1f}}}$\n"
                             f"vs Magritte: full_orig med={med_fo:.3f}, loreau med={med_lo:.3f}",
                             fontsize=8.5)

                for rates, med, mx, n in [('full_original_vs_magritte', med_fo, max_fo, n_fo),
                                          ('loreau_vs_magritte', med_lo, max_lo, n_lo),
                                          ('loreau_vs_full_original', med_rates, max_rates, n_rates)]:
                    summary_rows.append(dict(observable=obs_key, T_k=T, log_n_H2=log_n, comparison=rates,
                                              n_overlap=n, median_abs_resid=med, max_abs_resid=mx))

                ax.set_xscale('log')
                ax.set_ylim(0.0, 1.05)
                ax.grid(True, which='both', ls='--', alpha=0.25)
                if i == 2:
                    ax.set_xlabel(r'central-chord $\tau(\Delta F_1=0)$')
                if j == 0:
                    ax.set_ylabel(ylabel, fontsize=9)

        fig.legend(loc='upper center', ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, 1.03))
        plt.suptitle(f'{obs_title}: escape1d (both rate sets) vs. Magritte 3D NLTE, '
                     f'at his exact Fig. 5/6 parameters', y=1.07, fontsize=12)
        plt.tight_layout()
        out_png = os.path.join(a.out_dir, f'fig5_6_direct_{obs_key}.png')
        plt.savefig(out_png, dpi=170, bbox_inches='tight')
        plt.close(fig)
        print(f"saved {out_png}")

    summary = pd.DataFrame(summary_rows)
    out_csv = os.path.join(a.out_dir, 'residual_summary.csv')
    summary.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")
    print()
    print(summary.to_string(index=False))
    print()
    for comparison in summary['comparison'].unique():
        valid = summary[(summary['comparison'] == comparison)].dropna(subset=['median_abs_resid'])
        print(f"{comparison}: overall median |residual| = {valid['median_abs_resid'].median():.4f}, "
              f"max = {valid['max_abs_resid'].max():.4f}")


if __name__ == '__main__':
    main()
