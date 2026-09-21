"""Paper I, Work item 3: the real Fig. 5/6 comparison -- not against Eq. (11)
(the LTE no-anomaly floor, a different thing entirely), but against a direct
run of Stutzki & Winnewisser (1985)'s OWN escape-probability method, at his
own exact Fig. 5/6 parameters (T_k in {18,26,36} K, n'_H2 in
{10^3.5,10^5.0,10^7.0} cm^-3, N_NH3 in 10^13.7-10^15.1 cm^-2 at
Delta v = 0.3 km/s).

Three curves per panel, one panel per (T_k, n'_H2) combination (9 panels,
matching his own figure's own layout):
  (a) escape1d + full_original (95% Stutzki-sourced) rates -- a faithful
      stand-in for his own published curves, run at his exact parameters
      (stutzki85/run_fig5_6_grid.py, already computed).
  (b) the Magritte 3D NLTE "fig56" grid -- full non-LTE radiative transfer
      at the SAME (T_k, n'_H2, N/dv) grid points, with the tau_main and
      model.write() fixes already applied (w33/build_lut_fig56.py, already
      computed).
  (c) Eq. (11), the LTE no-anomaly reference (for context only -- both (a)
      and (b) include the trapping anomaly; Eq. 11 does not).

The quantitative comparison that matters is (a) vs (b): same physical
parameters, two different radiative-transfer treatments (1D escape
probability vs full 3D non-LTE). Reported per panel as the median/max
absolute ratio residual between the two curves, interpolating the escape1d
curve (80 points) onto the Magritte grid's own (sparser, 8-9 point) tau_main
values in log-tau space.

Caveat carried from reproduce_stutzki_fig5_6.py: the n'_H2=10^3.5 branch
implies large clump radii at high column (>0.15 pc rising toward ~10 pc at
the highest log_N_dv point, which also failed the Magritte convergence gate
and is excluded) -- well outside Stutzki's own ~0.01 pc clump picture. Kept
in rather than radius-masked here (unlike the general-grid script) because
the point of this comparison is the two radiative-transfer treatments at
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
ESCAPE1D_CSV = "/home/yasho379/magritte_rebuilt/production/output_escape1d/results_full_original_rates/fig56_curves.csv"
OUTDIR = "/home/yasho379/magritte_rebuilt/production/output_lut_fig56/results/fig5_6_direct_comparison/"

TEMPS = [18.0, 26.0, 36.0]
LOG_NS = [3.5, 5.0, 7.0]

# (panel title, magritte ratio col, escape1d ratio col, eq11 component, y label)
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
    return df


def load_escape1d():
    df = pd.read_csv(ESCAPE1D_CSV)
    df = df[df['converged'] & (df['tau_main'] > 0)]
    df['Avg_Inner_Ratio'] = (df['R_12'] + df['R_21']) / 2.0
    return df


def interp_residual(mag_tau, mag_ratio, esc_tau, esc_ratio):
    """Interpolate the (denser) escape1d curve onto the Magritte grid's own
    tau_main points, in log-tau space, and return per-point residuals."""
    order = np.argsort(esc_tau)
    esc_tau_s, esc_ratio_s = esc_tau[order], esc_ratio[order]
    log_esc_tau = np.log10(esc_tau_s)
    in_range = (mag_tau >= esc_tau_s.min()) & (mag_tau <= esc_tau_s.max())
    if in_range.sum() == 0:
        return np.array([]), np.array([])
    esc_at_mag = np.interp(np.log10(mag_tau[in_range]), log_esc_tau, esc_ratio_s)
    return mag_ratio[in_range] - esc_at_mag, mag_tau[in_range]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=OUTDIR)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    mag = load_magritte()
    esc = load_escape1d()
    print(f"Magritte fig56 grid: {len(mag)} converged rows")
    print(f"escape1d full_original fig56 curves: {len(esc)} converged rows")

    tau_ref = np.logspace(-2.2, 1.3, 300)
    summary_rows = []

    for obs_key, obs_title, mag_col_fixed, esc_col_fixed, eq11_comp, ylabel in OBSERVABLES:
        mag_col = mag_col_fixed or 'Avg_Inner_Ratio'
        esc_col = esc_col_fixed or 'Avg_Inner_Ratio'

        fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)
        for i, T in enumerate(TEMPS):
            for j, log_n in enumerate(LOG_NS):
                ax = axes[i, j]
                mag_sub = mag[np.isclose(mag['T_cloud'], T) & np.isclose(mag['log_n_H2'], log_n)]
                esc_sub = esc[np.isclose(esc['T_k'], T) & np.isclose(esc['log_n_H2'], log_n)]
                mag_sub = mag_sub.sort_values('tau_main')
                esc_sub = esc_sub.sort_values('tau_main')

                ax.plot(esc_sub['tau_main'], esc_sub[esc_col], color='tab:blue', lw=2.0,
                        label="escape1d, Stutzki's own rates" if (i == 0 and j == 0) else None)
                ax.plot(mag_sub['tau_main'], mag_sub[mag_col], color='tab:red', lw=1.6,
                        marker='o', ms=4, label='Magritte 3D NLTE' if (i == 0 and j == 0) else None)
                ax.plot(tau_ref, eq11_thermal_ratio(tau_ref, eq11_comp), color='black', lw=1.3,
                        ls='--', alpha=0.7, label='Eq. (11), no anomaly' if (i == 0 and j == 0) else None)

                resid, resid_tau = interp_residual(
                    mag_sub['tau_main'].values, mag_sub[mag_col].values,
                    esc_sub['tau_main'].values, esc_sub[esc_col].values)
                if len(resid) > 0:
                    med_abs = np.median(np.abs(resid))
                    max_abs = np.max(np.abs(resid))
                    ax.set_title(f"T={T:.0f}K, n'=$10^{{{log_n:.1f}}}$\n"
                                 f"|resid| median={med_abs:.3f} max={max_abs:.3f}", fontsize=9)
                else:
                    med_abs = max_abs = np.nan
                    ax.set_title(f"T={T:.0f}K, n'=$10^{{{log_n:.1f}}}$\nno overlap", fontsize=9)

                summary_rows.append(dict(observable=obs_key, T_k=T, log_n_H2=log_n,
                                          n_overlap=len(resid), median_abs_resid=med_abs,
                                          max_abs_resid=max_abs,
                                          n_magritte=len(mag_sub), n_escape1d=len(esc_sub)))

                ax.set_xscale('log')
                ax.set_ylim(0.0, 1.05)
                ax.grid(True, which='both', ls='--', alpha=0.25)
                if i == 2:
                    ax.set_xlabel(r'$\tau(\Delta F_1=0)$')
                if j == 0:
                    ax.set_ylabel(ylabel, fontsize=9)

        fig.legend(loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.02))
        plt.suptitle(f'{obs_title}: escape1d (Stutzki\'s own rates) vs. Magritte 3D NLTE, '
                     f'at his exact Fig. 5/6 parameters', y=1.06, fontsize=12)
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
    valid = summary.dropna(subset=['median_abs_resid'])
    print(f"overall median |residual| across all panels: {valid['median_abs_resid'].median():.4f}")
    print(f"overall max |residual| across all panels: {valid['max_abs_resid'].max():.4f}")


if __name__ == '__main__':
    main()
