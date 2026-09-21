"""Full chi^2 retrieval diagnostic against W33 (Tursun et al. 2022), using
the interpolated gold LUT -- same methodology and same diagnostic layout as
retrieve_stutzki1985_full.py's 23-position Stutzki analysis, applied to W33
so it gets the same depth of treatment rather than the "both pin against
the ceiling" one-line summary the paper carried before this script existed.

Only sources that pass the model-applicability quadrant gate
(w33_observed_ratios.quadrant_gate) are fit -- currently W33_A and W33_B
only; Main1/A1/B1 show a kinematic (infall/expansion) signature no static
sphere can reproduce at any parameters, so fitting them would be
meaningless regardless of chi^2.

For each fittable source, produces:
  1. The observed (1,1) spectrum reconstructed from the digitized/Table-4
     ratios, at the source's own observed linewidth.
  2. The best-fit model spectrum from the interpolated LUT, at the model's
     own intrinsic clump linewidth (0.3 km/s).
  2b. Best-fit (log_n_H2, T_k, log_N_dv), chi^2, eta_f, K predicted vs
      required, and physically_consistent() verdict.
  3. Profile-minimized chi^2 landscape in all three orthogonal 2D slices.
  4. Predicted (2,1)/(1,1) at the best fit vs the real Tursun et al. (2022)
     Table 4 (2,1) detection where available (W33_A, W33_B both have one)
     -- a genuine held-out test, exactly like Stutzki's Table 3 comparison,
     since (2,1) never enters the fit.

One PNG per fittable source plus a summary CSV, in
output_lut_gold/results/w33_retrieval/.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import params as p
import nh3_hyperfine as hf
import w33_observed_ratios as wr
from lut_interpolator import LutInterpolator, chi2_retrieve, RATIO_KEYS, RATIO_21
from stutzki_physics import eta_f as calc_eta_f, clump_count_K, clump_count_required, physically_consistent
from retrieve_stutzki1985_full import gaussian_spectrum, chi2_slice_plot, OFFSETS

DV_CLUMP_KMS = 0.3
OUTDIR = "/home/yasho379/magritte_rebuilt/production/output_lut_gold/results/w33_retrieval/"
QGRID = dict(n_log_n=200, n_T=73, n_log_Ndv=90)


def _to_vec(d, keys):
    return np.array([np.nan if d[k] is None else d[k] for k in keys])


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("building interpolator + dense query grid...", flush=True)
    interp = LutInterpolator()
    pts = interp.query_grid(**QGRID)
    ratio_grid = interp.predict_ratios(pts)
    log_n_ax = np.linspace(interp.axis_lo[0], interp.axis_hi[0], QGRID['n_log_n'])
    T_ax = np.linspace(interp.axis_lo[1], interp.axis_hi[1], QGRID['n_T'])
    ndv_ax = np.linspace(interp.axis_lo[2], interp.axis_hi[2], QGRID['n_log_Ndv'])
    shape3d = (QGRID['n_log_n'], QGRID['n_T'], QGRID['n_log_Ndv'])
    print(f"  {len(interp.df)} LUT rows, {len(pts)} query points", flush=True)

    summary = []
    for source in wr.W33_SOURCES:
        gate = wr.quadrant_gate(source)
        print(f"=== {source}: quadrant {gate['quadrant']} ({gate['label']}) ===", flush=True)
        if gate['quadrant'] != 'II':
            print(f"  excluded: not fittable by a static sphere at any parameters", flush=True)
            summary.append(dict(source=source, fittable=False, quadrant=gate['quadrant'],
                                 HIA_IS=gate['HIA_IS'], HIA_OS=gate['HIA_OS']))
            continue

        obs_full, err_full, meta = wr.observed_ratio_vector(source)
        obs_vec = _to_vec(obs_full, RATIO_KEYS)
        sigma_vec = _to_vec(err_full, RATIO_KEYS)

        best_idx, chi2_best, chi2_all = chi2_retrieve(interp, pts, ratio_grid, obs_vec, sigma_vec)
        log_n_fit, T_fit, ndv_fit = pts[best_idx]
        A_MAIN_fit = float(ratio_grid['A_MAIN'][best_idx])
        R21_11_pred = float(ratio_grid[RATIO_21][best_idx])
        model_ratios = {k: float(ratio_grid[k][best_idx]) for k in RATIO_KEYS}

        T_B_obs = p.LINES[source]['1,1']['T_mb']
        dv_obs = p.LINES[source]['1,1']['dv']
        dist = p.DISTANCE_PC
        ef = calc_eta_f(T_B_obs, A_MAIN_fit)
        K_req = clump_count_required(dv_obs, DV_CLUMP_KMS)
        K_pred = clump_count_K(T_fit, 10 ** log_n_fit, ef, dist, dv_obs, DV_CLUMP_KMS) if ef and ef > 0 else np.nan
        consistent, detail = physically_consistent(T_fit, 10 ** log_n_fit, T_B_obs, A_MAIN_fit,
                                                    dv_obs, DV_CLUMP_KMS, dist)

        v = np.linspace(-32, 32, 800)
        obs_amps_dict = {'R_10_MAIN': obs_full['R_10_MAIN'], 'R_12_MAIN': obs_full['R_12_MAIN'],
                          'R_21_MAIN': obs_full['R_21_MAIN'], 'R_01_MAIN': obs_full['R_01_MAIN']}
        obs_amps = [T_B_obs * (obs_amps_dict['R_10_MAIN'] or 0), T_B_obs * (obs_amps_dict['R_12_MAIN'] or 0),
                    T_B_obs, T_B_obs * (obs_amps_dict['R_21_MAIN'] or 0), T_B_obs * (obs_amps_dict['R_01_MAIN'] or 0)]
        obs_spec = gaussian_spectrum(v, obs_amps, dv_obs)
        model_amps = [A_MAIN_fit * model_ratios['R_10_MAIN'], A_MAIN_fit * model_ratios['R_12_MAIN'], A_MAIN_fit,
                      A_MAIN_fit * model_ratios['R_21_MAIN'], A_MAIN_fit * model_ratios['R_01_MAIN']]
        model_spec = gaussian_spectrum(v, model_amps, DV_CLUMP_KMS)

        R21_11_obs = obs_full.get('R_21_11')

        chi2_3d = chi2_all.reshape(shape3d)

        fig = plt.figure(figsize=(21, 10))
        gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.9)

        ax_spec = fig.add_subplot(gs[0, 0:3])
        ax_spec.plot(v, obs_spec, color='k', lw=1.6, label=f'observed (FWHM={dv_obs:.2f} km/s)')
        ax_spec.plot(v, model_spec, color='tab:blue', lw=1.4, ls='--',
                    label=f'best-fit model (FWHM={DV_CLUMP_KMS} km/s)')
        ax_spec.set_xlabel('v [km/s]', fontsize=9)
        ax_spec.set_ylabel(r'$T_B$ [K]', fontsize=9)
        ax_spec.set_title(f'{source}: (1,1) spectrum', fontsize=10)
        ax_spec.legend(fontsize=7)
        ax_spec.grid(alpha=0.25)

        obs_spec_norm = gaussian_spectrum(v, [x / T_B_obs for x in obs_amps], dv_obs)
        model_spec_norm = gaussian_spectrum(v, [x / A_MAIN_fit for x in model_amps], DV_CLUMP_KMS)
        ax_spec_n = fig.add_subplot(gs[0, 3:6])
        ax_spec_n.plot(v, obs_spec_norm, color='k', lw=1.6, label='observed / $T_{B,obs}$')
        ax_spec_n.plot(v, model_spec_norm, color='tab:blue', lw=1.4, ls='--', label='model / $T_{B,theor}$')
        ax_spec_n.axhline(1.0, color='gray', lw=0.6, ls=':')
        ax_spec_n.set_xlabel('v [km/s]', fontsize=9)
        ax_spec_n.set_ylabel(r'$T_B\,/\,T_B(\mathrm{main})$', fontsize=9)
        ax_spec_n.set_title('main-line-normalized (ratio shape)', fontsize=10)
        ax_spec_n.legend(fontsize=7)
        ax_spec_n.grid(alpha=0.25)

        ax_txt = fig.add_subplot(gs[0, 6:9])
        ax_txt.axis('off')
        lines = [
            f"$\\chi^2$ = {chi2_best:.3f}",
            "",
            f"log $n_{{H_2}}$ = {log_n_fit:.3f}",
            f"$T_k$ = {T_fit:.2f} K",
            f"log($N_{{NH_3}}/\\Delta v$) = {ndv_fit:.3f}",
            "",
            f"$T_{{B,obs}}$ = {T_B_obs:.3f} K   $T_{{B,theor}}$ = {A_MAIN_fit:.3f} K",
            f"$\\eta_f$ = {ef:.3f}" + ("  (>1, UNPHYSICAL)" if (ef and ef > 1) else ""),
            f"K predicted = {K_pred:.2f}   K required = {K_req:.2f}",
            f"physically consistent: {'YES' if consistent else 'NO'}",
        ]
        ax_txt.text(0.02, 0.95, "\n".join(lines), va='top', ha='left', fontsize=9.5,
                    transform=ax_txt.transAxes, family='monospace')

        ax21 = fig.add_subplot(gs[0, 9:12])
        labels, vals, colors = ['predicted'], [R21_11_pred], ['tab:blue']
        if R21_11_obs is not None:
            labels += ['Tursun+22\nobserved']
            vals += [R21_11_obs]
            colors += ['black']
        ax21.bar(labels, vals, color=colors, alpha=0.85)
        ax21.set_ylabel(r'$T_B(2,1)/T_B(1,1)$', fontsize=9)
        ax21.set_title('(2,1)/(1,1): held-out prediction', fontsize=10)
        for i, val in enumerate(vals):
            ax21.text(i, val, f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        if R21_11_obs is None:
            ax21.text(0.5, 0.5, 'no (2,1) detection\nfor this source', ha='center', va='center',
                      transform=ax21.transAxes, fontsize=8, color='gray')

        chi2_slice_plot(fig.add_subplot(gs[1, 0:4]), log_n_ax, T_ax, chi2_3d, 2,
                        r'log $n_{H_2}$', r'$T_k$ [K]', (log_n_fit, T_fit), chi2_best,
                        None, title=r'$T_k$ vs log($n_{H_2}$), profile min over log(N/dv)')
        chi2_slice_plot(fig.add_subplot(gs[1, 4:8]), T_ax, ndv_ax, np.moveaxis(chi2_3d, 0, -1), 2,
                        r'$T_k$ [K]', r'log($N_{NH_3}/\Delta v$)', (T_fit, ndv_fit), chi2_best,
                        None, title=r'log(N/dv) vs $T_k$, profile min over log($n_{H_2}$)')
        chi2_slice_plot(fig.add_subplot(gs[1, 8:12]), log_n_ax, ndv_ax, chi2_3d, 1,
                        r'log $n_{H_2}$', r'log($N_{NH_3}/\Delta v$)', (log_n_fit, ndv_fit), chi2_best,
                        None, title=r'log(N/dv) vs log($n_{H_2}$), profile min over $T_k$')

        out_png = os.path.join(OUTDIR, f'{source}.png')
        plt.savefig(out_png, dpi=140)
        plt.close(fig)
        print(f"  saved {out_png}  chi2={chi2_best:.3f}  log_n={log_n_fit:.3f}  T={T_fit:.1f}  "
              f"eta_f={ef:.3f}  consistent={consistent}", flush=True)

        summary.append(dict(
            source=source, fittable=True, quadrant=gate['quadrant'],
            HIA_IS=gate['HIA_IS'], HIA_OS=gate['HIA_OS'],
            chi2=chi2_best, log_n_H2_fit=log_n_fit, T_k_fit=T_fit, log_N_dv_fit=ndv_fit,
            T_B_obs=T_B_obs, T_B_theor=A_MAIN_fit, eta_f=ef,
            K_predicted=K_pred, K_required=K_req, physically_consistent=consistent,
            R21_11_predicted=R21_11_pred, R21_11_observed=R21_11_obs,
            pinned_log_n_lo=np.isclose(log_n_fit, interp.axis_lo[0], atol=0.05),
            pinned_log_n_hi=np.isclose(log_n_fit, interp.axis_hi[0], atol=0.05),
            pinned_T_lo=np.isclose(T_fit, interp.axis_lo[1], atol=0.3),
            pinned_T_hi=np.isclose(T_fit, interp.axis_hi[1], atol=0.3),
            pinned_ndv_lo=np.isclose(ndv_fit, interp.axis_lo[2], atol=0.02),
            pinned_ndv_hi=np.isclose(ndv_fit, interp.axis_hi[2], atol=0.02),
        ))

    df = pd.DataFrame(summary)
    csv_path = os.path.join(OUTDIR, 'summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nsaved {csv_path}  ({len(df)} sources, {df.fittable.sum()} fittable)")
    fit = df[df.fittable]
    if len(fit):
        print(f"median chi2 = {fit.chi2.median():.3f}")
        print(f"physically_consistent: {fit.physically_consistent.mean():.1%}")


if __name__ == '__main__':
    main()
