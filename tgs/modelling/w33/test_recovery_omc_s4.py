"""
End-to-end recovery test for the Stutzki-replicating inversion pipeline.

The question this answers: given a 5-ratio vector produced by our OWN forward
model at a known (T_k, n_H2, N_NH3/dv), can invert_ratio_vector find its way
back to those parameters -- and, where a degenerate branch scores better, does
stutzki_physics.physically_consistent() correctly reject it?

Ground truth: Stutzki & Winnewisser (1985) Table 1a, OMC S4
(T_k=27.82 K, n_H2=10^7.317 cm^-3, N_NH3=10^14.16 cm^-2) -- one of the '(a)'
positions whose fit includes the (2,2) hyperfine satellites, so the 5th ratio
(R_22_MAIN) is meaningful there, and already Stage-0-validated in
validate_stutzki.py.

IMPORTANT -- how the synthetic "observed" T_B is built, and why not Stutzki's
own T_B_obs:

  Our Magritte sphere at OMC S4's parameters gives A_MAIN = 1.487 K
  (tau_main = 0.039, i.e. very optically thin), against Stutzki's own
  T_B_theor = 10.714 K. That absolute-brightness disagreement is known and
  documented (see validate_stutzki.py's module docstring: his escape-
  probability treatment makes N_NH3/Delta_v the only geometry-sensitive
  quantity, decoupling R from n_H2/N_NH3 by construction, which our ray-traced
  3D sphere does not do). Feeding his real T_B_obs = 6.56 K against our
  A_MAIN = 1.487 K would give eta_f = 4.4, so the TRUE answer would fail the
  eta_f <= 1 check for a reason that has nothing to do with the inversion.

  So the synthetic observation is built self-consistently instead:
      T_B_obs := ETA_F_TRUE * A_MAIN(truth run)
  with ETA_F_TRUE = 0.612, Stutzki's own fitted filling factor for S4. In the
  synthetic world the truth then has eta_f = 0.612 by construction (passes),
  while a degenerate branch whose A_MAIN is much fainter gets eta_f > 1 and is
  correctly rejected. This is an ASSUMPTION made to keep the test internally
  consistent, not a derivation -- it tests the inversion machinery, NOT our
  model's absolute brightness scale against Stutzki's (that is a separate,
  known, unresolved discrepancy).

Run: python3 test_recovery_omc_s4.py
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_MODELLING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODELLING not in sys.path:
    sys.path.insert(0, _MODELLING)

import params as p
import stutzki_params as sp
from invert_ratios import invert_ratio_vector, RATIO_KEYS, DEFAULT_CLUMP_DV_KMS

FIELD, POSITION = 'OMC', 'S4'
ETA_F_TRUE = 0.612          # Stutzki's own fitted filling factor for this position
X_FIDUCIAL = 1e-8           # same fiducial abundance the inversion pipeline uses
MAX_NLTE = 200
PROCESSES = 14              # 16 cores, leave headroom
ODIR = "/home/yasho379/magritte_rebuilt/output_recovery_omc_s4/"


def truth_forward_run():
    """Forward-run the known answer once and return its 5-ratio vector."""
    from nh3_NLTE_sphere import run_model
    from nh3_NLTE_analysis import analyse_spectra
    from spectrum_utils import native_peak_tmb

    entry = sp.fit(FIELD, POSITION)
    T_k = entry['T_k']
    n_H2 = sp.n_H2_cm3(entry)
    N_NH3 = sp.N_NH3_cm2(entry)
    # Same convention as validate_stutzki.radius_for_target_N: N0 = 2*r*n*X.
    radius_sphere = N_NH3 / (2.0 * n_H2 * X_FIDUCIAL)
    vturb = p.fwhm_kms_to_vturb_ms(sp.CLUMP_DV_KMS)

    os.makedirs(os.path.join(ODIR, "fits"), exist_ok=True)
    os.makedirs(os.path.join(ODIR, "images"), exist_ok=True)

    print(f"[truth] running {FIELD} {POSITION}: T_k={T_k} K, n_H2={n_H2:.3e} cm^-3, "
          f"N_NH3={N_NH3:.3e} cm^-2, radius_sphere={radius_sphere:.3e}")
    conv = run_model(wdir="/home/yasho379/magritte_rebuilt/tgs/", odir=ODIR, XNH3=X_FIDUCIAL,
                      numberdensity=n_H2, vturb=vturb, T_cloud=T_k, max_NLTE=MAX_NLTE,
                      radius_sphere=radius_sphere)
    halting_iter, final_convergence, tau_main, extra_spectra, npoints, nboundary = conv
    hfs = analyse_spectra(odir=ODIR, XNH3=X_FIDUCIAL, numberdensity=n_H2, vturb=vturb,
                           T_cloud=T_k, radius_sphere=radius_sphere, max_NLTE=MAX_NLTE,
                           save_plots=False)
    velos22, Is22 = extra_spectra['22']
    A_MAIN_22, _, _ = native_peak_tmb(velos22, Is22, p.FREQ_HZ['2,2'])

    obs_ratios = {k: float(hfs[k]) for k in RATIO_KEYS if k in hfs}
    obs_ratios['R_22_MAIN'] = A_MAIN_22 / hfs['A_MAIN'] if hfs['A_MAIN'] else np.nan

    truth = dict(T_k=T_k, n_H2=n_H2, N_NH3=N_NH3, radius_sphere=radius_sphere,
                  A_MAIN=float(hfs['A_MAIN']), A_MAIN_22=float(A_MAIN_22),
                  tau_main=float(tau_main), final_convergence=float(final_convergence),
                  N_NH3_per_dv=N_NH3 / sp.CLUMP_DV_KMS, obs_ratios=obs_ratios)
    print(f"[truth] A_MAIN={truth['A_MAIN']:.4f} K, tau_main={truth['tau_main']:.4f}, "
          f"convergence={truth['final_convergence']:.2f}%")
    print(f"[truth] ratios: " + ", ".join(f"{k}={v:.4f}" for k, v in obs_ratios.items()))
    return truth


def _truth_in_subprocess():
    """Run the truth model in a child process so the parent never initialises
    Magritte/OpenMP. A parent that has already run a model cannot safely fork a
    Pool afterwards: the children inherit mutexes held by threads that do not
    exist in them and deadlock. invert_ratios now uses the spawn context, which
    also fixes this, but keeping the parent Magritte-free is belt and braces.
    """
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        return pool.apply(truth_forward_run)


def main():
    truth = _truth_in_subprocess()
    entry = sp.fit(FIELD, POSITION)

    T_B_obs = ETA_F_TRUE * truth['A_MAIN']   # see module docstring
    dv_obs_kms = entry['dv_obs']
    distance_pc = sp.SOURCE_DISTANCE_PC[FIELD]
    print(f"\n[synthetic obs] T_B_obs={T_B_obs:.4f} K (= {ETA_F_TRUE} * A_MAIN_truth), "
          f"dv_obs={dv_obs_kms} km/s, distance={distance_pc} pc")

    print("\n[invert] starting -- the truth is at log10(n_H2)="
          f"{np.log10(truth['n_H2']):.3f}, T_k={truth['T_k']}, "
          f"log10(N/dv)={np.log10(truth['N_NH3_per_dv']):.3f}\n")

    result = invert_ratio_vector(
        truth['obs_ratios'],
        clump_dv_kms=DEFAULT_CLUMP_DV_KMS,
        XNH3_fiducial=X_FIDUCIAL,
        odir=ODIR,
        out_csv=os.path.join(ODIR, "results", "recovery_omc_s4.csv"),
        processes=PROCESSES,
        max_NLTE=MAX_NLTE,
        T_B_obs=T_B_obs,
        dv_obs_kms=dv_obs_kms,
        distance_pc=distance_pc,
    )

    print("\n" + "=" * 70)
    print("RECOVERY TEST RESULT")
    print("=" * 70)
    print(f"truth:  n_H2={truth['n_H2']:.3e}  T_k={truth['T_k']}  "
          f"N/dv={truth['N_NH3_per_dv']:.3e}")

    def _report(label, row):
        if row is None:
            print(f"{label}: None")
            return
        n, T = float(row['numberdensity']), float(row['T_cloud'])
        Ndv = float(row['N_NH3_target']) / float(row['clump_dv'])
        print(f"{label}: n_H2={n:.3e} ({np.log10(n) - np.log10(truth['n_H2']):+.2f} dex)  "
              f"T_k={T:.2f} ({T - truth['T_k']:+.2f} K)  "
              f"N/dv={Ndv:.3e} ({np.log10(Ndv) - np.log10(truth['N_NH3_per_dv']):+.2f} dex)  "
              f"chi2={row['score']:.4f}")
        for k in ('check_eta_f', 'check_K_predicted', 'check_K_required'):
            if k in row:
                print(f"    {k}={row[k]}")

    _report("best (raw chi2)      ", result.get('best'))
    _report("best (phys. consist.)", result.get('best_physically_consistent'))
    print(f"rounds_run={result['rounds_run']}  converged_by_score={result['converged_by_score']}")
    print(f"csv: {result['out_csv']}")

    summary_path = os.path.join(ODIR, "results", "recovery_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(dict(
            truth=truth,
            T_B_obs=T_B_obs, dv_obs_kms=dv_obs_kms, distance_pc=distance_pc,
            rounds_run=result['rounds_run'],
            converged_by_score=result['converged_by_score'],
            best={k: str(v) for k, v in (result.get('best') or {}).items()},
            best_physically_consistent={k: str(v) for k, v in
                                          (result.get('best_physically_consistent') or {}).items()},
        ), f, indent=2, default=str)
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
