"""
Stage 0: validate the Magritte NLTE + hyperfine-Gaussian-fit pipeline against
Stutzki & Winnewisser (1985)'s own published fits, before trusting it on W33.

For each benchmark position (Table 1a: T_k, n_H2, N_NH3), Stutzki's escape-
probability radiative transfer treats N_NH3/Δv as the only geometry-sensitive
quantity -- his own Sect. 3 notes this uses the escape probability at the cloud
*center* as representative for the whole cloud, which makes his optical depth
(tau_G = sum(k_g) * R) depend only on N_NH3 (since k_g itself scales as N_NH3/R),
decoupling R from n_H2/N_NH3 by construction. Magritte's actual ray-traced 3D
solution has no such shortcut. A cheap, zero-new-compute check on the existing
W33 catalogue (see session notes) already showed that at fixed peak column
N0 = 2*r*n*X, tau_main/T_mb swing by factors of 2-4x as the (n, r) split varies,
even though hyperfine ratios stay nearly flat -- i.e. the geometry *does* matter
in full 3D NLTE, contrary to what Stutzki's approximation assumes.

Stutzki's benchmark densities (10^6-10^7 cm^-3) are 100-1000x the (1,1) critical
density (~2e3 cm^-3, from Danby et al. 1988's A/C used in params.py), so his gas
is essentially fully thermalized regardless of geometry -- unlike W33's
n(H2)~10^4 cm^-3, which sits right at the sub-to-full-thermalization transition.
This module tests that directly: for each benchmark, sweep the (X, r) split at
his own fixed (T_k, n_H2, N_NH3) and see whether Magritte agrees that it
shouldn't matter (validating the approximation in his regime) or disagrees
(explaining why W33 needs the fuller treatment).
"""

import os
import sys
import csv
import multiprocessing
import numpy as np

import params as p
import stutzki_params as sp

WDIR = "/home/yasho379/magritte_rebuilt/production/tgs/"
ODIR = "/home/yasho379/magritte_rebuilt/scratch/output/output_stutzki_validation/"

# Radius choices for the geometry-sensitivity sweep, expressed as the implied
# NH3 abundance X (since N_NH3 = 2*r*n*X is fixed, sweeping X sweeps r inversely).
# Spans ~3 orders of magnitude in linear size.
X_SWEEP = [1e-9, 1e-8, 1e-7]

# The existing 878-row W33 catalogue used max_NLTE=300 and, per an empirical check
# against its own CSV (session notes), rows with tau > 5 -- comparable to Stutzki's
# benchmark targets -- average ~253 iterations and often hit this cap at only
# ~99% (not exact) convergence. That's the project's own accepted convention for
# this regime, not a bug to chase further; match it rather than guessing higher.
DEFAULT_MAX_NLTE = 300


def radius_for_target_N(n_H2_cm3, X, N_NH3_cm2):
    """Solve `radius_sphere` (the cm-convention run_model expects -- see
    params.catalogue_radius_cm_to_m's docstring) for N0 = 2*r*n*X = N_NH3.
    NOT `r_cm * 100`: run_model does `r_out[m] = radius_sphere/100`, and
    N_target[cm^-2] = 2*r_out[m]*n[m^-3]*X/1e4 (see params.catalogue_peak_column_cm2)
    algebraically reduces to `radius_sphere = N_NH3_cm2 / (2*n_H2_cm3*X)` exactly --
    verified against that already-tested function rather than re-derived loosely.
    """
    return N_NH3_cm2 / (2.0 * n_H2_cm3 * X)


def run_one(field, position, entry, X, max_NLTE=DEFAULT_MAX_NLTE):
    # Local import (needed per-worker under multiprocessing) + path fixup so
    # `nh3_NLTE_sphere`/`nh3_NLTE_analysis` (one directory up, in tgs/modelling/)
    # resolve regardless of the multiprocessing start method.
    modelling_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if modelling_dir not in sys.path:
        sys.path.insert(0, modelling_dir)
    from nh3_NLTE_sphere import run_model
    from nh3_NLTE_analysis import analyse_spectra

    T_k = entry['T_k']
    n_H2 = sp.n_H2_cm3(entry)
    N_NH3 = sp.N_NH3_cm2(entry)
    # Stutzki's own fixed clump linewidth (Table 1a caption: "Delta-v = 0.3 km/s"),
    # NOT entry['dv_obs'] (the broad, ensemble/blended observed linewidth). Session
    # finding: using the observed linewidth as vturb dilutes tau by roughly the
    # ratio of the two linewidths (~5-7x here) -- the selective-trapping anomaly
    # mechanism this whole module exists to test requires the narrow clump width,
    # see run_grid_w33.py's module docstring for the full explanation.
    vturb = p.fwhm_kms_to_vturb_ms(sp.CLUMP_DV_KMS)
    radius_sphere = radius_for_target_N(n_H2, X, N_NH3)

    os.makedirs(os.path.join(ODIR, "fits"), exist_ok=True)
    os.makedirs(os.path.join(ODIR, "images"), exist_ok=True)

    conv = run_model(wdir=WDIR, odir=ODIR, XNH3=X, numberdensity=n_H2, vturb=vturb,
                      T_cloud=T_k, max_NLTE=max_NLTE, radius_sphere=radius_sphere)
    tau_main = conv[2]
    hfs = analyse_spectra(odir=ODIR, XNH3=X, numberdensity=n_H2, vturb=vturb,
                           T_cloud=T_k, radius_sphere=radius_sphere,
                           max_NLTE=max_NLTE, save_plots=False)
    N_dv_proxy = hfs.pop('N_NH3')  # analyse_spectra's Sobolev n*X*r/vturb proxy, NOT a column

    theta_arcsec = p.m_to_arcsec(radius_sphere / 100.0,
                                  distance_pc=sp.SOURCE_DISTANCE_PC[field])
    return dict(field=field, position=position, X=X, n_H2=n_H2, N_NH3_target=N_NH3,
                N_dv_proxy=N_dv_proxy, tau_main=tau_main,
                T_k=T_k, radius_sphere=radius_sphere, theta_arcsec=theta_arcsec,
                T_B_theor_stutzki=entry['T_B_theor'], T_B_obs=entry['T_B_obs'],
                eta_f_stutzki=entry['eta_f'], **hfs)


def run_subset(entries=None, x_sweep=X_SWEEP, max_NLTE=DEFAULT_MAX_NLTE):
    """Serial version -- run the geometry-sensitivity sweep in-process. Use
    run_subset_parallel for the real (many-hour) Stage 0 batch."""
    entries = entries if entries is not None else list(sp.iter_stage_0_subset())
    results = []
    for field, position, entry in entries:
        print(f"=== {field} {position}: T_k={entry['T_k']:.1f} K, "
              f"n_H2={sp.n_H2_cm3(entry):.2e} cm^-3, N_NH3={sp.N_NH3_cm2(entry):.2e} cm^-2, "
              f"Stutzki T_B_theor={entry['T_B_theor']:.2f} K, eta_f={entry['eta_f']:.3f} ===")
        for X in x_sweep:
            r = run_one(field, position, entry, X, max_NLTE=max_NLTE)
            results.append(r)
            print(f"  X={X:.1e} r={r['theta_arcsec']:.3f}\" "
                  f"A_MAIN={r['A_MAIN']:.3f} tau_main={r['tau_main']:.3f} "
                  f"R_01_MAIN={r['R_01_MAIN']:.3f}")
    return results


def _worker(task):
    """multiprocessing.Pool target -- matches ModelGrid.py's worker_task pattern
    (thread-limiting env vars set before any magritte-touching import happens in
    this process)."""
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    field, position, entry, X, max_NLTE = task
    try:
        r = run_one(field, position, entry, X, max_NLTE=max_NLTE)
        return dict(status="SUCCESS", **r)
    except Exception as e:
        return dict(status="FAILED", field=field, position=position, X=X, error=str(e))


def run_subset_parallel(entries=None, x_sweep=X_SWEEP, max_NLTE=DEFAULT_MAX_NLTE,
                         processes=4, out_csv=None):
    """Parallel Stage 0 batch (the real, many-hour run). Writes each result to
    `out_csv` as soon as it completes (safe against a long run being interrupted)."""
    entries = entries if entries is not None else list(sp.iter_stage_0_subset())
    tasks = [(field, position, entry, X, max_NLTE)
             for field, position, entry in entries for X in x_sweep]

    out_csv = out_csv or os.path.join(ODIR, "stage0_stutzki_validation.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.join(ODIR, "fits"), exist_ok=True)
    os.makedirs(os.path.join(ODIR, "images"), exist_ok=True)

    print(f"Stage 0: {len(tasks)} runs ({len(entries)} benchmarks x {len(x_sweep)} "
          f"abundance splits), max_NLTE={max_NLTE}, {processes} concurrent workers")

    # Fixed superset of columns known ahead of time (not derived from whichever
    # result happens to finish first under imap_unordered -- a FAILED row
    # finishing first would otherwise truncate the header and silently drop
    # SUCCESS-only columns for every later row).
    success_fields = ['field', 'position', 'X', 'n_H2', 'N_NH3_target', 'N_dv_proxy',
                       'tau_main', 'T_k', 'radius_sphere', 'theta_arcsec',
                       'T_B_theor_stutzki', 'T_B_obs', 'eta_f_stutzki',
                       'A_10', 'A_21', 'A_MAIN', 'A_12', 'A_01',
                       'R_01_MAIN', 'R_10_MAIN', 'R_21_MAIN', 'R_12_MAIN']
    fieldnames = ['status'] + success_fields + ['error']

    n_done = 0
    with multiprocessing.Pool(processes=processes, maxtasksperchild=1) as pool:
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in pool.imap_unordered(_worker, tasks):
                writer.writerow({k: result.get(k, "") for k in fieldnames})
                f.flush()
                n_done += 1
                tag = f"{result['field']} {result['position']} X={result['X']:.0e}"
                if result['status'] == 'SUCCESS':
                    print(f"[{n_done}/{len(tasks)}] OK   {tag}: A_MAIN={result['A_MAIN']:.3f} "
                          f"tau_main={result['tau_main']:.3f} (Stutzki T_B={result['T_B_theor_stutzki']:.2f} K)")
                else:
                    print(f"[{n_done}/{len(tasks)}] FAIL {tag}: {result['error']}")
    print(f"Wrote {n_done} rows to {out_csv}")
    return out_csv


if __name__ == "__main__":
    run_subset_parallel()
