"""Paper I's central result: the NH3 (2,1)/(1,1) test against Stutzki & Winnewisser (1985).

THE QUESTION
------------
Stutzki's escape-probability model, using Green (1981) NH3-He rates scaled by
alpha=1.5 with statistical redistribution among hyperfine sub-levels, predicted
(2,1)/(1,1) brightness ratios that his own observations contradicted (his
Table 3):

    position      predicted   observed          discrepancy
    S106 (0,0)      0.018      0.014 +/- 0.003    agrees
    OMC S3          0.052      0.015 +/- 0.002    3.5x too high
    OMC S4          0.068      0.028 +/- 0.002    2.4x too high

He attributed the S3/S4 failure to his fitted densities (>1e7 cm^-3) being
overestimated by roughly an order of magnitude (his Sect. 5).

Does full 3D NLTE with hyperfine-resolved NH3-H2 rates (Loreau et al. 2023)
resolve that discrepancy, or reproduce it? Resolving it means the 1985 failure
was an artefact of the rates and the escape-probability approximation.
Reproducing it independently confirms his density-overestimate explanation with
better physics. Either outcome is a result.

WHY THIS NEEDS NO ABSOLUTE CALIBRATION
--------------------------------------
The (2,1) and (1,1) lines arise in the same gas, so the beam filling factor
cancels in their ratio. That makes this test independent of the unresolved
absolute-normalisation question (the imager's true field of view), which is why
it can run before that is settled.

Models are built at Stutzki's own published Table 1a parameters -- we are
testing his parameters through our radiative transfer, not refitting.
"""

import os
import sys
import json
import time
import argparse
import multiprocessing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_MODELLING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODELLING not in sys.path:
    sys.path.insert(0, _MODELLING)

import numpy as np
import params as p
import stutzki_params as sp

WDIR = "/home/yasho379/magritte_rebuilt/tgs/"
ODIR = "/home/yasho379/magritte_rebuilt/output_stutzki_21_test/"

XNH3_FIDUCIAL = 1e-8
NRAYS = 48
RESOLUTION = 14
MAX_NLTE = 250
FOV_PAD = 1.15


def _worker(task):
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    if _MODELLING not in sys.path:
        sys.path.insert(0, _MODELLING)
    from nh3_NLTE_sphere import run_model
    from nh3_NLTE_analysis import analyse_spectra
    from spectrum_utils import native_peak_tmb

    field, position = task['field'], task['position']
    e = sp.fit(field, position)
    T_k = e['T_k']
    n_H2 = sp.n_H2_cm3(e)
    N_NH3 = sp.N_NH3_cm2(e)
    # Same convention as validate_stutzki: N0 = 2 r n X.
    radius_sphere = N_NH3 / (2.0 * n_H2 * XNH3_FIDUCIAL)
    vturb = p.fwhm_kms_to_vturb_ms(sp.CLUMP_DV_KMS)

    t0 = time.time()
    try:
        hi, conv, tau, extra, npoints, nboundary = run_model(
            wdir=WDIR, odir=ODIR, XNH3=XNH3_FIDUCIAL, numberdensity=n_H2,
            vturb=vturb, T_cloud=T_k, max_NLTE=MAX_NLTE,
            radius_sphere=radius_sphere,
            image_lines={'21': p.FREQ_HZ['2,1']},
            nrays=NRAYS, resolution=RESOLUTION, fov_pad_factor=FOV_PAD,
            spectrum=task['spectrum'])

        hfs = analyse_spectra(odir=ODIR, XNH3=XNH3_FIDUCIAL, numberdensity=n_H2,
                               vturb=vturb, T_cloud=T_k, radius_sphere=radius_sphere,
                               max_NLTE=MAX_NLTE, save_plots=False)

        out = dict(field=field, position=position, spectrum=task['spectrum'],
                   T_k=T_k, n_H2=n_H2, N_NH3=N_NH3, tau_main=float(tau),
                   final_convergence=float(conv), halting_iter=int(hi),
                   A_MAIN=float(hfs['A_MAIN']), wall_s=round(time.time() - t0, 1),
                   Status='SUCCESS')

        # (2,1) peak from whichever spectrum variant this task selected.
        for variant in ('center', 'integrated'):
            v21, I21 = extra[f'21_{variant}']
            pk21, noise21, det21 = native_peak_tmb(v21, I21, p.FREQ_HZ['2,1'])
            v11, I11 = extra[f'11_{variant}']
            pk11, _, _ = native_peak_tmb(v11, I11, p.FREQ_HZ['1,1'])
            out[f'T_B_21_{variant}'] = float(pk21)
            out[f'T_B_11_{variant}'] = float(pk11)
            out[f'noise_21_{variant}'] = float(noise21)
            out[f'detected_21_{variant}'] = bool(det21)
            out[f'ratio_{variant}'] = float(pk21 / pk11) if pk11 else np.nan
        return out
    except Exception as ex:
        return dict(field=field, position=position, Status='FAILED',
                    Error=str(ex)[:400], wall_s=round(time.time() - t0, 1))


def main(processes=3, spectrum='integrated'):
    for sub in ('fits', 'images', 'results'):
        os.makedirs(os.path.join(ODIR, sub), exist_ok=True)

    tasks = [dict(field=f, position=pos, spectrum=spectrum)
             for (f, pos) in sp.TABLE_3_21.keys()]
    print(f"Stutzki (2,1) test: {len(tasks)} positions, "
          f"nrays={NRAYS} resolution={RESOLUTION} max_NLTE={MAX_NLTE}\n")
    for t in tasks:
        e = sp.fit(t['field'], t['position'])
        print(f"  {t['field']} {t['position']:<6} T_k={e['T_k']:5.2f} K  "
              f"log n={e['log_nH2']:.3f}  log N={e['log_N_NH3']:.2f}")
    print()

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=processes, maxtasksperchild=1) as pool:
        results = list(pool.imap_unordered(_worker, tasks))

    print("\n" + "=" * 78)
    print("RESULT: modelled vs Stutzki predicted vs observed  T_B(2,1)/T_B(1,1)")
    print("=" * 78)
    hdr = f"{'position':<12}{'model':>9}{'his pred':>10}{'observed':>16}{'m/obs':>8}{'his/obs':>9}"
    print(hdr); print("-" * 78)
    rows = []
    for r in sorted(results, key=lambda r: (r['field'], r['position'])):
        key = (r['field'], r['position'])
        t3 = sp.TABLE_3_21[key]
        name = f"{r['field']} {r['position']}"
        if r.get('Status') != 'SUCCESS':
            print(f"{name:<12}  FAILED: {r.get('Error','')[:50]}")
            continue
        m = r[f'ratio_{spectrum}']
        obs, oerr, th = t3['ratio_obs'], t3['ratio_obs_err'], t3['ratio_theor']
        print(f"{name:<12}{m:9.4f}{th:10.4f}{obs:11.4f}+/-{oerr:.3f}"
              f"{m/obs:8.2f}{th/obs:9.2f}")
        r.update(stutzki_ratio_theor=th, stutzki_ratio_obs=obs,
                 stutzki_ratio_obs_err=oerr)
        rows.append(r)
    print("-" * 78)
    print("m/obs = model / observed.  his/obs = Stutzki's own prediction / observed.")
    print("His model overpredicted OMC S3 by 3.5x and S4 by 2.4x; the question is")
    print("whether 3D NLTE with NH3-H2 rates moves the model toward the observations.")

    print(f"\n{'position':<12}{'tau_main':>10}{'conv %':>9}{'A_MAIN K':>10}"
          f"{'(2,1) det?':>12}{'wall s':>9}")
    for r in rows:
        print(f"{r['field']+' '+r['position']:<12}{r['tau_main']:10.4f}"
              f"{r['final_convergence']:9.1f}{r['A_MAIN']:10.3f}"
              f"{str(r[f'detected_21_{spectrum}']):>12}{r['wall_s']:9.0f}")

    path = os.path.join(ODIR, 'results', 'stutzki_21_test.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWROTE {path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--processes', type=int, default=3)
    ap.add_argument('--spectrum', default='integrated', choices=['center', 'integrated'])
    a = ap.parse_args()
    main(processes=a.processes, spectrum=a.spectrum)
