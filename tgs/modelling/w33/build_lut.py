"""Build the precomputed lookup table of NH3 hyperfine spectra.

The table is NOISELESS by construction -- `noise_rms_K` is never passed. Noise
is a property of an observation, not of the model, so it is applied to the
observed side at inference time (see noise.py). Building the table first also
makes the noisy recovery test cheaper rather than riskier: once the table
exists that test is interpolation, not Magritte.

Moves Magritte's cost out of the inversion loop and into one bounded, resumable
run. After this exists an inversion is microseconds, chi^2 landscapes are free,
and MCMC becomes possible at all -- it needs 1e5-1e6 forward evaluations, which
is impossible with Magritte in the loop.

GRID
----
Axes are the three the observables actually depend on. X_NH3 and radius_sphere
are exactly degenerate (they enter only through N = 2 r n X), so they are one
axis, not two.

    log10 n_H2        3.5 -> 7.5   step 0.25 dex   17 points
    T_k [K]            12 -> 48    step 3 K        13 points
    log10 (N_NH3/dv)  14.0 -> 15.8 step 0.20 dex   10 points

2210 models at the Delta_v = 0.3 km/s reference rung. These are essentially
Stutzki & Winnewisser (1985)'s own tested extents plus a margin, and contain
every Table 1a benchmark including OMC S1 at log n' = 7.576. The 3 K temperature
step makes the grid land exactly on 18, 24, 30 and 36 K -- his Fig. 4 panel
temperatures -- so that figure is directly reproducible.

WHAT IS STORED, AND WHY SPECTRA
-------------------------------
The 1-D spectra are kept, not just the derived ratios. Storing only ratios would
freeze the table against the current observable set: no changing the peak
extractor, no moving to full-profile fitting, no adding a line, without
recomputing everything. At 500 channels x 3 lines x float32 that is ~6 KB per
model, so the whole table is tens of MB -- and the run asserts against a hard
cap before it can grow unnoticed. FITS cubes are deliberately not written; at
this grid size they would be hundreds of GB and nothing reads them back.

PINNED NUMERICS
---------------
nrays=48, resolution=10, max_NLTE=250, passed explicitly rather than left at a
signature default, and recorded in every row so the choice is auditable and a
partial rebuild stays possible if the deferred higher-resolution comparison
fails. max_NLTE is a time cap, not a convergence claim: rows that hit it are
flagged rather than silently trusted.
"""

import os
import sys
import csv
import json
import time
import argparse
import multiprocessing
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_MODELLING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODELLING not in sys.path:
    sys.path.insert(0, _MODELLING)

import params as p

WDIR = "/home/yasho379/magritte_rebuilt/tgs/"
DEFAULT_ODIR = "/home/yasho379/magritte_rebuilt/output_lut/"

# --- grid ------------------------------------------------------------------
LOG_N_LO, LOG_N_HI, LOG_N_STEP = 3.5, 7.5, 0.25
T_LO, T_HI, T_STEP = 12.0, 48.0, 3.0
LOG_NDV_LO, LOG_NDV_HI, LOG_NDV_STEP = 14.0, 15.8, 0.20
REFERENCE_DV_KMS = 0.3

# --- pinned numerics -------------------------------------------------------
NRAYS = 48
# Reverted from 14 back to 10: at resolution=14, the exact point
# (log n=7.3, T=27.8, log_Ndv=14.68) that converges cleanly in 219s at
# resolution=10 instead froze at fraction_not_converged = 35.7488% across
# multiple real iterations (confirmed via Magritte's own convergence metric,
# not a display artifact) in two independent runs on an otherwise-idle
# machine -- ruling out CPU contention. A third attempt with the model file
# deliberately deleted first (to rule out a stale/partially-written .hdf5 from
# an earlier killed process) was in progress when this was reverted; that
# question is open, not resolved, and resolution=14 should not be reused
# without either that diagnosis landing clean or a fresh investigation.
# resolution=10 is the settled-safe choice: verified converging correctly on
# three real points earlier this session (219-680s), giving 53 interior
# points per R7's measurement (vs 192 at resolution=14) -- less radial
# sampling of beta(r), accepted for now to unblock the LUT build.
RESOLUTION = 10
MAX_NLTE = 250
XNH3_FIDUCIAL = 1e-8
SPECTRUM = 'integrated'      # disc-averaged: what an unresolved source gives
FOV_PAD_FACTOR = 1.15        # keep the limb off the image edge
LINES = ('11', '22', '21')

CONV_THRESHOLD = 90.0
STORAGE_CAP_MB = 500.0       # hard ceiling; run refuses to start if projected above

HEADER = [
    'log_n_H2', 'T_cloud', 'log_N_dv', 'clump_dv', 'XNH3', 'numberdensity',
    'radius_sphere', 'N_NH3_target', 'vturb',
    'A_10', 'A_12', 'A_MAIN', 'A_21', 'A_01',
    'R_01_MAIN', 'R_10_MAIN', 'R_21_MAIN', 'R_12_MAIN',
    'I_10', 'I_12', 'I_MAIN', 'I_21', 'I_01',
    'RI_01_MAIN', 'RI_10_MAIN', 'RI_21_MAIN', 'RI_12_MAIN',
    'HIA_IS', 'HIA_OS', 'CEN_MAIN', 'SIG_MAIN', 'FWHM_MAIN',
    'A_MAIN_22', 'R_22_MAIN', 'A_MAIN_21', 'R_21_11', 'detected_21',
    'tau_main', 'halting_iter', 'final_convergence', 'convergence_ok',
    'nrays', 'resolution', 'max_NLTE', 'npoints', 'nboundary',
    'spectrum', 'fov_pad_factor',
    'wall_s', 'Status', 'Error_Msg',
]


def grid_axes():
    log_n = np.round(np.arange(LOG_N_LO, LOG_N_HI + 1e-9, LOG_N_STEP), 6)
    T = np.round(np.arange(T_LO, T_HI + 1e-9, T_STEP), 6)
    log_ndv = np.round(np.arange(LOG_NDV_LO, LOG_NDV_HI + 1e-9, LOG_NDV_STEP), 6)
    return log_n, T, log_ndv


# Stutzki & Winnewisser (1985) Fig. 4 panel temperatures. Verified to land
# exactly on the T axis, and his Fig. 4 axis ranges (log n 3.5-6.5,
# log N/dv 14.2-15.6) sit inside ours, so these rows reproduce his figure.
PAPER1_TEMPERATURES = (18.0, 24.0, 30.0, 36.0)


def grid_points(paper1_first=True):
    """Grid points, by default ordered so Paper I's Fig. 4 slice completes first.

    That slice is 17 x 4 x 10 = 680 models, 31% of the table, so ordering it to
    the front makes the figure available roughly a third of the way through the
    build instead of at the end. The remaining temperatures fill in behind it
    and block nothing. Resumption is by grid key, so the ordering is free.
    """
    log_n, T, log_ndv = grid_axes()
    pts = [(float(a), float(b), float(c)) for a in log_n for b in T for c in log_ndv]
    if not paper1_first:
        return pts
    first = [q for q in pts if any(abs(q[1] - t) < 1e-9 for t in PAPER1_TEMPERATURES)]
    rest = [q for q in pts if not any(abs(q[1] - t) < 1e-9 for t in PAPER1_TEMPERATURES)]
    return first + rest


def radius_from_N_dv(N_NH3_per_dv, dv_kms, numberdensity, XNH3):
    """N/dv * dv = target peak column; radius follows from N0 = 2 r n X."""
    N = N_NH3_per_dv * dv_kms
    return N / (2.0 * numberdensity * XNH3), N


def point_key(log_n, T, log_ndv):
    return f"{log_n:.4f}_{T:.4f}_{log_ndv:.4f}"


def _worker(task):
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    if _MODELLING not in sys.path:
        sys.path.insert(0, _MODELLING)

    from nh3_NLTE_sphere import run_model
    from nh3_NLTE_analysis import analyse_spectra
    from spectrum_utils import native_peak_tmb
    import nh3_hyperfine as hf

    t0 = time.time()
    base = dict(log_n_H2=task['log_n'], T_cloud=task['T_cloud'], log_N_dv=task['log_ndv'],
                clump_dv=task['dv'], XNH3=task['XNH3'], numberdensity=task['numberdensity'],
                radius_sphere=task['radius_sphere'], N_NH3_target=task['N_NH3_target'],
                vturb=task['vturb'], nrays=NRAYS, resolution=RESOLUTION,
                max_NLTE=MAX_NLTE, spectrum=SPECTRUM,
                fov_pad_factor=FOV_PAD_FACTOR, _key=task['key'])
    try:
        hi, conv, tau, extra = run_model(
            wdir=task['wdir'], odir=task['odir'], XNH3=task['XNH3'],
            numberdensity=task['numberdensity'], vturb=task['vturb'],
            T_cloud=task['T_cloud'], max_NLTE=MAX_NLTE,
            radius_sphere=task['radius_sphere'],
            image_lines={'21': p.FREQ_HZ['2,1']},
            nrays=NRAYS, resolution=RESOLUTION, spectrum=SPECTRUM,
            fov_pad_factor=FOV_PAD_FACTOR)

        hfs = analyse_spectra(odir=task['odir'], XNH3=task['XNH3'],
                               numberdensity=task['numberdensity'], vturb=task['vturb'],
                               T_cloud=task['T_cloud'], radius_sphere=task['radius_sphere'],
                               max_NLTE=MAX_NLTE, save_plots=False)

        # (2,2) and (2,1) main-line peaks -- both already imaged, so free.
        v22, I22 = extra['22']
        A22, _, _ = native_peak_tmb(v22, I22, p.FREQ_HZ['2,2'])
        v21, I21 = extra['21']
        A21, _, det21 = native_peak_tmb(v21, I21, p.FREQ_HZ['2,1'])

        spectra = {}
        for lbl in LINES:
            vv, ii = extra[f'{lbl}_{SPECTRUM}']
            spectra[lbl] = np.asarray(ii, dtype=np.float32)
        spectra['velos'] = np.asarray(extra['11_' + SPECTRUM][0], dtype=np.float32)

        row = dict(base)
        row.update({k: hfs[k] for k in hfs if k != 'N_NH3'})
        row.update(
            A_MAIN_22=float(A22),
            R_22_MAIN=float(A22 / hfs['A_MAIN']) if hfs['A_MAIN'] else np.nan,
            A_MAIN_21=float(A21),
            R_21_11=float(A21 / hfs['A_MAIN']) if hfs['A_MAIN'] else np.nan,
            detected_21=bool(det21),
            tau_main=float(tau), halting_iter=int(hi), final_convergence=float(conv),
            convergence_ok=bool(conv >= CONV_THRESHOLD),
            npoints=int(task.get('npoints', -1)), nboundary=int(task.get('nboundary', -1)),
            wall_s=round(time.time() - t0, 1), Status='SUCCESS')
        return row, spectra
    except Exception as e:
        base.update(Status='FAILED', Error_Msg=str(e)[:400],
                    wall_s=round(time.time() - t0, 1))
        return base, None


def build(odir=DEFAULT_ODIR, processes=14, dv_kms=REFERENCE_DV_KMS, limit=None,
          dry_run=False):
    os.makedirs(os.path.join(odir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(odir, 'fits'), exist_ok=True)
    os.makedirs(os.path.join(odir, 'images'), exist_ok=True)
    out_csv = os.path.join(odir, 'results', f'lut_dv{dv_kms:.2f}.csv')
    spec_dir = os.path.join(odir, 'spectra')
    os.makedirs(spec_dir, exist_ok=True)

    pts = grid_points()
    if limit:
        pts = pts[:limit]

    # Resume: skip anything already written.
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv, newline='') as f:
            for r in csv.DictReader(f):
                done.add(point_key(float(r['log_n_H2']), float(r['T_cloud']),
                                    float(r['log_N_dv'])))

    tasks = []
    for log_n, T, log_ndv in pts:
        key = point_key(log_n, T, log_ndv)
        if key in done:
            continue
        n = 10.0 ** log_n
        radius_sphere, N = radius_from_N_dv(10.0 ** log_ndv, dv_kms, n, XNH3_FIDUCIAL)
        tasks.append(dict(log_n=log_n, T_cloud=T, log_ndv=log_ndv, dv=dv_kms,
                          XNH3=XNH3_FIDUCIAL, numberdensity=n,
                          radius_sphere=radius_sphere, N_NH3_target=N,
                          vturb=p.fwhm_kms_to_vturb_ms(dv_kms),
                          wdir=WDIR, odir=odir, key=key))

    # Storage guard: 500 channels x 3 lines x float32, plus the shared axis.
    per_model_kb = (500 * len(LINES) * 4) / 1024.0
    projected_mb = per_model_kb * len(pts) / 1024.0
    print(f"grid: {len(pts)} points ({len(tasks)} to run, {len(done)} already done)")
    print(f"storage: ~{per_model_kb:.1f} KB/model -> ~{projected_mb:.1f} MB total "
          f"(cap {STORAGE_CAP_MB:.0f} MB)")
    if projected_mb > STORAGE_CAP_MB:
        raise RuntimeError(f"projected storage {projected_mb:.0f} MB exceeds cap")
    if dry_run:
        print("dry run -- nothing executed")
        return

    header_written = os.path.exists(out_csv)
    ctx = multiprocessing.get_context("spawn")
    t_start = time.time()
    n_ok = n_fail = 0
    with ctx.Pool(processes=processes, maxtasksperchild=1) as pool:
        with open(out_csv, 'a' if header_written else 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=HEADER, extrasaction='ignore')
            if not header_written:
                w.writeheader()
            for row, spectra in pool.imap_unordered(_worker, tasks):
                key = row.pop('_key', None)
                w.writerow({k: row.get(k, '') for k in HEADER})
                f.flush()
                if spectra is not None:
                    np.savez_compressed(os.path.join(spec_dir, f'{key}.npz'), **spectra)
                    n_ok += 1
                else:
                    n_fail += 1
                done_n = n_ok + n_fail
                if done_n % 10 == 0 or done_n == len(tasks):
                    el = time.time() - t_start
                    rate = el / max(done_n, 1)
                    print(f"  {done_n}/{len(tasks)}  ok={n_ok} fail={n_fail}  "
                          f"{el/60:.1f} min elapsed, ~{rate*(len(tasks)-done_n)/60:.0f} min left",
                          flush=True)
    print(f"\ndone: {n_ok} ok, {n_fail} failed in {(time.time()-t_start)/60:.1f} min")
    print(f"csv: {out_csv}\nspectra: {spec_dir}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--processes', type=int, default=14)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--dv', type=float, default=REFERENCE_DV_KMS)
    ap.add_argument('--odir', default=DEFAULT_ODIR)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    build(odir=a.odir, processes=a.processes, dv_kms=a.dv, limit=a.limit,
          dry_run=a.dry_run)
