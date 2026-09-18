"""
Hyperfine-ratio inversion pipeline for constant-density NH3 spheres, built to
replicate Stutzki & Winnewisser (1985)'s own fitting procedure as closely as
possible with our Magritte forward model in place of their escape-probability
one. See /home/yasho379/.claude/plans/nested-baking-pond.md for background.

Matches his method on the points that matter:
  - 5 observables: the 4 (1,1) satellite/main ratios PLUS (2,2)-main/(1,1)-main
    (not just the 4 (1,1)-internal ones -- his own paper uses this 5th ratio
    specifically because it carries T_kin/excitation information the (1,1)
    satellites alone don't).
  - Grid axes are (T_k, n_H2, N_NH3/Delta_v) -- N_NH3/Delta_v IS a real grid
    axis fit jointly with T_k/n_H2 via chi2 against the ratios, not held at an
    arbitrary fiducial and fit separately afterward. Delta_v (clump linewidth)
    is FIXED at his 0.3 km/s value (he doesn't search it either).
  - eta_f = T_B,obs/T_B,theor is derived strictly AFTER the chi2 fit, never
    mixed into the ratio comparison itself.
  - The high-density/low-density chi2-degeneracy his own method also hits is
    resolved with HIS two physical-consistency checks (eta_f<=1, and the
    Jeans-mass clump-count K >= Delta_v_obs/Delta_v_clump -- see
    stutzki_physics.py), not arbitrary search bounds. This directly replaces
    an earlier "just clamp the search range" plan after this session's own
    verification run reproduced his exact kind of degenerate double-minimum.
  - We do NOT apply his n' -> n_H2 pseudo-density correction (n'/1.75): that
    correction exists because his rates are NH3-He scaled by alpha=1.5; we use
    Loreau et al. NH3-H2 rates directly, so our numberdensity already IS true
    n_H2.

Reuses: nh3_NLTE_sphere.run_model, nh3_NLTE_analysis.analyse_spectra,
spectrum_utils.native_peak_tmb (for the (2,2) main-line peak -- (2,2) doesn't
need a 5-Gaussian hyperfine decomposition, just its own peak brightness),
stutzki_physics.py for the degeneracy-resolution checks, and the same
Pool+incremental-CSV pattern the (now relegated) grid-builder scripts used.
"""

import os
import sys
import csv
import multiprocessing
import time
import numpy as np

import params as p
import stutzki_physics as sph
from scoring import weighted_chi2, rank_candidates

WDIR = "/home/yasho379/magritte_rebuilt/production/tgs/"
DEFAULT_ODIR = "/home/yasho379/magritte_rebuilt/scratch/output/output_invert_ratios/"
GENERAL_CATALOGUE = ("/home/yasho379/magritte_rebuilt/scratch/output/output_test_1e-6_parallel_12rays_v2/"
                      "results/NLTE_nh3_1e-6_parallel_12rays_v2.csv")
W33_CATALOGUE = ("/home/yasho379/magritte_rebuilt/scratch/output/output_w33_grid_extension/"
                  "results/NLTE_nh3_w33_extension.csv")

# The 5 Stutzki observables: 4 (1,1) satellite/main ratios + (2,2)/(1,1) main.
RATIO_KEYS = ('R_01_MAIN', 'R_10_MAIN', 'R_21_MAIN', 'R_12_MAIN', 'R_22_MAIN')
# The 4-ratio subset alone, for scoring against catalogues that don't have R_22_MAIN
# (neither existing catalogue computed a (2,2) main amplitude -- see module docstring).
RATIO_KEYS_LEGACY = ('R_01_MAIN', 'R_10_MAIN', 'R_21_MAIN', 'R_12_MAIN')

CONV_THRESHOLD = 90.0  # per user guidance: >=90% convergence is realistically usable
DEFAULT_CLUMP_DV_KMS = 0.3  # Stutzki's own fixed intrinsic clump linewidth

MAX_ROUNDS = 5
TOP_K = 8
# (n_pts_log_nH2, n_pts_T_kin, n_pts_log_Ndv) per round, coarse -> fine.
GRID_N_SCHEDULE = [(5, 5, 5), (4, 4, 3), (3, 3, 2), (3, 3, 2), (3, 3, 2)]
CONVERGENCE_TOL = 0.05
MAX_STALE_RETRIES = 2  # a round with zero new successes retries at the SAME bounds
                        # this many times before halting, instead of expanding on stale data

INVERT_HEADER = [
    'round', 'stage', 'Status', 'T_cloud', 'vturb', 'clump_dv', 'XNH3', 'numberdensity',
    'radius_sphere', 'N_NH3_target', 'A_10', 'A_21', 'A_MAIN', 'A_12', 'A_01', 'A_MAIN_22',
    'R_01_MAIN', 'R_10_MAIN', 'R_21_MAIN', 'R_12_MAIN', 'R_22_MAIN',
    'N_NH3', 'tau_main', 'halting_iter', 'final_convergence', 'convergence_ok',
    'score', 'is_survivor', 'Error_Msg',
]


# --------------------------------------------------------------------------- #
# Bounds + grid construction -- axes are (log10 n_H2, T_k, log10[N_NH3/dv])
# --------------------------------------------------------------------------- #

class Bounds:
    def __init__(self, log_n_lo, log_n_hi, T_lo, T_hi, log_Ndv_lo, log_Ndv_hi):
        self.log_n_lo, self.log_n_hi = log_n_lo, log_n_hi
        self.T_lo, self.T_hi = T_lo, T_hi
        self.log_Ndv_lo, self.log_Ndv_hi = log_Ndv_lo, log_Ndv_hi

    def copy(self):
        return Bounds(self.log_n_lo, self.log_n_hi, self.T_lo, self.T_hi,
                       self.log_Ndv_lo, self.log_Ndv_hi)

    def __repr__(self):
        return (f"Bounds(n_H2=[{10**self.log_n_lo:.2e},{10**self.log_n_hi:.2e}], "
                f"T=[{self.T_lo:.1f},{self.T_hi:.1f}], "
                f"N/dv=[{10**self.log_Ndv_lo:.2e},{10**self.log_Ndv_hi:.2e}])")


def build_grid(bounds, n_per_axis, T_kin_fixed=None):
    """Returns a list of (numberdensity_cm3, T_cloud_K, N_NH3_per_dv_cm2kms) tuples."""
    n_n, n_T, n_Ndv = n_per_axis
    log_ns = np.linspace(bounds.log_n_lo, bounds.log_n_hi, max(n_n, 1))
    Ts = [T_kin_fixed] if T_kin_fixed is not None else np.linspace(bounds.T_lo, bounds.T_hi, max(n_T, 1))
    log_Ndvs = np.linspace(bounds.log_Ndv_lo, bounds.log_Ndv_hi, max(n_Ndv, 1))
    grid = []
    for ln in log_ns:
        for T in Ts:
            for lNdv in log_Ndvs:
                grid.append((float(10 ** ln), float(T), float(10 ** lNdv)))
    return grid


def task_key(numberdensity, T_cloud, clump_dv, XNH3, radius_sphere, ndigits=6):
    return (round(np.log10(numberdensity), ndigits), round(T_cloud, ndigits),
            round(clump_dv, ndigits), round(np.log10(XNH3), ndigits),
            round(np.log10(radius_sphere), ndigits))


def radius_from_N_dv(N_NH3_per_dv, clump_dv_kms, numberdensity, XNH3):
    """N_NH3_per_dv [cm^-2/(km/s)] * clump_dv [km/s] = target peak column
    N_NH3 [cm^-2]; radius (cm, run_model's own convention) follows from the
    N0=2*r*n*X identity at the given (numberdensity, XNH3)."""
    N_NH3_target = N_NH3_per_dv * clump_dv_kms
    return N_NH3_target / (2.0 * numberdensity * XNH3), N_NH3_target


# --------------------------------------------------------------------------- #
# Boundary detection / bounds adjustment
# --------------------------------------------------------------------------- #

def is_boundary_hugging(survivors, bounds, grid_n, frac=1.0, T_kin_fixed=None):
    if not survivors:
        return False
    n_n, n_T, n_Ndv = grid_n
    step_n = (bounds.log_n_hi - bounds.log_n_lo) / max(n_n - 1, 1)
    step_T = (bounds.T_hi - bounds.T_lo) / max(n_T - 1, 1)
    step_Ndv = (bounds.log_Ndv_hi - bounds.log_Ndv_lo) / max(n_Ndv - 1, 1)
    for r in survivors:
        log_n = np.log10(float(r['numberdensity']))
        if log_n <= bounds.log_n_lo + frac * step_n or log_n >= bounds.log_n_hi - frac * step_n:
            return True
        if T_kin_fixed is None and n_T > 1:
            T = float(r['T_cloud'])
            if T <= bounds.T_lo + frac * step_T or T >= bounds.T_hi - frac * step_T:
                return True
        log_Ndv = np.log10(float(r['N_NH3_target']) / float(r['clump_dv']))
        if log_Ndv <= bounds.log_Ndv_lo + frac * step_Ndv or log_Ndv >= bounds.log_Ndv_hi - frac * step_Ndv:
            return True
    return False


def expand_bounds(bounds, survivors, grid_n, factor=2.0, T_kin_fixed=None):
    b = bounds.copy()
    n_n, n_T, n_Ndv = grid_n
    step_n = (bounds.log_n_hi - bounds.log_n_lo) / max(n_n - 1, 1)
    step_T = (bounds.T_hi - bounds.T_lo) / max(n_T - 1, 1)
    step_Ndv = (bounds.log_Ndv_hi - bounds.log_Ndv_lo) / max(n_Ndv - 1, 1)
    log_ns = [np.log10(float(r['numberdensity'])) for r in survivors]
    log_Ndvs = [np.log10(float(r['N_NH3_target']) / float(r['clump_dv'])) for r in survivors]
    span_n = bounds.log_n_hi - bounds.log_n_lo
    span_T = bounds.T_hi - bounds.T_lo
    span_Ndv = bounds.log_Ndv_hi - bounds.log_Ndv_lo

    if min(log_ns) <= bounds.log_n_lo + step_n:
        b.log_n_lo -= span_n * (factor - 1)
    if max(log_ns) >= bounds.log_n_hi - step_n:
        b.log_n_hi += span_n * (factor - 1)
    if min(log_Ndvs) <= bounds.log_Ndv_lo + step_Ndv:
        b.log_Ndv_lo -= span_Ndv * (factor - 1)
    if max(log_Ndvs) >= bounds.log_Ndv_hi - step_Ndv:
        b.log_Ndv_hi += span_Ndv * (factor - 1)
    if T_kin_fixed is None and n_T > 1:
        Ts = [float(r['T_cloud']) for r in survivors]
        if min(Ts) <= bounds.T_lo + step_T:
            b.T_lo -= span_T * (factor - 1)
        if max(Ts) >= bounds.T_hi - step_T:
            b.T_hi += span_T * (factor - 1)
        b.T_lo = max(b.T_lo, 2.8)
    return b


def shrink_bounds_around(survivors, base_bounds, shrink_factor=0.4, T_kin_fixed=None):
    log_ns = [np.log10(float(r['numberdensity'])) for r in survivors]
    log_Ndvs = [np.log10(float(r['N_NH3_target']) / float(r['clump_dv'])) for r in survivors]
    span_n = base_bounds.log_n_hi - base_bounds.log_n_lo
    span_Ndv = base_bounds.log_Ndv_hi - base_bounds.log_Ndv_lo
    pad_n = shrink_factor * span_n / 2
    pad_Ndv = shrink_factor * span_Ndv / 2

    if T_kin_fixed is not None:
        T_lo = T_hi = T_kin_fixed
    else:
        Ts = [float(r['T_cloud']) for r in survivors]
        span_T = base_bounds.T_hi - base_bounds.T_lo
        pad_T = shrink_factor * span_T / 2
        T_lo, T_hi = max(min(Ts) - pad_T, 2.8), max(Ts) + pad_T

    return Bounds(
        log_n_lo=min(log_ns) - pad_n, log_n_hi=max(log_ns) + pad_n,
        T_lo=T_lo, T_hi=T_hi,
        log_Ndv_lo=min(log_Ndvs) - pad_Ndv, log_Ndv_hi=max(log_Ndvs) + pad_Ndv,
    )


def merge_top_k(survivors, new_scored, K):
    combined = survivors + new_scored
    seen_keys, unique = set(), []
    for r in combined:
        key = task_key(float(r['numberdensity']), float(r['T_cloud']), float(r['clump_dv']),
                        float(r['XNH3']), float(r['radius_sphere']))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(r)
    unique.sort(key=lambda r: r['score'] if not np.isnan(r['score']) else np.inf)
    return unique[:K]


# --------------------------------------------------------------------------- #
# Stage 0: free catalogue lookup before any new compute
# --------------------------------------------------------------------------- #

# Both pre-existing catalogues were produced BEFORE the hyperfine labelling fix,
# so their R_01_MAIN / R_10_MAIN columns have the outer satellite pair swapped
# (see nh3_hyperfine.py). Scoring an observed vector against them would seed the
# search from a mirrored anomaly. They are therefore quarantined: seeding is OFF
# by default and must be re-enabled explicitly, which should only happen once a
# catalogue has been regenerated with corrected labels.
QUARANTINED_CATALOGUES = (GENERAL_CATALOGUE, W33_CATALOGUE)


def catalogue_lookup(obs_ratios, obs_sigmas=None, catalogue_paths=None, top_k=20,
                      allow_quarantined=False):
    """Scores against RATIO_KEYS_LEGACY (4 ratios) since neither existing
    catalogue computed R_22_MAIN -- this is a cheap SEED for Stage 1's bounds
    only, not part of the actual (5-ratio) fit.

    Returns None unless `catalogue_paths` names a catalogue built after the
    hyperfine labelling fix, or `allow_quarantined=True` is passed deliberately.
    """
    if catalogue_paths is None and not allow_quarantined:
        return None
    catalogue_paths = catalogue_paths or list(QUARANTINED_CATALOGUES)
    obs_vec = np.array([obs_ratios[k] for k in RATIO_KEYS_LEGACY])
    sigma_vec = (np.array([obs_sigmas.get(k, np.nan) for k in RATIO_KEYS_LEGACY])
                 if obs_sigmas is not None else None)

    all_rows = []
    for path in catalogue_paths:
        if not os.path.exists(path):
            continue
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                if row.get('Status') != 'SUCCESS':
                    continue
                fc = row.get('Final Convergence')
                if fc not in (None, ''):
                    try:
                        if float(fc) < CONV_THRESHOLD:
                            continue
                    except ValueError:
                        pass
                try:
                    model_vec = np.array([float(row[k]) for k in RATIO_KEYS_LEGACY])
                except (KeyError, ValueError):
                    continue
                score = weighted_chi2(model_vec, obs_vec, sigma_vec)
                if np.isnan(score):
                    continue
                row = dict(row)
                row['score'] = score
                all_rows.append(row)

    if not all_rows:
        return None
    all_rows.sort(key=lambda r: r['score'])
    top = all_rows[:top_k]

    log_ns = [np.log10(float(r['numberdensity'])) for r in top]
    Ts = [float(r['T_cloud']) for r in top]
    try:
        log_Ndvs = [np.log10(float(r['N_NH3']) / (float(r['vturb']) / 1000.0)) for r in top]
    except (KeyError, ValueError, ZeroDivisionError):
        log_Ndvs = [14.5] * len(top)  # Stutzki's own grid midpoint as a fallback

    margin_n = 0.3
    margin_T = max(3.0, (max(Ts) - min(Ts)) * 0.5 + 1.0)
    margin_Ndv = 0.5
    bounds = Bounds(
        log_n_lo=min(log_ns) - margin_n, log_n_hi=max(log_ns) + margin_n,
        T_lo=max(min(Ts) - margin_T, 2.8), T_hi=max(Ts) + margin_T,
        log_Ndv_lo=min(log_Ndvs) - margin_Ndv, log_Ndv_hi=max(log_Ndvs) + margin_Ndv,
    )
    caveat = ("catalogue seed scored on the 4 legacy ratios only (R_22_MAIN not "
              "available in either existing catalogue) -- bounds are a starting "
              "point for the real 5-ratio Stage 1 fit, not itself a 5-ratio result")
    return dict(bounds=bounds, top_rows=top, caveat=caveat)


# --------------------------------------------------------------------------- #
# Forward-model worker
# --------------------------------------------------------------------------- #

def _invert_worker(task):
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    modelling_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if modelling_dir not in sys.path:
        sys.path.insert(0, modelling_dir)
    from nh3_NLTE_sphere import run_model
    from nh3_NLTE_analysis import analyse_spectra
    from spectrum_utils import native_peak_tmb

    base = dict(round=task['round'], stage=task['stage'], T_cloud=task['T_cloud'],
                vturb=task['vturb'], clump_dv=task['clump_dv'], XNH3=task['XNH3'],
                numberdensity=task['numberdensity'], radius_sphere=task['radius_sphere'],
                N_NH3_target=task.get('N_NH3_target', ''), _key=task.get('_key'))
    try:
        conv = run_model(wdir=task['wdir'], odir=task['odir'], XNH3=task['XNH3'],
                          numberdensity=task['numberdensity'], vturb=task['vturb'],
                          T_cloud=task['T_cloud'], max_NLTE=task['max_NLTE'],
                          radius_sphere=task['radius_sphere'])
        halting_iter, final_convergence, tau_main, extra_spectra, npoints, nboundary = conv
        hfs = analyse_spectra(odir=task['odir'], XNH3=task['XNH3'],
                               numberdensity=task['numberdensity'], vturb=task['vturb'],
                               T_cloud=task['T_cloud'], radius_sphere=task['radius_sphere'],
                               max_NLTE=task['max_NLTE'], save_plots=False)

        # (2,2) main-line peak: run_model always images '22' by default, so this
        # is free (no extra Magritte compute) -- just extraction, reusing the
        # already-validated noise-floor-aware peak extractor from spectrum_utils.py.
        velos22, Is22 = extra_spectra['22']
        A_MAIN_22, noise_22, detected_22 = native_peak_tmb(velos22, Is22, p.FREQ_HZ['2,2'])
        R_22_MAIN = A_MAIN_22 / hfs['A_MAIN'] if hfs['A_MAIN'] else np.nan

        convergence_ok = bool(final_convergence >= CONV_THRESHOLD)
        return dict(Status="SUCCESS", tau_main=tau_main, halting_iter=halting_iter,
                    final_convergence=final_convergence, convergence_ok=convergence_ok,
                    A_MAIN_22=A_MAIN_22, R_22_MAIN=R_22_MAIN, **hfs, **base)
    except Exception as e:
        return dict(Status="FAILED", Error_Msg=str(e), **base)


# Magritte spins up OpenMP threads. If a caller has already run a model in the
# parent process, forking a Pool afterwards hands the children mutexes held by
# threads that do not exist in them, and every worker blocks forever in
# futex_wait -- observed as 14 workers at 0% CPU with a byte-frozen log. The
# spawn context does not inherit parent memory or lock state, so it is immune
# regardless of what the caller did first. Workers re-import, which is
# negligible against multi-minute models.
MP_CONTEXT = "spawn"

# A round with no completed task for this multiple of the slowest observed model
# is treated as hung. Silence must never be indistinguishable from progress.
WATCHDOG_STALL_FACTOR = 3.0
WATCHDOG_MIN_GRACE_S = 1800.0


def _run_new_tasks(new_tasks, processes, out_csv, header_written):
    results = []
    if not new_tasks:
        return results, header_written
    mode = 'a' if header_written else 'w'
    ctx = multiprocessing.get_context(MP_CONTEXT)
    slowest = 0.0
    last_completion = time.monotonic()
    with ctx.Pool(processes=processes, maxtasksperchild=1) as pool:
        with open(out_csv, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=INVERT_HEADER)
            if not header_written:
                writer.writeheader()
                header_written = True
            it = pool.imap_unordered(_invert_worker, new_tasks)
            while True:
                try:
                    result = next(it)
                except StopIteration:
                    break
                now = time.monotonic()
                slowest = max(slowest, now - last_completion)
                last_completion = now
                writer.writerow({k: result.get(k, "") for k in INVERT_HEADER})
                f.flush()
                results.append(result)
                stall_limit = max(WATCHDOG_MIN_GRACE_S, WATCHDOG_STALL_FACTOR * slowest)
                if now - last_completion > stall_limit:
                    raise RuntimeError(
                        f"watchdog: no task completed in {now - last_completion:.0f}s "
                        f"(limit {stall_limit:.0f}s). Workers are likely deadlocked; "
                        f"{len(results)}/{len(new_tasks)} tasks finished."
                    )
    return results, header_written


# --------------------------------------------------------------------------- #
# Stage 1: 5-ratio chi2 fit over (n_H2, T_k, N_NH3/dv), refinement loop
# --------------------------------------------------------------------------- #

def invert_ratio_vector(obs_ratios, obs_sigmas=None, T_kin_fixed=None,
                         clump_dv_kms=DEFAULT_CLUMP_DV_KMS, XNH3_fiducial=1e-8,
                         wdir=WDIR, odir=None, processes=4, max_NLTE=200,
                         out_csv=None, catalogue_paths=None,
                         max_rounds=MAX_ROUNDS, top_k=TOP_K, dry_run=False,
                         # for Stage-3-style post-fit physical-consistency check:
                         T_B_obs=None, dv_obs_kms=None, distance_pc=None):
    """5-ratio chi2 fit. If T_B_obs/dv_obs_kms/distance_pc are all given, every
    survivor is also checked against stutzki_physics.physically_consistent()
    (eta_f<=1, Jeans-mass clump-count K) and the best PHYSICALLY CONSISTENT
    survivor is reported separately from the raw best-by-score, since (as this
    session's own verification run demonstrated) the raw chi2 best can be a
    degenerate, unphysical solution.
    """
    odir = odir or DEFAULT_ODIR
    os.makedirs(os.path.join(odir, "fits"), exist_ok=True)
    os.makedirs(os.path.join(odir, "images"), exist_ok=True)
    out_csv = out_csv or os.path.join(odir, "results", "invert_ratios.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    obs_vec = np.array([obs_ratios[k] for k in RATIO_KEYS])
    sigma_vec = (np.array([obs_sigmas.get(k, np.nan) for k in RATIO_KEYS])
                 if obs_sigmas is not None else None)

    seed = catalogue_lookup(obs_ratios, obs_sigmas, catalogue_paths, top_k=20)
    if seed is not None:
        bounds = seed['bounds']
        catalogue_seed_used, catalogue_seed_caveat = True, seed['caveat']
        print(f"Stage 0: catalogue seed found -> starting {bounds}")
        print(f"  caveat: {catalogue_seed_caveat}")
    else:
        bounds = Bounds(log_n_lo=2.5, log_n_hi=6.5, T_lo=8.0, T_hi=60.0,
                          log_Ndv_lo=13.5, log_Ndv_hi=15.5)  # Stutzki's own N/dv range as a default
        catalogue_seed_used, catalogue_seed_caveat = False, None
        print("Stage 0: no catalogue seed (pre-fix catalogues are quarantined -- their "
              "outer-satellite labels are swapped) -> Stutzki's own default range "
              f"{bounds}")

    if T_kin_fixed is not None:
        bounds.T_lo = bounds.T_hi = T_kin_fixed

    seen = {}
    survivors = []
    header_written = False
    bounds_history = []
    best_score_history = []
    converged_by_score = False
    stale_retries = 0
    round_idx = 0

    while round_idx < max_rounds:
        grid_n = GRID_N_SCHEDULE[min(round_idx, len(GRID_N_SCHEDULE) - 1)]
        candidates = build_grid(bounds, grid_n, T_kin_fixed=T_kin_fixed)
        bounds_history.append(bounds.copy())

        new_tasks, cached = [], []
        for numberdensity, T_cloud, N_NH3_per_dv in candidates:
            radius_sphere, N_NH3_target = radius_from_N_dv(N_NH3_per_dv, clump_dv_kms,
                                                              numberdensity, XNH3_fiducial)
            vturb = p.fwhm_kms_to_vturb_ms(clump_dv_kms)
            key = task_key(numberdensity, T_cloud, clump_dv_kms, XNH3_fiducial, radius_sphere)
            if key in seen:
                cached.append(seen[key])
                continue
            new_tasks.append(dict(round=round_idx, stage='ratio_fit', wdir=wdir, odir=odir,
                                   max_NLTE=max_NLTE, XNH3=XNH3_fiducial, numberdensity=numberdensity,
                                   vturb=vturb, T_cloud=T_cloud, clump_dv=clump_dv_kms,
                                   radius_sphere=radius_sphere, N_NH3_target=N_NH3_target, _key=key))

        print(f"\nRound {round_idx}: {len(candidates)} candidates ({grid_n[0]}x{grid_n[1]}x{grid_n[2]}), "
              f"{len(new_tasks)} new, {len(cached)} cached, bounds={bounds}")

        if dry_run:
            round_idx += 1
            continue

        new_results, header_written = _run_new_tasks(new_tasks, processes, out_csv, header_written)
        for r in new_results:
            key = r.pop('_key', None)
            if key is not None and r.get('Status') == 'SUCCESS':
                seen[key] = r

        new_successes = [r for r in new_results if r.get('Status') == 'SUCCESS']

        if new_tasks and not new_successes:
            # Every NEW task this round failed. Do NOT expand bounds on stale
            # prior-round survivors -- that's exactly what let a 3-round-old
            # result drag the search out to n_H2~1e113 in this session's first
            # real test. Retry the SAME bounds a bounded number of times, then
            # give up cleanly.
            stale_retries += 1
            print(f"Round {round_idx}: {len(new_tasks)} new tasks, 0 successes -- "
                  f"retry {stale_retries}/{MAX_STALE_RETRIES} at same bounds.")
            if stale_retries > MAX_STALE_RETRIES:
                print("Too many stale/failed rounds -- stopping without further expansion.")
                break
            round_idx += 1
            continue
        stale_retries = 0

        round_results = [r for r in (new_successes + cached) if r.get('Status') == 'SUCCESS']
        if round_results:
            scored = rank_candidates(
                round_results,
                lambda r: weighted_chi2([float(r[k]) for k in RATIO_KEYS], obs_vec, sigma_vec),
            )
            for r in scored:
                r['is_survivor'] = False
            survivors = merge_top_k(survivors, scored, K=top_k)
            for r in survivors:
                r['is_survivor'] = True

        if not survivors:
            print("No survivors -- stopping.")
            break

        best = survivors[0]
        print(f"Round {round_idx}: best score={best['score']:.4f}  n_H2={float(best['numberdensity']):.3e}  "
              f"T={best['T_cloud']}  N/dv={float(best['N_NH3_target'])/clump_dv_kms:.3e}  "
              f"convergence_ok={best.get('convergence_ok')}")

        if is_boundary_hugging(survivors, bounds, grid_n, T_kin_fixed=T_kin_fixed):
            bounds = expand_bounds(bounds, survivors, grid_n, factor=2.0, T_kin_fixed=T_kin_fixed)
            print(f"  boundary-hugging -> expanding to {bounds}")
            round_idx += 1
            continue

        if round_idx >= 1 and best_score_history:
            prev = best_score_history[-1]
            cur = best['score']
            if prev > 0 and abs(prev - cur) / prev < CONVERGENCE_TOL:
                converged_by_score = True
                best_score_history.append(cur)
                print(f"Round {round_idx}: score improvement < {CONVERGENCE_TOL:.0%} -- stopping.")
                break
        best_score_history.append(best['score'])

        bounds = shrink_bounds_around(survivors, bounds, shrink_factor=0.4, T_kin_fixed=T_kin_fixed)
        round_idx += 1

    result = dict(
        best=survivors[0] if survivors else None,
        survivors=survivors,
        bounds_history=bounds_history,
        catalogue_seed_used=catalogue_seed_used,
        catalogue_seed_caveat=catalogue_seed_caveat,
        converged_by_score=converged_by_score,
        rounds_run=round_idx,
        out_csv=out_csv,
    )

    if survivors and T_B_obs is not None and dv_obs_kms is not None and distance_pc is not None:
        checked = []
        for r in survivors:
            ok, detail = sph.physically_consistent(
                T_k=float(r['T_cloud']), n_H2_cm3=float(r['numberdensity']),
                T_B_obs=T_B_obs, T_B_theor=float(r['A_MAIN']),
                dv_obs_kms=dv_obs_kms, dv_clump_kms=clump_dv_kms, distance_pc=distance_pc,
            )
            r = dict(r)
            r['physically_consistent'] = ok
            r.update({f'check_{k}': v for k, v in detail.items()})
            checked.append(r)
        result['survivors_checked'] = checked
        plausible = [r for r in checked if r['physically_consistent']]
        result['best_physically_consistent'] = plausible[0] if plausible else None
        if not plausible:
            print("\nWARNING: none of the top-K survivors pass eta_f<=1 / Jeans-mass "
                  "clump-count consistency -- the raw chi2 best is likely a degenerate, "
                  "unphysical solution (exactly what this session's earlier test found).")
    elif survivors:
        print("\nNo T_B_obs/dv_obs_kms/distance_pc given -- skipping the physical-consistency "
              "check. The raw chi2 best MAY be a degenerate/unphysical solution; this is not "
              "verified without eta_f/K.")

    return result


# --------------------------------------------------------------------------- #
# Stage 3: (2,1)/(3,2) out-of-sample check (unchanged in spirit from before)
# --------------------------------------------------------------------------- #

def geometry_check(fitted_params, source_name=None, wdir=WDIR, odir=None, max_NLTE=200):
    odir = odir or DEFAULT_ODIR
    os.makedirs(os.path.join(odir, "fits"), exist_ok=True)
    os.makedirs(os.path.join(odir, "images"), exist_ok=True)

    from spectrum_utils import native_peak_tmb
    from nh3_NLTE_sphere import run_model

    numberdensity = float(fitted_params['numberdensity'])
    T_cloud = float(fitted_params['T_cloud'])
    clump_dv = float(fitted_params['clump_dv'])
    vturb = p.fwhm_kms_to_vturb_ms(clump_dv)
    XNH3 = float(fitted_params['XNH3'])
    radius_sphere = float(fitted_params['radius_sphere'])

    conv = run_model(wdir=wdir, odir=odir, XNH3=XNH3, numberdensity=numberdensity, vturb=vturb,
                      T_cloud=T_cloud, max_NLTE=max_NLTE, radius_sphere=radius_sphere,
                      image_lines={'21': p.FREQ_HZ['2,1'], '32': p.FREQ_HZ['3,2']})
    tau_main, extra_spectra = conv[2], conv[3]

    freq_by_label = {'21': p.FREQ_HZ['2,1'], '32': p.FREQ_HZ['3,2']}
    report = {}
    for lbl in ('21', '32'):
        velos, Is = extra_spectra[lbl]
        peak, noise, detected = native_peak_tmb(velos, Is, freq_by_label[lbl])
        report[lbl] = dict(T_mb_predicted=peak, noise=noise, detected=detected)
        if source_name is not None:
            line_key = '2,1' if lbl == '21' else '3,2'
            obs_line = p.LINES.get(source_name, {}).get(line_key)
            if obs_line is not None:
                report[lbl]['T_mb_observed'] = obs_line['T_mb']
                report[lbl]['tension'] = ("model undetected, real detection exists"
                                           if (not detected and obs_line['T_mb'] > 0.1) else None)
    return report


if __name__ == "__main__":
    # Smoke test: dry-run only, no Magritte compute.
    obs = dict(R_10_MAIN=0.3387, R_21_MAIN=0.4856, R_12_MAIN=0.4522, R_01_MAIN=0.4122, R_22_MAIN=0.6988)
    result = invert_ratio_vector(obs, dry_run=True, max_rounds=1)
    print("\nDry run complete. bounds_history:", result['bounds_history'])
