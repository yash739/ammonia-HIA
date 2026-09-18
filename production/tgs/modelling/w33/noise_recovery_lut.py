"""Noise-injection recovery test against the real (coarse) LUT.

Phase 1 of the plan, updated for the fact that the LUT and its noise
machinery both now exist (Phase 1 was spec'd before either did). This is
the fitness-for-purpose certification the Eq.(11) comparison never was:
can this pipeline recover a KNOWN (n_H2, T_k, N_NH3/dv) from a noisy
observed spectrum, and down to what S/N?

Needs ZERO new Magritte compute: truth spectra are the LUT's own stored
disc-integrated (1,1)/(2,2) spectra (output_lut_coarse/spectra/*.npz).
Noise is injected with the already-existing tgs/modelling/noise.py
machinery; retrieval is a plain weighted_chi2 argmin over the LUT's own
converged rows (RATIO_KEYS, scoring.weighted_chi2) -- no new grid search,
since we are not launching new models, just re-scoring what already exists.

Parallelized across processes -- pure CPU-bound Python (curve_fit +
numpy chi2 scoring), no Magritte/OpenMP involved, so plain
multiprocessing with a spawn context (this project's standing convention
to avoid fork-after-threads deadlocks) is safe and simple.

Usage: python3 noise_recovery_lut.py [--processes N]
"""
import os
import sys
import argparse
import multiprocessing

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

LUT_CSV = "/home/yasho379/magritte_rebuilt/production/output_lut_coarse/results/lut_dv0.30.csv"
SPEC_DIR = "/home/yasho379/magritte_rebuilt/production/output_lut_coarse/spectra"

RATIO_KEYS = ('R_01_MAIN', 'R_10_MAIN', 'R_21_MAIN', 'R_12_MAIN', 'R_22_MAIN')

TRUTH_POINTS = [
    ('low density', 4.0, 24.0, 14.9),
    ('mid density (worst interpolation region)', 5.5, 24.0, 14.9),
    ('high density', 7.0, 24.0, 14.9),
]

SNR_LADDER = [100, 50, 20, 10, 5]
M_REALIZATIONS = 200
PASS_TOL_DEX = 0.5  # LUT's own log_n axis spacing

# Per-worker globals, set once by _worker_init -- avoids re-pickling the
# LUT dataframe and re-importing heavy modules (nh3_hyperfine, astropy) on
# every one of the ~3000 tasks.
_G = {}


def point_key(log_n, T, log_ndv):
    return f"{log_n:.4f}_{T:.4f}_{log_ndv:.4f}"


def _worker_init():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import params as p
    from scoring import weighted_chi2
    from noise import add_channel_noise, rms_for_target_snr, line_seed
    from nh3_NLTE_analysis import fit_five_gaussians, subtract_baseline, intensity_to_Tmb
    from spectrum_utils import native_peak_tmb
    from stutzki_physics import physically_consistent

    df = pd.read_csv(LUT_CSV)
    df = df[df['Status'] == 'SUCCESS']
    df = df[df['convergence_ok']]
    df = df.reset_index(drop=True)

    spectra = {}
    for label, log_n, T, log_ndv in TRUTH_POINTS:
        key = point_key(log_n, T, log_ndv)
        npz_path = os.path.join(SPEC_DIR, f"{key}.npz")
        if os.path.exists(npz_path):
            d = np.load(npz_path)
            spectra[key] = dict(v11=d['velos'].astype(float), I11=d['11'].astype(float),
                                 v22=d['velos'].astype(float), I22=d['22'].astype(float))

    _G.update(dict(p=p, weighted_chi2=weighted_chi2, add_channel_noise=add_channel_noise,
                    rms_for_target_snr=rms_for_target_snr, line_seed=line_seed,
                    fit_five_gaussians=fit_five_gaussians, subtract_baseline=subtract_baseline,
                    intensity_to_Tmb=intensity_to_Tmb, native_peak_tmb=native_peak_tmb,
                    physically_consistent=physically_consistent,
                    lut_df=df, model_mat=df[list(RATIO_KEYS)].values, spectra=spectra))


def _fit_ratios(v11, I11, v22, I22, rms_K, seed):
    g = _G
    seed11 = g['line_seed'](seed, '11')
    seed22 = g['line_seed'](seed, '22')
    I11n = g['add_channel_noise'](I11, rms_K, g['p'].FREQ_HZ['1,1'], seed=seed11)
    I22n = g['add_channel_noise'](I22, rms_K, g['p'].FREQ_HZ['2,2'], seed=seed22)

    tmb11 = g['intensity_to_Tmb'](v11, I11n, g['p'].FREQ_HZ['1,1'])
    tmb11_bs = g['subtract_baseline'](v11, tmb11)
    try:
        pars = g['fit_five_gaussians'](v11, tmb11_bs, 'one')
    except Exception:
        return None
    A10, A21, AMAIN, A12, A01 = pars[0:15:3]
    if AMAIN <= 0:
        return None

    A22, _, _ = g['native_peak_tmb'](v22, I22n, g['p'].FREQ_HZ['2,2'])

    ratios = {
        'R_01_MAIN': A01 / AMAIN, 'R_10_MAIN': A10 / AMAIN,
        'R_21_MAIN': A21 / AMAIN, 'R_12_MAIN': A12 / AMAIN,
        'R_22_MAIN': A22 / AMAIN,
    }
    return ratios, AMAIN


# Assumed for the branch-rejection filter -- this synthetic test's "truth"
# is a single clump, not a real beam-averaged ensemble observation, so
# there is no true dv_obs to measure. W33's own real linewidths (Tursun+
# 2022, L5) span 2.2-3.4 km/s; 2.5 km/s is used here as an illustrative,
# clearly-assumed representative value, not a derived one.
ASSUMED_DV_OBS_KMS = 2.5
DV_CLUMP_KMS = 0.3       # matches build_lut.REFERENCE_DV_KMS
DISTANCE_PC = 2400.0     # params.DISTANCE_PC, W33 (Immer et al. 2013)
N_TOP_CANDIDATES = 30    # how many best-chi2 rows to branch-check


def _worker_task(task):
    """task = (label, log_n_true, T_true, log_ndv_true, snr, rms_K, m).

    Returns (label, snr, raw_log_n, raw_T, filtered_log_n, filtered_T,
    filter_fell_back) -- the raw chi2-best AND the best candidate (among
    the top N_TOP_CANDIDATES by chi2) that also passes Stutzki's own
    eta_f<=1 / Jeans-clump-count physical-consistency filter, so the two
    can be compared directly on the same noisy realization.
    """
    label, log_n_true, T_true, log_ndv_true, snr, rms_K, m = task
    g = _G
    key = point_key(log_n_true, T_true, log_ndv_true)
    spec = g['spectra'].get(key)
    if spec is None:
        return (label, snr, None, None, None, None, None)

    seed = hash((key, snr, m)) & 0xFFFFFFFF
    fit = _fit_ratios(spec['v11'], spec['I11'], spec['v22'], spec['I22'], rms_K, seed)
    if fit is None:
        return (label, snr, None, None, None, None, None)
    obs, T_B_obs = fit

    obs_vec = np.array([obs[k] for k in RATIO_KEYS])
    scores = np.array([g['weighted_chi2'](row, obs_vec) for row in g['model_mat']])
    if np.all(np.isnan(scores)):
        return (label, snr, None, None, None, None, None)

    order = np.argsort(np.where(np.isnan(scores), np.inf, scores))
    best_idx = order[0]
    best = g['lut_df'].iloc[best_idx]
    raw_log_n, raw_T = float(best['log_n_H2']), float(best['T_cloud'])

    filtered_log_n, filtered_T, fell_back = None, None, True
    for idx in order[:N_TOP_CANDIDATES]:
        if not np.isfinite(scores[idx]):
            continue
        cand = g['lut_df'].iloc[idx]
        T_B_theor = float(cand['A_MAIN'])
        ok, detail = g['physically_consistent'](
            T_k=float(cand['T_cloud']), n_H2_cm3=10.0 ** float(cand['log_n_H2']),
            T_B_obs=T_B_obs, T_B_theor=T_B_theor,
            dv_obs_kms=ASSUMED_DV_OBS_KMS, dv_clump_kms=DV_CLUMP_KMS,
            distance_pc=DISTANCE_PC)
        if ok:
            filtered_log_n, filtered_T = float(cand['log_n_H2']), float(cand['T_cloud'])
            fell_back = False
            break
    if filtered_log_n is None:
        # No candidate in the top N passed both checks -- fall back to the
        # raw chi2-best, flagged, rather than silently returning nothing.
        filtered_log_n, filtered_T = raw_log_n, raw_T

    return (label, snr, raw_log_n, raw_T, filtered_log_n, filtered_T, fell_back)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--processes', type=int, default=12)
    a = ap.parse_args()

    # Need peak T_mb per truth point to build the S/N ladder -- compute once
    # in the main process (cheap, no fitting needed).
    import params as p
    from nh3_NLTE_analysis import intensity_to_Tmb
    from noise import rms_for_target_snr

    tasks = []
    peak_tmb = {}
    for label, log_n, T, log_ndv in TRUTH_POINTS:
        key = point_key(log_n, T, log_ndv)
        npz_path = os.path.join(SPEC_DIR, f"{key}.npz")
        if not os.path.exists(npz_path):
            print(f"SKIP {label}: no spectrum at {npz_path}")
            continue
        d = np.load(npz_path)
        v11, I11 = d['velos'].astype(float), d['11'].astype(float)
        tmb11_clean = intensity_to_Tmb(v11, I11, p.FREQ_HZ['1,1'])
        peak_tmb[label] = float(np.max(tmb11_clean))
        for snr in SNR_LADDER:
            rms_K = rms_for_target_snr(peak_tmb[label], snr)
            for m in range(M_REALIZATIONS):
                tasks.append((label, log_n, T, log_ndv, snr, rms_K, m))

    print(f"{len(tasks)} total (truth_point x S/N x realization) tasks, "
          f"{a.processes} workers", flush=True)

    ctx = multiprocessing.get_context("spawn")
    all_results = []
    with ctx.Pool(processes=a.processes, initializer=_worker_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker_task, tasks, chunksize=8)):
            all_results.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tasks)} done", flush=True)

    # Aggregate
    from collections import defaultdict
    by_key = defaultdict(list)
    for label, snr, raw_log_n, raw_T, filt_log_n, filt_T, fell_back in all_results:
        by_key[(label, snr)].append((raw_log_n, raw_T, filt_log_n, filt_T, fell_back))

    def _stats(vals, true_val):
        arr = np.array(vals)
        bias = np.median(arr) - true_val
        scatter = np.std(arr)
        frac_within_tol = np.mean(np.abs(arr - true_val) <= PASS_TOL_DEX)
        return bias, scatter, frac_within_tol

    rows_out = []
    for label, log_n_true, T_true, log_ndv_true in TRUTH_POINTS:
        if label not in peak_tmb:
            continue
        print(f"\n=== {label}: log_n={log_n_true} T={T_true} log_ndv={log_ndv_true} "
              f"(peak T_mb={peak_tmb[label]:.3f} K) ===")
        for snr in SNR_LADDER:
            entries = by_key.get((label, snr), [])
            recovered = [e for e in entries if e[0] is not None]
            n_fail = len(entries) - len(recovered)
            if not recovered:
                print(f"  S/N={snr:4d}: all {len(entries)} realizations failed to fit/retrieve")
                continue

            raw_ln = [e[0] for e in recovered]
            filt_ln = [e[2] for e in recovered]
            n_fellback = sum(1 for e in recovered if e[4])

            raw_bias, raw_scatter, raw_frac = _stats(raw_ln, log_n_true)
            filt_bias, filt_scatter, filt_frac = _stats(filt_ln, log_n_true)
            raw_pass = raw_frac >= 0.68
            filt_pass = filt_frac >= 0.68
            print(f"  S/N={snr:4d}  RAW   : bias={raw_bias:+.3f} dex scatter={raw_scatter:.3f} "
                  f"within {PASS_TOL_DEX}dex={raw_frac:.1%}  {'PASS' if raw_pass else 'FAIL'}")
            print(f"           FILTERED: bias={filt_bias:+.3f} dex scatter={filt_scatter:.3f} "
                  f"within {PASS_TOL_DEX}dex={filt_frac:.1%}  {'PASS' if filt_pass else 'FAIL'}"
                  f"  (fell back to raw: {n_fellback}/{len(recovered)}, fails={n_fail}/{len(entries)})")
            rows_out.append(dict(label=label, log_n_true=log_n_true, T_true=T_true, snr=snr,
                                  raw_bias_dex=raw_bias, raw_scatter_dex=raw_scatter,
                                  raw_frac_within_tol=raw_frac, raw_passed=raw_pass,
                                  filt_bias_dex=filt_bias, filt_scatter_dex=filt_scatter,
                                  filt_frac_within_tol=filt_frac, filt_passed=filt_pass,
                                  n_fellback=n_fellback, n_fail=n_fail, n_total=len(entries)))

    out_csv = "/home/yasho379/magritte_rebuilt/production/output_lut_coarse/results/noise_recovery_summary.csv"
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")


if __name__ == '__main__':
    main()
