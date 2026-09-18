"""Fit the escape-probability model directly against Stutzki & Winnewisser's
own digitized observations (Table 2, 1984) and compare the retrieved
(T_k, n_H2, N_NH3) and the high-/low-density branch structure against their
own published fits (Table 1a/1b, 1985).

Reuses tgs/modelling/w33/stutzki_tables_full.py (the already-built,
already-validated parser for the full digitized dataset -- including its
resolved ratio-column-order ambiguity, see that module's docstring) and the
region/position key-matching logic from w33/plot_chi2_landscapes.py, but
does NOT use that script's "legacy grid" (a precomputed catalogue from the
Magritte pipeline) -- this fits directly against nh3_escape_model, on the
fly, with (T_k, log_n_H2, log_N_dv) as continuous free parameters, since the
escape-probability solve is fast enough (~10ms/point) that a real per-source
optimization is cheap rather than needing a precomputed grid at all.

Chi^2 convention: Stutzki's own Table 1a/1b chi2 values use a different
(unstated) normalization than this script's weighted_chi2 (mean of
normalized-residual squares, w33/scoring.py's convention, reused here for
consistency with the rest of this codebase) -- so chi2 VALUES are not
expected to match digit-for-digit; what's being tested is whether the same
(T_k, n_H2, N_NH3) region comes out as the minimum, and whether the same
two-branch (high-density adopted / low-density rejected) degeneracy shows up
at the same approximate parameters.
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import nh3_escape_model as m

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'w33'))
import stutzki_tables_full as sf
from scoring import weighted_chi2

DV_CLUMP = 0.3  # km/s, matches Stutzki's Delta_v and this model's default

RATIO_KEYS = ('R_01_MAIN', 'R_10_MAIN', 'R_12_MAIN', 'R_21_MAIN')  # observed-table naming
MODEL_KEYS = ('R_01', 'R_10', 'R_12', 'R_21')  # this model's naming, same order


# ---------------------------------------------------------------------------
# Region/position key matching between Table 2 (1984, observations) and
# Table 1a/1b (1985, his fits) -- copied from plot_chi2_landscapes.py rather
# than importing it, to avoid that module's legacy_catalogue.py dependency
# (a precomputed pre-fix Magritte grid this script deliberately doesn't use).
# ---------------------------------------------------------------------------
def _norm_pos_multi(p):
    mm = re.match(r'\((-?\d+),\s*(-?\d+)\)\s*(.*)', p)
    if mm:
        base = f"{mm.group(1)},{mm.group(2)}"
        suffix = mm.group(3).strip()
        return f"{base} {suffix}" if suffix else base
    return p.strip()


def _to_1985_key(name, offset):
    name0 = re.sub(r'\s*\([a-z]\)\s*$', '', name).strip()
    om = {'OMC 2': 'OMC2', 'OMC S1': 'S1', 'OMC S2': 'S2', 'OMC S3': 'S3', 'OMC S4': 'S4'}
    if name0 in om:
        return ('OMC', om[name0])
    if name0.startswith('S106'):
        return ('S106', _norm_pos_multi(offset))
    if name0.startswith('S87'):
        tag = '21.0 km/s' if '21.0' in name0 else ('23.5 km/s' if '23.5' in name0 else None)
        return ('S87', f'0,0 {tag}') if tag else None
    if name0.startswith('W48'):
        base = _norm_pos_multi(offset)
        return ('W48', f'{base} 45 km/s' if '45.0' in name0 else base)
    return None


_A1_NORM = {(region, _norm_pos_multi(pos)): e for (region, pos), e in sf.TABLE_1A_FULL.items()}
_B1_NORM = {(region, _norm_pos_multi(pos)): e for (region, pos), e in sf.TABLE_1B_FULL.items()}


def sources_with_1985_fits():
    """(csv_key, key_1985) pairs for every Table-2 row that has a matching
    Table-1a entry -- i.e. exactly the positions Stutzki himself fit."""
    out = []
    for csv_key in sf.TABLE_2_1984_FULL:
        k85 = _to_1985_key(*csv_key)
        if k85 is not None and k85 in _A1_NORM:
            out.append((csv_key, k85))
    return out


# ---------------------------------------------------------------------------
# On-the-fly chi^2 fit against nh3_escape_model.
# ---------------------------------------------------------------------------
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = m.NH3Model()
    return _model


def model_ratio_vector(T_k, log_n_H2, log_N_dv, use_22=False):
    model = _get_model()
    Cmat = model.collision_matrix(T_k)
    out = m.run_one(model, Cmat, T_k=T_k, n_H2=10.0 ** log_n_H2, log_N_dv=log_N_dv)
    keys = MODEL_KEYS + (('R_2211',) if use_22 else ())
    return np.array([out.get(k, np.nan) for k in keys]), out


def _objective(params, obs_vec, sigma_vec, use_22):
    T_k, log_n_H2, log_N_dv = params
    if not (10.0 <= T_k <= 60.0 and 3.0 <= log_n_H2 <= 8.5 and 13.0 <= log_N_dv <= 17.0):
        return 1e6
    vec, out = model_ratio_vector(T_k, log_n_H2, log_N_dv, use_22=use_22)
    if not out['converged']:
        return 1e6
    # NaN entries (population-inverted group at this point) are excluded from
    # the mean by weighted_chi2's own masking, NOT hard-rejected here -- an
    # earlier version rejected any point with even one NaN component, which
    # flattens the objective across an entire masing neighbourhood (all
    # points equally 1e6) and stalls Nelder-Mead with nowhere to step.
    # Letting the surviving finite components still carry gradient signal
    # lets the optimizer walk through/around masing regions instead of
    # getting stuck at the seed.
    chi2 = weighted_chi2(vec, obs_vec, sigma_vec)
    if np.isnan(chi2):
        return 1e6  # every component masered -- truly no signal here
    return chi2


def fit_source(csv_key, use_22=False, seeds=None):
    """Returns list of local-minimum fit results (one per seed basin),
    sorted by chi2 ascending -- seeds are chosen to land in the high-density
    and low-density branches separately, matching Stutzki's own Table 1a/1b."""
    e = sf.TABLE_2_1984_FULL[csv_key]
    if any(e[k] is None for k in RATIO_KEYS):
        return None
    obs_vec = np.array([e[k] for k in RATIO_KEYS])
    sigma_vec = np.array([e[k + '_err'] or 0.1 * e[k] for k in RATIO_KEYS])
    use_22 = use_22 and e.get('R_22_MAIN') is not None
    if use_22:
        obs_vec = np.append(obs_vec, e['R_22_MAIN'])
        sigma_vec = np.append(sigma_vec, e.get('R_22_MAIN_err') or 0.05)

    if seeds is None:
        seeds = [(25.0, 6.5, 14.3), (25.0, 4.3, 14.0)]  # high-density, low-density

    results = []
    for T0, n0, N0 in seeds:
        res = minimize(_objective, x0=[T0, n0, N0], args=(obs_vec, sigma_vec, use_22),
                        method='Nelder-Mead',
                        options=dict(xatol=1e-3, fatol=1e-5, maxiter=400, adaptive=True))
        T_k, log_n_H2, log_N_dv = res.x
        vec, out = model_ratio_vector(T_k, log_n_H2, log_N_dv, use_22=use_22)
        eta_f = e['T_B_11'] / out['T_B_main'] if out['T_B_main'] else np.nan
        results.append(dict(T_k=T_k, log_n_H2=log_n_H2, log_N_dv=log_N_dv,
                             chi2=res.fun, converged=out['converged'],
                             T_B_theor=out['T_B_main'], T_B_obs=e['T_B_11'], eta_f=eta_f,
                             seed=(T0, n0, N0)))
    results.sort(key=lambda r: r['chi2'])
    # de-duplicate near-identical basins (both seeds converging to the same point)
    dedup = []
    for r in results:
        if not any(abs(r['log_n_H2'] - d['log_n_H2']) < 0.3 and abs(r['T_k'] - d['T_k']) < 3
                   for d in dedup):
            dedup.append(r)
    return dedup


if __name__ == '__main__':
    targets = sources_with_1985_fits()
    print(f'{len(targets)} positions have a matching Table 1a fit')

    rows = []
    t0 = time.time()
    for csv_key, key85 in targets:
        fit_a = _A1_NORM[key85]
        fit_b = _B1_NORM.get(key85)
        use_22 = fit_a.get('includes_22_hfs', False)
        branches = fit_source(csv_key, use_22=use_22)
        if branches is None:
            print(f'{csv_key}: non-detection, skipped')
            continue
        for i, br in enumerate(branches):
            rows.append(dict(source=csv_key[0], offset=csv_key[1], branch=i,
                              **br,
                              stutzki_a_Tk=fit_a['T_k'], stutzki_a_log_n=fit_a['log_nH2'],
                              stutzki_a_log_N=fit_a['log_N_NH3'] - np.log10(DV_CLUMP),
                              stutzki_a_chi2=fit_a['chi2'],
                              stutzki_b_Tk=fit_b['T_k'] if fit_b else np.nan,
                              stutzki_b_log_n=fit_b['log_nH2'] if fit_b else np.nan,
                              stutzki_b_log_N=(fit_b['log_N_NH3'] - np.log10(DV_CLUMP)) if fit_b else np.nan))
        print(f'{csv_key}: {len(branches)} basin(s) found, '
              f'best log_n={branches[0]["log_n_H2"]:.2f} T={branches[0]["T_k"]:.1f} chi2={branches[0]["chi2"]:.3f} '
              f'(Stutzki 1a: log_n={fit_a["log_nH2"]:.2f} T={fit_a["T_k"]:.1f})')

    print(f'\ntotal time {time.time()-t0:.1f}s')
    df = pd.DataFrame(rows)
    out_csv = '/home/yasho379/magritte_rebuilt/scratch/output/output_stutzki85/fit_observations.csv'
    df.to_csv(out_csv, index=False)
    print(f'saved {out_csv}')
