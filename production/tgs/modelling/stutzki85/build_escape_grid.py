"""Full (log_n_H2, T_cloud, log_N_dv) grid via the 1D escape-probability
model, on the EXACT axis values used by the Magritte "gold" LUT
(build_lut_gold.LOG_N_AXIS/T_AXIS/LOG_NDV_AXIS), so a chi^2 retrieval against
this grid is directly comparable to the gold-LUT retrieval.

guard_masers=False throughout (per instruction): masing groups (tau_G<0)
report the literal, un-sign-checked Eq.(10) brightness rather than NaN -- see
nh3_escape_model.group_brightness_temperatures' docstring for why that is
the historically-faithful choice, matching what Stutzki's own code almost
certainly did. `any_maser` is still recorded per row so this can be revisited.

Walks (T, log_n) rows with log_N_dv always ascending from a cold thermal
start (same anti-bistability ordering as grid.compute_fig4_grid), warm-
starting within each row and carrying the previous row's converged
populations across n_H2 for speed -- the model is cheap (~10-60 ms/point)
so the full 1764-point grid takes minutes, not the hours/days the Magritte
LUT needs.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'w33'))
import nh3_escape_model as m
import rate_swap_full as rsf
import rate_swap_transcribed as rst_full
from build_lut_gold import LOG_N_AXIS, T_AXIS, LOG_NDV_AXIS

BASE_OUTDIR = "/home/yasho379/magritte_rebuilt/production/output_escape1d/"
# 'loreau' = hyperfine-resolved NH3-H2 rates (Loreau et al. 2023), the
# default throughout this project. 'original' = Stutzki & Winnewisser's own
# 1985 rates via the PARTIAL hand-transcribed (1,1)-source-only hybrid
# (rate_swap_full.build_full_hybrid_collision_matrix) -- kept for
# comparison against the fuller version below. 'full_original' = the same
# rates via the COMPLETE transcription of Green (1980) Table III x
# Stutzki's Table 1 (rate_swap_transcribed.py, built from user-supplied
# CSVs in production/references/) -- 95% of the off-diagonal entries among
# the six tracked manifolds are Stutzki-sourced (vs a small fraction under
# 'original'), only 443U/444U's outgoing pathways fall back to Loreau
# (nothing higher than (4,4) exists in this scheme for them to excite
# into, so that fallback is expected, not a gap).
RATES_DIRNAME = {'loreau': 'results_loreau_rates', 'original': 'results_original_rates',
                  'full_original': 'results_full_original_rates'}
RATES_FN = {
    'loreau': lambda model, T: model.collision_matrix(T),
    'original': rsf.build_full_hybrid_collision_matrix,
    'full_original': rst_full.build_complete_original_collision_matrix,
}


def outdir_for(rates):
    d = os.path.join(BASE_OUTDIR, RATES_DIRNAME[rates])
    os.makedirs(d, exist_ok=True)
    return d


def out_csv_for(rates, fine=False):
    return os.path.join(outdir_for(rates), 'grid_fine.csv' if fine else 'grid.csv')


# Backward-compat aliases -- unchanged Loreau paths, still importable as
# before for scripts that haven't been switched to the --rates flag.
OUT_CSV = out_csv_for('loreau', fine=False)
FINE_OUT_CSV = out_csv_for('loreau', fine=True)
GUARD_MASERS = False
DV_CLUMP_KMS = m.DV_CLUMP_DEFAULT_KMS

AMP_KEYS = ['T_B_main', 'T_B_outer_01', 'T_B_outer_10', 'T_B_inner_12',
            'T_B_inner_21', 'T_B_22_main', 'T_B_21_main']
RATIO_KEYS = ['R_01', 'R_10', 'R_12', 'R_21', 'R_2211', 'R_21_11']


def compute_grid(log_n_vals, T_vals, log_ndv_vals, out_csv, collision_matrix_fn=None):
    collision_matrix_fn = collision_matrix_fn or RATES_FN['loreau']
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    model = m.NH3Model()
    m.assert_expected_grouping(model)

    rows = []
    t_start = time.time()
    for T in T_vals:
        Cmat = collision_matrix_fn(model, T)
        x0 = None
        n_fail = 0
        t_row = time.time()
        for log_n in log_n_vals:
            n_H2 = 10.0 ** log_n
            for log_Ndv in log_ndv_vals:  # ascending -- tracks the low-tau-connected branch
                out = m.run_one(model, Cmat, T_k=T, n_H2=n_H2, log_N_dv=log_Ndv,
                                dv_clump_kms=DV_CLUMP_KMS, x0=x0, guard_masers=GUARD_MASERS)
                x0 = out['x']
                if not out['converged']:
                    n_fail += 1
                row = dict(log_n_H2=log_n, T_cloud=T, log_N_dv=log_Ndv,
                          converged=out['converged'], any_maser=out['any_maser'],
                          tau_main=out['tau_main'], n_iter=out['n_iter'])
                for k in AMP_KEYS + RATIO_KEYS:
                    row[k] = out[k]
                rows.append(row)
        n_pts = len(log_n_vals) * len(log_ndv_vals)
        print(f"T={T:.0f}K: {n_pts} points in {time.time()-t_row:.1f}s, {n_fail} unconverged", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}  ({len(df)} rows, {time.time()-t_start:.1f}s total)")
    print(f"unconverged: {(~df.converged).sum()}/{len(df)}   any_maser: {df.any_maser.sum()}/{len(df)}")
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--fine', action='store_true',
                    help="finer DIRECT grid (no interpolation needed downstream) instead of "
                         "gold's exact 14x14x9 axes. Chosen over interpolating the coarse grid "
                         "because guard_masers=False lets some ratios diverge near the masing "
                         "threshold (R_01 up to ~1e13 seen) -- a literal pole that would corrupt "
                         "a LinearNDInterpolator's whole local neighbourhood, not just that point. "
                         "The model is cheap enough (~9 ms/point) to just compute a fine grid "
                         "directly instead: poles then stay isolated single cells.")
    ap.add_argument('--n-log-n', type=int, default=70)
    ap.add_argument('--n-T', type=int, default=50)
    ap.add_argument('--n-log-ndv', type=int, default=36)
    ap.add_argument('--rates', choices=['loreau', 'original', 'full_original'], default='loreau')
    a = ap.parse_args()

    out_csv = out_csv_for(a.rates, fine=a.fine)
    cmat_fn = RATES_FN[a.rates]
    if a.fine:
        log_n_vals = np.linspace(min(LOG_N_AXIS), max(LOG_N_AXIS), a.n_log_n)
        T_vals = np.linspace(min(T_AXIS), max(T_AXIS), a.n_T)
        ndv_vals = np.linspace(min(LOG_NDV_AXIS), max(LOG_NDV_AXIS), a.n_log_ndv)
        compute_grid(log_n_vals, T_vals, ndv_vals, out_csv, collision_matrix_fn=cmat_fn)
    else:
        compute_grid(LOG_N_AXIS, T_AXIS, LOG_NDV_AXIS, out_csv, collision_matrix_fn=cmat_fn)


if __name__ == '__main__':
    main()
