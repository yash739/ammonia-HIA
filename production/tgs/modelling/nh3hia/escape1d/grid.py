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

import nh3hia.escape1d.model as m
import nh3hia.escape1d.rates_stutzki as rst_full
from nh3hia.lut.axes import LOG_N_AXIS, T_AXIS, LOG_NDV_AXIS
from nh3hia import paths

BASE_OUTDIR = paths.PRODUCTION + "/output_escape1d/"
# 'loreau' = hyperfine-resolved NH3-H2 rates (Loreau et al. 2023), the
# default throughout this project. 'full_original' = Stutzki & Winnewisser's
# own 1985 rates via the complete transcription of Green (1980) Table III x
# Stutzki's Table 1 (nh3hia/escape1d/rates_stutzki.py, built from the CSVs in
# production/references/) -- 95% of the off-diagonal entries among the six
# tracked manifolds are Stutzki-sourced; only 443U/444U's outgoing pathways
# fall back to Loreau (nothing higher than (4,4) exists in this scheme for
# them to excite into, so that fallback is expected, not a gap).
# (A third, partial reconstruction called 'original' was retired on
# 2026-09-24; its code is in archive/escape1d_early/rate_swap_full.py.)
RATES_DIRNAME = {'loreau': 'results_loreau_rates',
                 'full_original': 'results_full_original_rates'}
RATES_FN = {
    'loreau': lambda model, T: model.collision_matrix(T),
    'full_original': rst_full.build_complete_original_collision_matrix,
}


def outdir_for(rates, dv_kms=None):
    """dv_kms=None (or the reference 0.3 km/s) reproduces the existing
    'results_{rates}_rates' path byte-for-byte, so nothing already on disk
    at the reference linewidth moves. Any other dv_kms gets its own
    'results_{rates}_rates_dv{dv:.2f}' directory."""
    base = RATES_DIRNAME[rates]
    if dv_kms is not None and abs(dv_kms - m.DV_CLUMP_DEFAULT_KMS) > 1e-9:
        base = f'{base}_dv{dv_kms:.2f}'
    d = os.path.join(BASE_OUTDIR, base)
    os.makedirs(d, exist_ok=True)
    return d


def out_csv_for(rates, fine=False, dv_kms=None):
    return os.path.join(outdir_for(rates, dv_kms=dv_kms), 'grid_fine.csv' if fine else 'grid.csv')


# Backward-compat aliases -- unchanged Loreau paths, still importable as
# before for scripts that haven't been switched to the --rates flag.
OUT_CSV = out_csv_for('loreau', fine=False)
FINE_OUT_CSV = out_csv_for('loreau', fine=True)
GUARD_MASERS = False
DV_CLUMP_KMS = m.DV_CLUMP_DEFAULT_KMS

AMP_KEYS = ['T_B_main', 'T_B_outer_01', 'T_B_outer_10', 'T_B_inner_12',
            'T_B_inner_21', 'T_B_22_main', 'T_B_21_main']
RATIO_KEYS = ['R_01', 'R_10', 'R_12', 'R_21', 'R_2211', 'R_21_11']


def compute_grid(log_n_vals, T_vals, log_ndv_vals, out_csv, collision_matrix_fn=None,
                  dv_clump_kms=None):
    collision_matrix_fn = collision_matrix_fn or RATES_FN['loreau']
    dv_clump_kms = DV_CLUMP_KMS if dv_clump_kms is None else dv_clump_kms
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
                                dv_clump_kms=dv_clump_kms, x0=x0, guard_masers=GUARD_MASERS)
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
    ap.add_argument('--rates', choices=['loreau', 'full_original'], default='loreau')
    ap.add_argument('--log-n-max', type=float, default=None,
                    help="override the fine grid's upper log10(n_H2) bound (default: gold's "
                         "own ceiling, max(LOG_N_AXIS)=7.5). Writes to a separate "
                         "'grid_fine_ext<max>.csv' file so the standard grid_fine.csv is left "
                         "untouched -- extending the ceiling is an experiment, not a replacement.")
    ap.add_argument('--dv', type=float, default=None,
                    help="clump linewidth (km/s), default the reference 0.3 (m.DV_CLUMP_DEFAULT_KMS). "
                         "Any other value writes to its own 'results_{rates}_rates_dv{dv:.2f}/' "
                         "directory (see outdir_for) -- the log_N_dv AXIS values are unchanged, so "
                         "this tests how the assumed clump linewidth alone shifts the predicted "
                         "anomaly at fixed N_NH3/dv, per the master plan's L3 motivation.")
    a = ap.parse_args()

    cmat_fn = RATES_FN[a.rates]
    if a.fine:
        log_n_lo = min(LOG_N_AXIS)
        log_n_hi = a.log_n_max if a.log_n_max is not None else max(LOG_N_AXIS)
        if a.log_n_max is not None:
            out_csv = os.path.join(outdir_for(a.rates, dv_kms=a.dv), f'grid_fine_ext{a.log_n_max:.1f}.csv')
        else:
            out_csv = out_csv_for(a.rates, fine=True, dv_kms=a.dv)
        log_n_vals = np.linspace(log_n_lo, log_n_hi, a.n_log_n)
        T_vals = np.linspace(min(T_AXIS), max(T_AXIS), a.n_T)
        ndv_vals = np.linspace(min(LOG_NDV_AXIS), max(LOG_NDV_AXIS), a.n_log_ndv)
        compute_grid(log_n_vals, T_vals, ndv_vals, out_csv, collision_matrix_fn=cmat_fn, dv_clump_kms=a.dv)
    else:
        out_csv = out_csv_for(a.rates, fine=a.fine, dv_kms=a.dv)
        compute_grid(LOG_N_AXIS, T_AXIS, LOG_NDV_AXIS, out_csv, collision_matrix_fn=cmat_fn, dv_clump_kms=a.dv)


if __name__ == '__main__':
    main()
