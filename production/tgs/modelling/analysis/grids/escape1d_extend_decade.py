"""Extend an existing fine escape1d grid (nh3hia/escape1d/grid.py --fine) one
decade higher in log_n_H2, WITHOUT recomputing the already-done 3.5-7.5
range. Reuses the base grid_fine.csv rows as-is and only solves the new
7.5-8.5 slice, at the same T_cloud/log_N_dv axis values (same np.linspace
calls against T_AXIS/LOG_NDV_AXIS the base grid used, so the two slices
concatenate into one clean regular grid), then writes the merged result to
grid_fine_ext8.5.csv.

Motivation: check whether the log_n_H2=7.5 ceiling pinning seen in the
Stutzki/W33 retrievals (build_lut_gold.LOG_N_AXIS's own max) is a genuine
chi^2 optimum or just the axis truncating the answer -- extending headroom
one decade higher is the cheap way to find out, and redoing the 126000
already-converged points for that would be pure waste.
"""
import os
import sys

import numpy as np
import pandas as pd

import nh3hia.escape1d.grid as beg
from nh3hia.lut.axes import LOG_N_AXIS, T_AXIS, LOG_NDV_AXIS

N_LOG_N_PER_DECADE = 18  # matches base grid's own point density (~17.5/dex for 70 pts over 4.0 dex)


def extend_one_decade(rates, n_T=50, n_log_ndv=36):
    base_csv = beg.out_csv_for(rates, fine=True)
    base = pd.read_csv(base_csv)
    hi_old = max(LOG_N_AXIS)  # 7.5 -- already fully covered by base
    hi_new = hi_old + 1.0     # 8.5

    T_vals = np.linspace(min(T_AXIS), max(T_AXIS), n_T)
    ndv_vals = np.linspace(min(LOG_NDV_AXIS), max(LOG_NDV_AXIS), n_log_ndv)
    # include hi_old as an overlap/consistency check point, drop it after
    new_log_n_vals = np.linspace(hi_old, hi_new, N_LOG_N_PER_DECADE)

    cmat_fn = beg.RATES_FN[rates]
    tmp_csv = os.path.join(beg.outdir_for(rates), f'_tmp_extra_decade_{rates}.csv')
    print(f"[{rates}] base grid already has {len(base)} rows at log_n<= {hi_old} -- "
          f"NOT recomputing those. Solving only the new {hi_old}-{hi_new} slice "
          f"({len(new_log_n_vals)} x {n_T} x {n_log_ndv} = "
          f"{len(new_log_n_vals)*n_T*n_log_ndv} points)...", flush=True)
    new_df = beg.compute_grid(new_log_n_vals, T_vals, ndv_vals, tmp_csv, collision_matrix_fn=cmat_fn)

    overlap = new_df[np.isclose(new_df.log_n_H2, hi_old)]
    base_at_hi = base[np.isclose(base.log_n_H2, hi_old)].sort_values(['T_cloud', 'log_N_dv'])
    overlap_sorted = overlap.sort_values(['T_cloud', 'log_N_dv'])
    if len(overlap) == len(base_at_hi):
        for k in ['R_01', 'R_10', 'R_21', 'R_12', 'R_2211', 'T_B_main']:
            d = np.abs(overlap_sorted[k].values - base_at_hi[k].values)
            finite = np.isfinite(d)
            if finite.any():
                print(f"  overlap check {k}: max |diff| = {d[finite].max():.3e} "
                      f"(n={finite.sum()}/{len(d)} finite)")
    else:
        print(f"  WARNING: overlap row count mismatch ({len(overlap)} vs {len(base_at_hi)}) -- "
              f"skipping consistency check")

    new_only = new_df[new_df.log_n_H2 > hi_old + 1e-9]
    combined = pd.concat([base, new_only], ignore_index=True)
    # round the axis columns -- base (CSV round-tripped) and new_only (freshly
    # computed in this process) can otherwise differ by ~1e-15 on nominally
    # identical np.linspace(T_AXIS) values, which fooled a downstream
    # exact-uniqueness check into seeing 51 T's instead of 50 (verified: only
    # ever one such near-duplicate per axis, at the 1e-15 level -- true float
    # noise, not real new grid points).
    for col in ['log_n_H2', 'T_cloud', 'log_N_dv']:
        combined[col] = combined[col].round(8)
    out_csv = os.path.join(beg.outdir_for(rates), 'grid_fine_ext8.5.csv')
    combined.to_csv(out_csv, index=False)
    os.remove(tmp_csv)
    print(f"[{rates}] saved {out_csv}  ({len(combined)} rows = {len(base)} reused + "
          f"{len(new_only)} newly computed)")
    return out_csv


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--rates', choices=['loreau', 'full_original'], default='loreau')
    a = ap.parse_args()
    extend_one_decade(a.rates)


if __name__ == '__main__':
    main()
