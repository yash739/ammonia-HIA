"""Interpolation completeness for the 1D escape-probability model's coarse
grid (nh3hia/escape1d/grid.py, no --fine -- same axes as the gold Magritte LUT,
1764 points), mirroring analysis/validation/interpolation_completeness.py and
analysis/validation/interpolation_axis_completeness.py so the two models' interpolation
behaviour is directly comparable.

Same method as nh3hia/lut/interpolator.py: interpolate log-amplitude (here, the
per-group brightness temperatures, clipped at a small positive floor before
the log), then form ratios from the interpolated amplitudes.

This deliberately measures interpolation on the SAME grid used for the
retrieval's coarse characterization -- not the fine direct grid the actual
retrieval used (analysis/retrieval/escape1d_stutzki1984.py), which was built
specifically to avoid interpolation. Running hold-one-out here quantifies
*why* that choice was made: guard_masers=False means a nontrivial fraction
of rows sit at or near a genuine pole in Eq.(10) (T_B_outer_10<0 for 618/1695
rows on this grid; see magritte-quirks-2026-09-18.md Sec 4.2), and clipping
a masing amplitude to a small positive floor before taking its log is exactly
the kind of distortion a Delaunay-based interpolator would propagate to its
whole local neighbourhood -- this script measures how badly.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

from nh3hia.lut.axes import LOG_N_AXIS, T_AXIS, LOG_NDV_AXIS
from nh3hia.escape1d.grid import out_csv_for, outdir_for

AMPS = ['T_B_outer_01', 'T_B_outer_10', 'T_B_main', 'T_B_inner_21', 'T_B_inner_12']
RATIOS = [('R_01', 'T_B_outer_01'), ('R_10', 'T_B_outer_10'),
          ('R_21', 'T_B_inner_21'), ('R_12', 'T_B_inner_12')]
PASS_TOL_PCT = 3.3  # 1/3 of the ~10% observational ratio uncertainty floor


def load(path):
    df = pd.read_csv(path)
    df = df[df['converged']]
    return df.reset_index(drop=True)


def hold_one_out_3d(df, subsample=None, seed=0):
    X = df[['log_n_H2', 'T_cloud', 'log_N_dv']].values
    logA = {a: np.log10(df[a].values.clip(min=1e-12)) for a in AMPS}
    lo, hi = X.min(axis=0), X.max(axis=0)
    is_interior = np.all((X > lo + 1e-9) & (X < hi - 1e-9), axis=1)
    interior_idx = np.where(is_interior)[0]
    if subsample and len(interior_idx) > subsample:
        rng = np.random.default_rng(seed)
        interior_idx = rng.choice(interior_idx, size=subsample, replace=False)

    rows = []
    t0 = time.time()
    for k, i in enumerate(interior_idx):
        mask = np.ones(len(df), dtype=bool)
        mask[i] = False
        xi = X[i]
        pred_amp = {}
        ok = True
        for a in AMPS:
            interp = LinearNDInterpolator(X[mask], logA[a][mask])
            val = interp(xi[None, :])[0]
            if np.isnan(val):
                ok = False
                break
            pred_amp[a] = 10 ** val
        if not ok:
            continue
        rec = dict(idx=int(i), log_n_H2=xi[0], T_cloud=xi[1], log_N_dv=xi[2],
                   any_maser=bool(df['any_maser'].values[i]))
        for name, num_amp in RATIOS:
            true_val = df[name].values[i]
            pred_val = pred_amp[num_amp] / pred_amp['T_B_main']
            rec[f'err_{name}'] = abs(pred_val - true_val) / abs(true_val) * 100 if abs(true_val) > 1e-8 else np.nan
        rows.append(rec)
        if (k + 1) % 200 == 0:
            print(f"  3D: {k+1}/{len(interior_idx)} ({time.time()-t0:.0f}s elapsed)", flush=True)
    return pd.DataFrame(rows), len(interior_idx)


def axis_hold_one_out(df, axis_col, slice_cols):
    rows = []
    for slice_vals, g in df.groupby(slice_cols):
        g = g.sort_values(axis_col)
        x = g[axis_col].values
        if len(x) < 3:
            continue
        logA = {a: np.log10(g[a].values.clip(min=1e-12)) for a in AMPS}
        for k in range(1, len(x) - 1):
            xk = np.delete(x, k)
            pred_amp = {}
            for a in AMPS:
                yk = np.delete(logA[a], k)
                pred_amp[a] = 10 ** np.interp(x[k], xk, yk)
            rec = {axis_col: x[k], 'any_maser': bool(g['any_maser'].values[k])}
            for sc, sv in zip(np.atleast_1d(slice_cols), np.atleast_1d(slice_vals)):
                rec[sc] = sv
            for name, num_amp in RATIOS:
                true_val = g[name].values[k]
                pred_val = pred_amp[num_amp] / pred_amp['T_B_main']
                rec[f'err_{name}'] = abs(pred_val - true_val) / abs(true_val) * 100 if abs(true_val) > 1e-8 else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def summarize(res, label):
    print(f"\n=== {label}: {len(res)} evaluations ===")
    for name, _ in RATIOS:
        e = res[f'err_{name}'].dropna()
        if len(e) == 0:
            print(f"  {name:8s} no data")
            continue
        print(f"  {name:8s} n={len(e):4d}  median={e.median():8.2f}%  p90={e.quantile(0.9):8.2f}%  "
              f"max={e.max():10.2f}%  pass<{PASS_TOL_PCT}%: {(e < PASS_TOL_PCT).mean():.1%}")
    for name, _ in RATIOS:
        e_nm = res.loc[~res['any_maser'], f'err_{name}'].dropna()
        e_m = res.loc[res['any_maser'], f'err_{name}'].dropna()
        if len(e_nm) and len(e_m):
            print(f"  {name:8s} non-masing median={e_nm.median():6.2f}%  masing median={e_m.median():10.2f}%")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--rates', choices=['loreau', 'full_original'], default='loreau')
    ap.add_argument('--grid', default=None, help='override the grid path implied by --rates')
    ap.add_argument('--subsample', type=int, default=None)
    a = ap.parse_args()

    grid_path = a.grid or out_csv_for(a.rates, fine=False)
    OUTDIR = outdir_for(a.rates)
    suffix = f'_{a.rates}'

    df = load(grid_path)
    print(f"{len(df)} converged rows ({df['any_maser'].mean():.1%} masing)")

    print("\n--- pooled 3D hold-one-out ---")
    res3d, n_interior = hold_one_out_3d(df, subsample=a.subsample)
    print(f"{n_interior} interior points tested, {len(res3d)} scored successfully")
    res3d.to_csv(os.path.join(OUTDIR, f'interpolation_holdout_escape1d{suffix}.csv'), index=False)
    summarize(res3d, 'pooled 3D')

    fig, axes = plt.subplots(2, len(RATIOS), figsize=(4.2 * len(RATIOS), 7.5))
    for j, (name, _) in enumerate(RATIOS):
        e = res3d[f'err_{name}'].dropna()
        ax = axes[0, j]
        ax.hist(e.clip(upper=100), bins=40, color='tab:blue', alpha=0.8)
        ax.axvline(PASS_TOL_PCT, color='k', ls='--', lw=1)
        med, p90, passfrac = e.median(), e.quantile(0.9), (e < PASS_TOL_PCT).mean()
        ax.set_title(f"{name}\nmedian={med:.2f}%  p90={p90:.2f}%  pass={passfrac:.0%}", fontsize=9)
        ax.set_xlabel('hold-one-out error [%] (clipped at 100)')
        if j == 0:
            ax.set_ylabel('count')
        ax2 = axes[1, j]
        sc = ax2.scatter(res3d.log_n_H2, res3d.T_cloud, c=res3d[f'err_{name}'].clip(upper=50),
                          cmap='inferno_r', s=18, vmin=0, vmax=50)
        ax2.set_xlabel('log(n_H2)')
        if j == 0:
            ax2.set_ylabel('T_cloud [K]')
        plt.colorbar(sc, ax=ax2, label='% error', shrink=0.85)
        ax2.set_title('where it fails', fontsize=8)
    plt.suptitle(f"escape1d coarse-grid interpolation completeness -- pooled 3D hold-one-out, "
                 f"{n_interior} interior points ({len(df)} converged rows, guard_masers=False)\n"
                 f"Dashed line = {PASS_TOL_PCT}% pass threshold", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(os.path.join(OUTDIR, f'interpolation_completeness_escape1d{suffix}.png'), dpi=150)
    print(f"saved {os.path.join(OUTDIR, f'interpolation_completeness_escape1d{suffix}.png')}")

    print("\n--- axis-specific hold-one-out ---")
    res_T = axis_hold_one_out(df, 'T_cloud', ['log_n_H2', 'log_N_dv'])
    res_n = axis_hold_one_out(df, 'log_n_H2', ['T_cloud', 'log_N_dv'])
    res_T.to_csv(os.path.join(OUTDIR, f'interpolation_holdout_escape1d_Taxis{suffix}.csv'), index=False)
    res_n.to_csv(os.path.join(OUTDIR, f'interpolation_holdout_escape1d_naxis{suffix}.csv'), index=False)
    summarize(res_T, 'T_cloud axis (fixed log_n, log_Ndv)')
    summarize(res_n, 'log_n_H2 axis (fixed T, log_Ndv)')

    fig, axes = plt.subplots(2, len(RATIOS), figsize=(4.2 * len(RATIOS), 7.0))
    for j, (name, _) in enumerate(RATIOS):
        for row, (res, label) in enumerate([(res_T, 'T-axis LOO'), (res_n, 'log(n)-axis LOO')]):
            ax = axes[row, j]
            e = res[f'err_{name}'].dropna()
            if len(e) == 0:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
                continue
            ax.hist(e.clip(upper=100), bins=30, color='tab:blue', alpha=0.8)
            ax.axvline(PASS_TOL_PCT, color='k', ls='--', lw=1)
            ax.set_title(f"{label}: {name}\nmedian={e.median():.2f}%  pass={(e<PASS_TOL_PCT).mean():.0%}", fontsize=8)
            ax.set_xlabel('% error')
    plt.suptitle(f"escape1d coarse-grid axis-specific interpolation completeness\n"
                 f"Top: T_cloud axis ({len(res_T)} evals). Bottom: log(n_H2) axis ({len(res_n)} evals)",
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    plt.savefig(os.path.join(OUTDIR, f'interpolation_axis_completeness_escape1d{suffix}.png'), dpi=150)
    print(f"saved {os.path.join(OUTDIR, f'interpolation_axis_completeness_escape1d{suffix}.png')}")


if __name__ == '__main__':
    main()
