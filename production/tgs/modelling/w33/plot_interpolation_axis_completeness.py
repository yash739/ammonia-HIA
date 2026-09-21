"""Axis-specific interpolation completeness for the gold LUT: 1D hold-one-out
along the T_cloud axis and along the log_n_H2 axis separately, each at fixed
values of the other two axes -- complementary to
plot_interpolation_completeness.py's pooled 3D check, since it isolates which
AXIS is under-resolved rather than mixing all three together.

Method matches this project's earlier per-axis checks (viz_interp_temp_axis.py,
the log_Ndv-axis check): at each "slice" (fixed pair of the other two axes)
with full coverage along the axis under test, hold out each INTERIOR value on
that axis (endpoints excluded -- an edge point has no bracketing data, so
leave-one-out there would test extrapolation, not interpolation), rebuild a
1D linear interpolator on log-amplitude from the remaining points on that same
slice, and predict the held-out point's four (1,1) satellite ratios.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_lut_gold import LOG_N_AXIS, T_AXIS, LOG_NDV_AXIS

DEFAULT_LUT = "/home/yasho379/magritte_rebuilt/production/output_lut_gold/results/lut_dv0.30.csv"
PC_CM = 3.0857e18
AMPS = ['A_01', 'A_10', 'A_MAIN', 'A_21', 'A_12']
RATIOS = [('R_01_MAIN', 'A_01'), ('R_10_MAIN', 'A_10'),
          ('R_21_MAIN', 'A_21'), ('R_12_MAIN', 'A_12')]
PASS_TOL_PCT = 3.3


def load(path, mask_radius=False):
    # mask_radius off by default -- matches lut_interpolator.py's
    # DEFAULT_MAX_RADIUS_PC=None policy (convergence_ok is the real
    # data-quality gate; the radius mask was an extra precaution, not a
    # substitute). Pass mask_radius=True to re-enable it.
    df = pd.read_csv(path)
    df = df[df['Status'] == 'SUCCESS']
    df = df[df['convergence_ok']]
    if mask_radius:
        df = df[df['radius_sphere'] <= 0.5 * PC_CM]
    return df.reset_index(drop=True)


def axis_hold_one_out(df, axis_col, slice_cols):
    """1D hold-one-out along axis_col, grouped by (slice_cols) pairs.
    Returns a DataFrame of per-point errors, one row per (slice, held-out axis value).
    """
    rows = []
    for slice_vals, g in df.groupby(slice_cols):
        g = g.sort_values(axis_col)
        x = g[axis_col].values
        if len(x) < 3:
            continue  # need at least one true interior point
        logA = {a: np.log10(g[a].values.clip(min=1e-12)) for a in AMPS}
        for k in range(1, len(x) - 1):  # interior only
            xk = np.delete(x, k)
            pred_amp = {}
            for a in AMPS:
                yk = np.delete(logA[a], k)
                pred_amp[a] = 10 ** np.interp(x[k], xk, yk)
            rec = {axis_col: x[k]}
            for sc, sv in zip(np.atleast_1d(slice_cols), np.atleast_1d(slice_vals)):
                rec[sc] = sv
            for name, num_amp in RATIOS:
                true_val = g[name].values[k]
                pred_val = pred_amp[num_amp] / pred_amp['A_MAIN']
                rec[f'err_{name}'] = abs(pred_val - true_val) / true_val * 100 if true_val > 1e-8 else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def summarize(res, label):
    print(f"\n=== {label}: {len(res)} interior evaluations ===")
    for name, _ in RATIOS:
        e = res[f'err_{name}'].dropna()
        if len(e) == 0:
            print(f"  {name:12s} no data")
            continue
        print(f"  {name:12s} n={len(e):4d}  median={e.median():6.2f}%  p90={e.quantile(0.9):6.2f}%  "
              f"max={e.max():6.2f}%  pass<{PASS_TOL_PCT}%: {(e < PASS_TOL_PCT).mean():.1%}")


def plot_summary(ax_row, res, label):
    for j, (name, _) in enumerate(RATIOS):
        ax = ax_row[j]
        e = res[f'err_{name}'].dropna()
        if len(e) == 0:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
            continue
        ax.hist(e.clip(upper=30), bins=30, color='tab:blue', alpha=0.8)
        ax.axvline(PASS_TOL_PCT, color='k', ls='--', lw=1)
        ax.set_title(f"{label}: {name}\nmedian={e.median():.2f}%  pass={(e<PASS_TOL_PCT).mean():.0%}", fontsize=8)
        ax.set_xlabel('% error')


def coverage_stats(df, axis_col, axis_vals, slice_cols):
    """How many (slice) groups have full coverage along axis_col."""
    full = 0
    total = 0
    for _, g in df.groupby(slice_cols):
        total += 1
        if set(np.round(g[axis_col].values, 4)) >= set(np.round(axis_vals, 4)):
            full += 1
    return full, total


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--lut', default=DEFAULT_LUT)
    ap.add_argument('--out', default='/home/yasho379/magritte_rebuilt/production/output_lut_gold/results/interpolation_axis_completeness.png')
    ap.add_argument('--csv-out-prefix', default='/home/yasho379/magritte_rebuilt/production/output_lut_gold/results/interpolation_holdout')
    a = ap.parse_args()

    df = load(a.lut)
    print(f"{len(df)} converged rows (radius mask: off)")

    n_full, n_tot = coverage_stats(df, 'T_cloud', np.array(T_AXIS), ['log_n_H2', 'log_N_dv'])
    print(f"log_n x log_Ndv slices with FULL T coverage (all 14): {n_full}/{n_tot}")
    d_full, d_tot = coverage_stats(df, 'log_n_H2', np.array(LOG_N_AXIS), ['T_cloud', 'log_N_dv'])
    print(f"T x log_Ndv slices with FULL log_n coverage (all 14): {d_full}/{d_tot}")

    res_T = axis_hold_one_out(df, 'T_cloud', ['log_n_H2', 'log_N_dv'])
    res_n = axis_hold_one_out(df, 'log_n_H2', ['T_cloud', 'log_N_dv'])
    res_T.to_csv(f'{a.csv_out_prefix}_Taxis.csv', index=False)
    res_n.to_csv(f'{a.csv_out_prefix}_naxis.csv', index=False)
    summarize(res_T, 'T_cloud axis (fixed log_n, log_Ndv)')
    summarize(res_n, 'log_n_H2 axis (fixed T, log_Ndv)')

    fig, axes = plt.subplots(2, len(RATIOS), figsize=(4.2 * len(RATIOS), 7.0))
    plot_summary(axes[0], res_T, 'T-axis LOO')
    plot_summary(axes[1], res_n, 'log(n)-axis LOO')
    plt.suptitle(f"Gold LUT axis-specific interpolation completeness (hold-one-out along a single axis, "
                 f"other two fixed)\nTop: T_cloud axis ({len(res_T)} evals, {n_full}/{n_tot} slices fully "
                 f"covered). Bottom: log(n_H2) axis ({len(res_n)} evals, {d_full}/{d_tot} slices fully covered)",
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    plt.savefig(a.out, dpi=150)
    print(f"\nsaved {a.out}")


if __name__ == '__main__':
    main()
