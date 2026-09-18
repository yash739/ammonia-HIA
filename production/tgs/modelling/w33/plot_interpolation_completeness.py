"""Interpolation completeness for the gold LUT: hold-one-out validation of the
log-amplitude LinearNDInterpolator (same method as lut_interpolator.py) across
whatever fraction of the grid is currently converged.

For each point strictly interior on all three axes (has real grid coverage on
both sides in log_n, T and log_N_dv -- so a genuine interpolation, not an
extrapolation, is being tested), rebuild the interpolator from every OTHER
point and predict the held-out point's four (1,1) satellite ratios. Reports
the % error distribution and where in the grid it is worst.
"""
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

DEFAULT_LUT = "/home/yasho379/magritte_rebuilt/production/output_lut_gold/results/lut_dv0.30.csv"
PC_CM = 3.0857e18
AMPS = ['A_01', 'A_10', 'A_MAIN', 'A_21', 'A_12']
RATIOS = [('R_01_MAIN', 'A_01'), ('R_10_MAIN', 'A_10'),
          ('R_21_MAIN', 'A_21'), ('R_12_MAIN', 'A_12')]
PASS_TOL_PCT = 3.3  # 1/3 of the ~10% observational ratio uncertainty floor


def load(path, mask_radius=True):
    df = pd.read_csv(path)
    df = df[df['Status'] == 'SUCCESS']
    df = df[df['convergence_ok']]
    if mask_radius:
        df = df[df['radius_sphere'] <= 0.5 * PC_CM]
    return df.reset_index(drop=True)


def hold_one_out(df, subsample=None, seed=0):
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
        rec = dict(idx=int(i), log_n_H2=xi[0], T_cloud=xi[1], log_N_dv=xi[2])
        for name, num_amp in RATIOS:
            true_val = df[name].values[i]
            pred_val = pred_amp[num_amp] / pred_amp['A_MAIN']
            rec[f'err_{name}'] = abs(pred_val - true_val) / true_val * 100 if true_val > 1e-8 else np.nan
        rows.append(rec)
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(interior_idx)} ({time.time()-t0:.0f}s elapsed)", flush=True)
    return pd.DataFrame(rows), len(interior_idx)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--lut', default=DEFAULT_LUT)
    ap.add_argument('--subsample', type=int, default=None,
                     help='cap the number of interior points tested (random subsample) for speed')
    ap.add_argument('--out', default='/home/yasho379/magritte_rebuilt/production/output_lut_gold/results/interpolation_completeness.png')
    ap.add_argument('--csv-out', default='/home/yasho379/magritte_rebuilt/production/output_lut_gold/results/interpolation_holdout_gold.csv')
    a = ap.parse_args()

    df = load(a.lut)
    print(f"{len(df)} converged rows (radius-masked)")
    res, n_interior = hold_one_out(df, subsample=a.subsample)
    print(f"{n_interior} interior points tested, {len(res)} scored successfully")
    os.makedirs(os.path.dirname(a.csv_out), exist_ok=True)
    res.to_csv(a.csv_out, index=False)

    err_cols = [f'err_{n}' for n, _ in RATIOS]
    fig, axes = plt.subplots(2, len(RATIOS), figsize=(4.2 * len(RATIOS), 7.5))

    for j, (name, _) in enumerate(RATIOS):
        e = res[f'err_{name}'].dropna()
        ax = axes[0, j]
        ax.hist(e.clip(upper=30), bins=40, color='tab:blue', alpha=0.8)
        ax.axvline(PASS_TOL_PCT, color='k', ls='--', lw=1)
        med, p90, passfrac = e.median(), e.quantile(0.9), (e < PASS_TOL_PCT).mean()
        ax.set_title(f"{name}\nmedian={med:.2f}%  p90={p90:.2f}%  pass={passfrac:.0%}", fontsize=9)
        ax.set_xlabel('hold-one-out error [%] (clipped at 30)')
        if j == 0:
            ax.set_ylabel('count')

        ax2 = axes[1, j]
        sc = ax2.scatter(res.log_n_H2, res.T_cloud, c=res[f'err_{name}'].clip(upper=20),
                          cmap='inferno_r', s=18, vmin=0, vmax=20)
        ax2.set_xlabel('log(n_H2)')
        if j == 0:
            ax2.set_ylabel('T_cloud [K]')
        plt.colorbar(sc, ax=ax2, label='% error', shrink=0.85)
        ax2.set_title('where it fails (log_n vs T, all log_Ndv overplotted)', fontsize=8)

    plt.suptitle(f"Gold LUT interpolation completeness -- hold-one-out, "
                 f"{n_interior} interior points ({len(df)} total converged rows, radius-masked)\n"
                 f"Dashed line = {PASS_TOL_PCT}% pass threshold (1/3 of ~10% observational floor)",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(a.out, dpi=150)
    print(f"saved {a.out}")
    print("\nSummary:")
    for name, _ in RATIOS:
        e = res[f'err_{name}'].dropna()
        print(f"  {name:12s} median={e.median():6.2f}%  p90={e.quantile(0.9):6.2f}%  "
              f"max={e.max():6.2f}%  pass<{PASS_TOL_PCT}%: {(e < PASS_TOL_PCT).mean():.1%}")


if __name__ == '__main__':
    main()
