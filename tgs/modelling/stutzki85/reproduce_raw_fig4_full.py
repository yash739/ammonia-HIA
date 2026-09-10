"""Full 4-temperature x 5-panel reproduction of Stutzki & Winnewisser (1985b)
Fig. 4a-c, using his OWN (1,1)<->(2,1) rotational collision rate (the hybrid
matrix from rate_swap_test.py) and the literal, unguarded Eq.(10) formula
(guard_masers=False -- no sign check on tau_G). See reproduce_raw_fig4a.py
and the preceding conversation for why this combination, not the physically
-guarded default in grid.py/make_figures.py, is the historically faithful
reproduction target: a 1985 implementation had no reason to separately check
the sign of tau_G, so it would have kept producing plausible-looking large
ratios exactly like this straight through the region where the escape
-probability closure's own tau>=0 assumption is violated.
"""
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import nh3_escape_model as m
import rate_swap_test as rst
import grid as gridmod

OUT_DIR = '/home/yasho379/magritte_rebuilt/output_stutzki85'

FIG4_PANELS = [
    # Column order matches stutzki_1985.pdf Fig. 4a-c exactly: Fig.4a is
    # F1=0->1 then F1=1->0 (p.18); Fig.4b is F1=2->1 (p.18) then F1=1->2,
    # explicitly labelled "Fig. 4b (continued)" on p.19 -- so 2->1 comes
    # first in his layout, not 1->2.
    ('R_01', r'$T_B(F_1=0\to1)/T_B(\Delta F_1=0)$', False),
    ('R_10', r'$T_B(F_1=1\to0)/T_B(\Delta F_1=0)$', False),
    ('R_21', r'$T_B(F_1=2\to1)/T_B(\Delta F_1=0)$', False),
    ('R_12', r'$T_B(F_1=1\to2)/T_B(\Delta F_1=0)$', False),
    ('R_2211', r'$T_B(2,2)/T_B(1,1)$', False),
    ('T_B_main', r'$T_B(1,1;\Delta F_1=0)$ [K]', True),  # Fig. 4c's brightness panel
]


def make_fig4_raw(df, out_path, rate_label="Stutzki's own (1,1)-(2,1) rate"):
    """Each of the 24 subplots gets its own colour scale (2nd-98th percentile
    of ITS OWN finite values) and its own colorbar -- a shared scale across
    the ratio panels was letting the low-n_H2 corner's extreme outliers wash
    out the minima in every other panel; per-panel scaling makes each
    panel's own structure (including the T_B_main brightness panel, which is
    in Kelvin and was never comparable to the dimensionless ratio panels'
    scale in the first place) legible on its own terms."""
    temps = sorted(df['T_k'].unique())
    n_cols = len(FIG4_PANELS)
    fig, axes = plt.subplots(len(temps), n_cols,
                              figsize=(3.6 * n_cols, 3.1 * len(temps)),
                              sharex=True, sharey=True)
    for i, T in enumerate(temps):
        df_T = df[df['T_k'] == T]
        x = df_T['log_n_H2'].values
        y = df_T['log_N_dv'].values
        gx = np.linspace(x.min(), x.max(), 220)
        gy = np.linspace(y.min(), y.max(), 220)
        GX, GY = np.meshgrid(gx, gy)
        for j, (col, title, is_temperature) in enumerate(FIG4_PANELS):
            ax = axes[i, j]
            raw = df_T[col].values
            finite = raw[np.isfinite(raw)]
            lo, hi = np.percentile(finite, [2, 98])
            if not is_temperature:
                lo = min(lo, 0.0)  # ratio panels: always show the tau=0/LTE floor if present
            z = np.clip(raw, lo, hi)
            valid = np.isfinite(z)
            GZ = griddata(np.column_stack([x[valid], y[valid]]), z[valid], (GX, GY), method='linear')
            cf = ax.contourf(GX, GY, GZ, levels=30, cmap='viridis', vmin=lo, vmax=hi)
            levels = np.linspace(lo, hi, 21)
            cs = ax.contour(GX, GY, GZ, levels=levels, colors='k', linewidths=0.35, alpha=0.7)
            ax.clabel(cs, inline=True, fontsize=5, fmt='%.2f', levels=levels[::2])
            cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(title, fontsize=10)
            if j == 0:
                ax.set_ylabel(f'$T_k$={T:.0f}K\n' + r'$\log(N_{NH_3}/\Delta v)$', fontsize=8)
            if i == len(temps) - 1:
                ax.set_xlabel(r'$\log(n_{H_2})$', fontsize=8)
    plt.suptitle(f'Literal (unguarded) reproduction of Fig. 4a-c with {rate_label}\n'
                  '(each panel individually colour-scaled to its own 2nd-98th percentile)',
                  fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f'saved {out_path}')


def hybrid_builder(model, T):
    return rst.build_hybrid_collision_matrix(model, T)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--recompute', action='store_true')
    a = ap.parse_args()

    csv_path = f'{OUT_DIR}/raw_fig4_hybrid_full.csv'
    if a.recompute:
        model = m.NH3Model()
        m.assert_expected_grouping(model)
        t0 = time.time()
        df4 = gridmod.compute_fig4_grid(model, Cmat_builder=hybrid_builder,
                                         solver_kwargs={'guard_masers': False})
        print(f"Fig4 (raw, hybrid rates): {len(df4)} points in {time.time()-t0:.1f}s, "
              f"{(~df4['converged']).sum()} unconverged, {df4['any_maser'].sum()} tau<0 points")
        df4.to_csv(csv_path, index=False)
    else:
        df4 = pd.read_csv(csv_path)
        print(f'loaded {len(df4)} rows from {csv_path}')

    make_fig4_raw(df4, f'{OUT_DIR}/raw_fig4_hybrid_full.png')
