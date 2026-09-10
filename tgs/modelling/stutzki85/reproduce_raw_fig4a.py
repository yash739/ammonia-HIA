"""Reproduce Stutzki & Winnewisser (1985b) Fig. 4a (T=36K and T=30K, the
F1=0->1 and F1=1->0 panels) as literally as possible: his own (1,1)<->(2,1)
rotational collision rate (from rate_swap_test.py's hybrid matrix, built
from Green 1980 Table III + Stutzki & Winnewisser 1985a Table 1) combined
with the UNGUARDED Eq.(10) formula (guard_masers=False, i.e. no sign check
on tau_G -- see nh3_escape_model.group_brightness_temperatures' docstring).

This is deliberately NOT the physically-honest default (that stays in
grid.py / make_figures.py, NaN for tau_G<0) -- this script exists only to
test whether it reproduces what his own, presumably equally unguarded, 1985
code would have plotted.
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
from grid import _arange_inclusive, FIG4_LOG_N_H2, FIG4_LOG_N_DV

TEMPS = (30.0, 36.0)


def compute(temps=TEMPS):
    log_n_h2_vals = _arange_inclusive(*FIG4_LOG_N_H2)
    log_n_dv_vals = _arange_inclusive(*FIG4_LOG_N_DV)
    model = m.NH3Model()
    rows = []
    for T in temps:
        t0 = time.time()
        Cmat = rst.build_hybrid_collision_matrix(model, T)
        x0 = None
        for log_n in log_n_h2_vals:
            n_H2 = 10.0 ** log_n
            for log_N in log_n_dv_vals:
                out = m.run_one(model, Cmat, T_k=T, n_H2=n_H2, log_N_dv=log_N,
                                 x0=x0, guard_masers=False)
                if not out['converged']:
                    out = m.run_one(model, Cmat, T_k=T, n_H2=n_H2, log_N_dv=log_N,
                                     x0=None, guard_masers=False)
                x0 = out['x']
                rows.append(dict(T_k=T, log_n_H2=log_n, log_N_dv=log_N, n_H2=n_H2,
                                  **{k: v for k, v in out.items() if k != 'x'}))
        print(f"T_k={T:.0f}K done in {time.time()-t0:.1f}s")
    return pd.DataFrame(rows)


def plot(df, out_path):
    temps = sorted(df['T_k'].unique())
    fig, axes = plt.subplots(len(temps), 2, figsize=(7.5, 3.4 * len(temps)),
                              sharex=True, sharey=True)
    if len(temps) == 1:
        axes = axes[None, :]
    for i, T in enumerate(temps):
        df_T = df[df['T_k'] == T]
        x = df_T['log_n_H2'].values
        y = df_T['log_N_dv'].values
        gx = np.linspace(x.min(), x.max(), 220)
        gy = np.linspace(y.min(), y.max(), 220)
        GX, GY = np.meshgrid(gx, gy)
        for j, (col, title) in enumerate([('R_01', r'$F_1=0\to1$'), ('R_10', r'$F_1=1\to0$')]):
            ax = axes[i, j]
            z = np.clip(df_T[col].values, -5, 5)  # clip pathological outliers only for display
            valid = np.isfinite(z)
            GZ = griddata(np.column_stack([x[valid], y[valid]]), z[valid], (GX, GY), method='linear')
            cf = ax.contourf(GX, GY, GZ, levels=20, cmap='viridis')
            cs = ax.contour(GX, GY, GZ, levels=[0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.6,2.0,2.5,3.0],
                             colors='k', linewidths=0.5)
            ax.clabel(cs, inline=True, fontsize=6, fmt='%.1f')
            if i == 0:
                ax.set_title(title, fontsize=11)
            if j == 0:
                ax.set_ylabel(f'$T_k$={T:.0f}K\n' + r'$\log(N_{NH_3}/\Delta v)$', fontsize=9)
            if i == len(temps) - 1:
                ax.set_xlabel(r'$\log(n_{H_2})$', fontsize=9)
    plt.suptitle("Literal (unguarded) reproduction with Stutzki's OWN (1,1)-(2,1) rate\n"
                  "-- compare directly against his Fig. 4a", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f'saved {out_path}')


if __name__ == '__main__':
    df = compute()
    df.to_csv('/home/yasho379/magritte_rebuilt/output_stutzki85/raw_fig4a_hybrid.csv', index=False)
    plot(df, '/home/yasho379/magritte_rebuilt/output_stutzki85/raw_fig4a_hybrid.png')
