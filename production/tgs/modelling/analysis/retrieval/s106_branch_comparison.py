"""One-off deep dive on S106 (0,0) under full_original rates: the raw
chi2-best (global) retrieval lands on the low-density branch and its
(2,1)/(1,1) prediction under-shoots the observation (see
analysis/retrieval/escape1d_stutzki1984.py's regular output). This regenerates
that position's diagnostic figure with the HIGH-density branch's local
minimum added alongside it, since it is (a) nearly chi2-degenerate with
the global best (Delta chi2=0.10) and (b) the one that actually passes
the physically_consistent filter used throughout this project -- the
low-density point fails it (K_predicted << K_required). Comparing both
branches' (2,1) prediction is the point of this script.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import nh3hia.stutzki.params as sp
from nh3hia.lut.interpolator import chi2_retrieve
from nh3hia.stutzki.physics import eta_f as calc_eta_f, clump_count_K, clump_count_required, physically_consistent
from analysis.retrieval.magritte_stutzki1984 import gaussian_spectrum, chi2_slice_plot
from nh3hia.escape1d.grid import out_csv_for, outdir_for, DV_CLUMP_KMS

RATIO_KEYS = ('R_01', 'R_10', 'R_21', 'R_12', 'R_2211')
STUTZKI_KEY = {'R_01': 'R_01', 'R_10': 'R_10', 'R_21': 'R_21', 'R_12': 'R_12', 'R_2211': 'R_22_MAIN'}
RATIO_21 = 'R_21_11'
REGION, POS = 'S106', '0,0'


def load_grid(path):
    df = pd.read_csv(path)
    ratio_cols = ['R_01', 'R_10', 'R_12', 'R_21', 'R_2211', 'R_21_11', 'T_B_main']
    df.loc[~df['converged'], ratio_cols] = np.nan
    return df


def main():
    grid_path = out_csv_for('full_original', fine=True)
    outdir = os.path.join(outdir_for('full_original'), 'stutzki1985_retrieval')
    os.makedirs(outdir, exist_ok=True)

    df = load_grid(grid_path)
    pts = df[['log_n_H2', 'T_cloud', 'log_N_dv']].values
    ratio_grid = {k: df[k].values for k in RATIO_KEYS}
    ratio_grid['A_MAIN'] = df['T_B_main'].values
    ratio_grid[RATIO_21] = df[RATIO_21].values

    log_n_ax = np.sort(df.log_n_H2.unique())
    T_ax = np.sort(df.T_cloud.unique())
    ndv_ax = np.sort(df.log_N_dv.unique())
    shape3d = (len(log_n_ax), len(T_ax), len(ndv_ax))
    assert len(df) == np.prod(shape3d)
    order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
    pts, ratio_grid = pts[order], {k: v[order] for k, v in ratio_grid.items()}

    entry = sp.TABLE_2_1984[(REGION, POS)]
    a1a = sp.TABLE_1A.get(REGION, {}).get(POS)
    obs = {k: entry.get(STUTZKI_KEY[k], np.nan) for k in RATIO_KEYS}
    err = {k: entry.get(STUTZKI_KEY[k] + '_err', np.nan) for k in RATIO_KEYS}
    obs_vec = np.array([obs[k] for k in RATIO_KEYS])
    sigma_vec = np.array([err[k] for k in RATIO_KEYS])

    best_idx, chi2_best, chi2_all = chi2_retrieve(None, pts, ratio_grid, obs_vec, sigma_vec, ratio_keys=RATIO_KEYS)
    chi2_3d = chi2_all.reshape(shape3d)

    def branch_point(flat_idx):
        log_n_fit, T_fit, ndv_fit = pts[flat_idx]
        A_MAIN_fit = float(ratio_grid['A_MAIN'][flat_idx])
        R21_11_pred = float(ratio_grid[RATIO_21][flat_idx])
        chi2_here = float(chi2_all[flat_idx])
        T_B_obs = entry['T_B_11']
        dist = sp.SOURCE_DISTANCE_PC.get(REGION, 500.0)
        dv_obs = entry['dv']
        ef = calc_eta_f(T_B_obs, A_MAIN_fit)
        K_req = clump_count_required(dv_obs, DV_CLUMP_KMS)
        K_pred = clump_count_K(T_fit, 10 ** log_n_fit, ef, dist, dv_obs, DV_CLUMP_KMS) if ef and ef > 0 else np.nan
        consistent, detail = physically_consistent(T_fit, 10 ** log_n_fit, T_B_obs, A_MAIN_fit,
                                                    dv_obs, DV_CLUMP_KMS, dist)
        model_ratios = {k: float(ratio_grid[k][flat_idx]) for k in RATIO_KEYS}
        return dict(log_n=log_n_fit, T=T_fit, ndv=ndv_fit, chi2=chi2_here, A_MAIN=A_MAIN_fit,
                    R21_11=R21_11_pred, eta_f=ef, K_pred=K_pred, K_req=K_req, consistent=consistent,
                    model_ratios=model_ratios)

    low = branch_point(best_idx)

    mask_hi = log_n_ax >= 6.0
    sub = chi2_3d[mask_hi]
    idx_hi = np.unravel_index(np.nanargmin(sub), sub.shape)
    i_n = np.where(mask_hi)[0][idx_hi[0]]
    flat_idx_hi = np.ravel_multi_index((i_n, idx_hi[1], idx_hi[2]), shape3d)
    high = branch_point(flat_idx_hi)

    print(f"LOW branch:  log_n={low['log_n']:.3f} T={low['T']:.1f} chi2={low['chi2']:.3f} "
          f"R21_11={low['R21_11']:.4f} eta_f={low['eta_f']:.3f} consistent={low['consistent']}")
    print(f"HIGH branch: log_n={high['log_n']:.3f} T={high['T']:.1f} chi2={high['chi2']:.3f} "
          f"R21_11={high['R21_11']:.4f} eta_f={high['eta_f']:.3f} K_pred={high['K_pred']:.2f} "
          f"K_req={high['K_req']:.2f} consistent={high['consistent']}")

    T_B_obs = entry['T_B_11']
    dv_obs = entry['dv']
    v = np.linspace(-32, 32, 800)
    obs_amps = [T_B_obs * entry['R_10'], T_B_obs * entry['R_12'], T_B_obs,
                T_B_obs * entry['R_21'], T_B_obs * entry['R_01']]
    obs_spec = gaussian_spectrum(v, obs_amps, dv_obs)

    def model_spec_for(branch):
        amps = [branch['A_MAIN'] * branch['model_ratios']['R_10'], branch['A_MAIN'] * branch['model_ratios']['R_12'],
                branch['A_MAIN'], branch['A_MAIN'] * branch['model_ratios']['R_21'],
                branch['A_MAIN'] * branch['model_ratios']['R_01']]
        return gaussian_spectrum(v, amps, DV_CLUMP_KMS)

    t321 = sp.TABLE_3_21.get((REGION, POS))

    fig = plt.figure(figsize=(21, 10))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.9)

    ax_spec = fig.add_subplot(gs[0, 0:3])
    ax_spec.plot(v, obs_spec, color='k', lw=1.6, label=f'observed (FWHM={dv_obs:.2f} km/s)')
    ax_spec.plot(v, model_spec_for(low), color='tab:orange', lw=1.4, ls='--', label='low-density branch')
    ax_spec.plot(v, model_spec_for(high), color='tab:blue', lw=1.4, ls='--', label='high-density branch')
    ax_spec.set_xlabel('v [km/s]', fontsize=9)
    ax_spec.set_ylabel(r'$T_B$ [K]', fontsize=9)
    ax_spec.set_title(f'{REGION} {POS}: (1,1) spectrum, both branches', fontsize=10)
    ax_spec.legend(fontsize=7)
    ax_spec.grid(alpha=0.25)

    obs_spec_norm = gaussian_spectrum(v, [x / T_B_obs for x in obs_amps], dv_obs)
    ax_spec_n = fig.add_subplot(gs[0, 3:6])
    ax_spec_n.plot(v, obs_spec_norm, color='k', lw=1.6, label='observed / $T_{B,obs}$')
    for branch, color, lbl in [(low, 'tab:orange', 'low branch'), (high, 'tab:blue', 'high branch')]:
        amps = [branch['model_ratios']['R_10'], branch['model_ratios']['R_12'], 1.0,
                branch['model_ratios']['R_21'], branch['model_ratios']['R_01']]
        ax_spec_n.plot(v, gaussian_spectrum(v, amps, DV_CLUMP_KMS), color=color, lw=1.4, ls='--',
                       label=f'{lbl} / $T_{{B,theor}}$')
    ax_spec_n.axhline(1.0, color='gray', lw=0.6, ls=':')
    ax_spec_n.set_xlabel('v [km/s]', fontsize=9)
    ax_spec_n.set_ylabel(r'$T_B\,/\,T_B(\mathrm{main})$', fontsize=9)
    ax_spec_n.set_title('main-line-normalized (ratio shape)', fontsize=10)
    ax_spec_n.legend(fontsize=7)
    ax_spec_n.grid(alpha=0.25)

    ax_txt = fig.add_subplot(gs[0, 6:9])
    ax_txt.axis('off')
    lines = [
        "LOW-density branch (global best chi2):",
        f"  log n = {low['log_n']:.3f}   T = {low['T']:.1f} K",
        f"  chi2 = {low['chi2']:.3f}   eta_f = {low['eta_f']:.3f}",
        f"  consistent: {'YES' if low['consistent'] else 'NO'}",
        "",
        "HIGH-density branch (local min., log n >= 6):",
        f"  log n = {high['log_n']:.3f}   T = {high['T']:.1f} K",
        f"  chi2 = {high['chi2']:.3f}  (Delta = {high['chi2']-low['chi2']:+.3f})",
        f"  eta_f = {high['eta_f']:.3f}",
        f"  K pred = {high['K_pred']:.1f}   K req = {high['K_req']:.1f}",
        f"  consistent: {'YES' if high['consistent'] else 'NO'}",
        "",
        f"(Table 1a: log n={a1a['log_nH2']:.3f}, T={a1a['T_k']:.2f} K)" if a1a else "",
    ]
    ax_txt.text(0.02, 0.95, "\n".join(lines), va='top', ha='left', fontsize=9.5,
                transform=ax_txt.transAxes, family='monospace')

    ax21 = fig.add_subplot(gs[0, 9:12])
    labels = ['low branch\n(best chi2)', 'high branch\n(consistent)']
    vals = [low['R21_11'], high['R21_11']]
    colors = ['tab:orange', 'tab:blue']
    if t321 is not None:
        labels += ['Stutzki\ntheory', 'Stutzki\nobserved']
        vals += [t321['ratio_theor'], t321['ratio_obs']]
        colors += ['gray', 'black']
    ax21.bar(labels, vals, color=colors, alpha=0.85)
    ax21.set_ylabel(r'$T_B(2,1)/T_B(1,1)$', fontsize=9)
    ax21.set_title('(2,1)/(1,1): both branches vs. theory/obs.', fontsize=10)
    for i, val in enumerate(vals):
        ax21.text(i, val, f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    chi2_slice_plot(fig.add_subplot(gs[1, 0:4]), log_n_ax, T_ax, chi2_3d, 2,
                    r'log $n_{H_2}$', r'$T_k$ [K]', (low['log_n'], low['T']), chi2_best,
                    (a1a['log_nH2'], a1a['T_k']) if a1a else None,
                    title=r'$T_k$ vs log($n_{H_2}$), profile min over log(N/dv)')
    chi2_slice_plot(fig.add_subplot(gs[1, 4:8]), T_ax, ndv_ax, np.moveaxis(chi2_3d, 0, -1), 2,
                    r'$T_k$ [K]', r'log($N_{NH_3}/\Delta v$)', (low['T'], low['ndv']), chi2_best,
                    None, title=r'log(N/dv) vs $T_k$, profile min over log($n_{H_2}$)')
    chi2_slice_plot(fig.add_subplot(gs[1, 8:12]), log_n_ax, ndv_ax, chi2_3d, 1,
                    r'log $n_{H_2}$', r'log($N_{NH_3}/\Delta v$)', (low['log_n'], low['ndv']), chi2_best,
                    None, title=r'log(N/dv) vs log($n_{H_2}$), profile min over $T_k$')

    out_png = os.path.join(outdir, 'S106_0_0_branch_comparison.png')
    plt.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == '__main__':
    main()
