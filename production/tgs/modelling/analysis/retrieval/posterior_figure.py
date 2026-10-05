"""Marginal posteriors for the five positions with a (2,1) measurement
(OMC S3, OMC S4, S106 (0,0), W33 A, W33 B), for the three model/rate
combinations: 3D Magritte + Loreau rates (gold LUT, interpolated query grid),
1D escape probability + Loreau rates, and 1D + S85's own rates (fine grids).

Columns: log n, T_k, log N/dv, and the posterior-predictive (2,1)/(1,1)
against the observation. Solid: likelihood with the default 5% model-error
floor; dotted: the same with S85's physical-consistency checks as a prior.
Curves are peak-normalised marginals (nh3hia/lut/posterior.py).

Note: the S85-rate grids have S85's n' on the density axis (n_H2 = n'/1.75).
The 1D grids used are the log n <= 8.5 extensions, matching the 3D grid's range.

Usage: python3 -m analysis.retrieval.posterior_figure [--out PATH]
"""
import argparse
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nh3hia import paths
from nh3hia.lut import posterior as post
from nh3hia.lut.interpolator import LutInterpolator
from nh3hia.escape1d.grid import outdir_for, DV_CLUMP_KMS
from nh3hia.stutzki import params as sp
from nh3hia.w33 import params as wp
from nh3hia.w33 import observed as wr
from analysis.retrieval.magritte_stutzki1984 import QGRID
from analysis.retrieval.escape1d_stutzki1984 import load_grid

KEYS_3D = ['R_01_MAIN', 'R_10_MAIN', 'R_21_MAIN', 'R_12_MAIN', 'R_22_MAIN']
KEYS_1D = ['R_01', 'R_10', 'R_21', 'R_12', 'R_2211']
POSITIONS = [('S106', '0,0'), ('OMC', 'S3'), ('OMC', 'S4'), ('W33', 'W33_A'), ('W33', 'W33_B')]
MODELS = [('3D, Loreau', 'tab:blue'), ('1D, Loreau', 'tab:orange'), ("1D, S85 rates (n')", 'tab:green')]


def observation(region, pos):
    """(obs_vec, sigma_vec, T_B_obs, dv_obs, dist_pc, R21_obs, R21_err) in KEYS_3D order."""
    if region == 'W33':
        o, e, _ = wr.observed_ratio_vector(pos)
        vec = lambda d: np.array([np.nan if d.get(k) is None else d[k] for k in KEYS_3D])
        return (vec(o), vec(e), wp.LINES[pos]['1,1']['T_mb'], wp.LINES[pos]['1,1']['dv'],
                wp.DISTANCE_PC, o['R_21_11'], e['R_21_11'])
    t = sp.TABLE_2_1984[(region, pos)]
    obs = np.array([t['R_01'], t['R_10'], t['R_21'], t['R_12'], t.get('R_22_MAIN', np.nan)])
    err = np.array([t['R_01_err'], t['R_10_err'], t['R_21_err'], t['R_12_err'], t.get('R_22_MAIN_err', np.nan)])
    t3 = sp.TABLE_3_21[(region, pos)]
    return (obs, err, t['T_B_11'], t['dv'], sp.SOURCE_DISTANCE_PC.get(region, 500.0),
            t3['ratio_obs'], t3['ratio_obs_err'])


def model_grids():
    """name -> (pts, model_mat, A_MAIN, R21) for the three model sets."""
    interp = LutInterpolator()
    pts = interp.query_grid(**QGRID)
    rg = interp.predict_ratios(pts)
    out = {MODELS[0][0]: (pts, np.column_stack([rg[k] for k in KEYS_3D]), rg['A_MAIN'], rg['R_21_11'])}
    for (name, _), rates in zip(MODELS[1:], ('loreau', 's85')):
        df = load_grid(os.path.join(outdir_for(rates), 'grid_fine_ext8.5.csv'))   # log n up to 8.5, as the 3D grid
        out[name] = (df[['log_n_H2', 'T_cloud', 'log_N_dv']].values,
                     np.column_stack([df[k].values for k in KEYS_1D]),
                     df['T_B_main'].values, df['R_21_11'].values)
    return out


def marginal(x, w):
    vals, inv = np.unique(np.round(x, 6), return_inverse=True)
    m = np.bincount(inv, weights=w, minlength=len(vals))
    return vals, m / m.max() if m.max() > 0 else m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(paths.LUT_GOLD, 'results', 'posteriors_key_positions.png'))
    a = ap.parse_args()
    grids = model_grids()

    fig, axes = plt.subplots(len(POSITIONS), 4, figsize=(17, 3.0 * len(POSITIONS)))
    r21_bins = np.linspace(0, 0.30, 121)
    for i, (region, pos) in enumerate(POSITIONS):
        obs, err, T_B_obs, dv_obs, dist, r21_obs, r21_err = observation(region, pos)
        for name, color in MODELS:
            pts, mm, A_main, r21 = grids[name]
            ll = post.log_likelihood(mm, obs, err, post.DEFAULT_F_MODEL)
            prior = post.consistency_prior(pts, A_main, T_B_obs, dv_obs, dist, DV_CLUMP_KMS)
            for ls, ll_use in (('-', ll), (':', np.where(prior, ll, -np.inf))):
                w = post.weights_from_loglike(ll_use)
                if w.sum() == 0:
                    continue
                for j in range(3):
                    x, m = marginal(pts[:, j], w)
                    axes[i, j].plot(x, m, ls=ls, color=color, lw=1.6,
                                    label=name if (ls == '-' and i == 0 and j == 0) else None)
                h, e = np.histogram(np.nan_to_num(r21, nan=-1), bins=r21_bins, weights=w)
                c = 0.5 * (e[1:] + e[:-1])
                h = gaussian_filter1d(h, 1.5)   # the grid's discrete R21 values comb an unsmoothed histogram
                axes[i, 3].plot(c, h / h.max() if h.max() > 0 else h, ls=ls, color=color, lw=1.6)
        axes[i, 3].axvspan(r21_obs - r21_err, r21_obs + r21_err, color='0.6', alpha=0.5)
        axes[i, 3].axvline(r21_obs, color='k', lw=1.2)
        label = f'{region} {pos}' if region != 'W33' else pos.replace('_', ' ')
        axes[i, 0].set_ylabel(f'{label}\nposterior (peak = 1)', fontsize=10)
        for j in range(4):
            axes[i, j].set_ylim(0, 1.08)
            axes[i, j].grid(alpha=0.25)

    for j, (lab, lim) in enumerate([(r'log $n_{\rm H_2}$ [cm$^{-3}$]', (3.5, 8.5)), (r'$T_k$ [K]', (9, 48)),
                                    (r'log($N_{\rm NH_3}/\Delta v$)', (14.0, 16.25)),
                                    (r'predicted $T_B(2,1)/T_B(1,1)$', (0, 0.30))]):
        axes[-1, j].set_xlabel(lab, fontsize=10)
        for i in range(len(POSITIONS)):
            axes[i, j].set_xlim(*lim)
    handles = [plt.Line2D([], [], color=c, lw=1.6, label=n) for n, c in MODELS]
    handles += [plt.Line2D([], [], color='k', lw=1.6, ls='-', label='5% model error'),
                plt.Line2D([], [], color='k', lw=1.6, ls=':', label='+ S85 consistency prior'),
                plt.Rectangle((0, 0), 1, 1, color='0.6', alpha=0.5, label='observed (2,1)/(1,1) $\\pm1\\sigma$')]
    fig.legend(handles=handles, loc='upper center', ncol=6, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(a.out, dpi=130)
    print('saved', a.out)


if __name__ == '__main__':
    main()
