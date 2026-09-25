"""LUT interpolation for continuous chi^2 retrieval, per Phase 3 of the
master plan: interpolate log-amplitudes on (log_n_H2, T_cloud, log_N_dv),
then form ratios from the interpolated amplitudes -- not ratios directly
(amplitudes are smoother; a ratio steepens wherever its denominator gets
small).

This is the same method already validated by the hold-one-out check
(median errors 0.7-0.9% on the combined coarse+offset grid; see
interpolation_validity.tex): scipy LinearNDInterpolator on a Delaunay
triangulation of the irregular grid point cloud. A plain
scipy.interpolate.RegularGridInterpolator does NOT apply here -- the
combined grid is not a full cross product (the offset grid only refines
the density axis at 4 of 7 temperatures and adds T=27K only at the full
density axis), so the point cloud is irregular by construction.

Retrieval against raw LUT rows (nearest-grid-point chi^2) silently pins
solutions to whichever grid node happens to be closest, which is
indistinguishable from a genuine off-grid or boundary-pinned solution
without this interpolator. Use `chi2_retrieve` for any real retrieval;
reserve row-wise chi^2 for cases that specifically want the raw grid
(e.g. re-deriving the hold-one-out check itself).
"""

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

DEFAULT_LUT = "/home/yasho379/magritte_rebuilt/production/output_lut_gold/results/lut_dv0.30.csv"
# Excludes the low-density/high-column corner where the implied clump radius
# exceeds 0.5 pc (>>Stutzki's ~0.01 pc clumps): that corner is subcritically
# thermalized against strong radiative pumping and shows non-convergent,
# unphysical ratios (see important_notes/magritte-quirks-2026-09-16.md, 2.1).
PC_CM = 3.0857e18
DEFAULT_MAX_RADIUS_PC = None  # off by default -- pass max_radius_pc=0.5 to re-enable.
# The corner it would exclude (log_n<=4.5, log_N_dv>=15.35) is genuinely
# subcritically-thermalized and often unconverged, but the flag caller sets
# is what actually gates data quality (convergence_ok == final_convergence>=90%,
# see build_lut.CONV_THRESHOLD); the mask was an extra precaution on top of
# that, not a substitute for it.

AMPS = ('A_01', 'A_10', 'A_MAIN', 'A_21', 'A_12', 'A_MAIN_22', 'A_MAIN_21')
RATIO_KEYS = ('R_01_MAIN', 'R_10_MAIN', 'R_21_MAIN', 'R_12_MAIN', 'R_22_MAIN')
_RATIO_NUM = {'R_01_MAIN': 'A_01', 'R_10_MAIN': 'A_10', 'R_21_MAIN': 'A_21',
              'R_12_MAIN': 'A_12', 'R_22_MAIN': 'A_MAIN_22'}
# (2,1)/(1,1) ratio -- NOT in RATIO_KEYS/_RATIO_NUM (and so never enters
# chi2_retrieve unless explicitly asked for): the whole point of storing it
# is to predict it at a point retrieved from the other 5 ratios alone, as a
# genuine held-out test (Stutzki's own Table 3 comparison), not to fit it.
RATIO_21 = 'R_21_11'


class LutInterpolator:
    """Holds one LinearNDInterpolator per amplitude, built once, plus a
    dense query grid spanning the LUT's own axis ranges -- points outside
    the point cloud's convex hull evaluate to NaN automatically (no
    extrapolation), so a genuine boundary-pinned retrieval still pins,
    while an interior one now resolves between grid nodes.
    """

    def __init__(self, lut_path=DEFAULT_LUT, max_radius_pc=DEFAULT_MAX_RADIUS_PC,
                 exclude_keys=None):
        """exclude_keys: optional iterable of (log_n_H2, T_cloud, log_N_dv)
        tuples to drop from the point cloud before interpolating (matched to
        1e-3) -- for hold-out tests where a truth node must not be in the
        interpolant it is being recovered from."""
        df = pd.read_csv(lut_path)
        df = df[df['Status'] == 'SUCCESS']
        df = df[df['convergence_ok']]
        if max_radius_pc:
            df = df[df['radius_sphere'] <= max_radius_pc * PC_CM]
        if exclude_keys:
            drop = np.zeros(len(df), dtype=bool)
            for (ln, tt, ld) in exclude_keys:
                drop |= (np.isclose(df['log_n_H2'], ln, atol=1e-3) &
                         np.isclose(df['T_cloud'], tt, atol=1e-3) &
                         np.isclose(df['log_N_dv'], ld, atol=1e-3))
            df = df[~drop]
        df = df.reset_index(drop=True)
        self.df = df
        self.X = df[['log_n_H2', 'T_cloud', 'log_N_dv']].values
        self._interps = {
            a: LinearNDInterpolator(self.X, np.log10(df[a].values.clip(min=1e-12)))
            for a in AMPS
        }
        self.axis_lo = self.X.min(axis=0)
        self.axis_hi = self.X.max(axis=0)

    def query_grid(self, n_log_n=200, n_T=73, n_log_Ndv=90):
        """Dense (log_n_H2, T_cloud, log_N_dv) query grid over the LUT's
        own axis bounding box. Points outside the point cloud's convex hull
        interpolate to NaN and are dropped automatically downstream.
        """
        log_n = np.linspace(self.axis_lo[0], self.axis_hi[0], n_log_n)
        T = np.linspace(self.axis_lo[1], self.axis_hi[1], n_T)
        log_Ndv = np.linspace(self.axis_lo[2], self.axis_hi[2], n_log_Ndv)
        LN, TT, LD = np.meshgrid(log_n, T, log_Ndv, indexing='ij')
        pts = np.column_stack([LN.ravel(), TT.ravel(), LD.ravel()])
        return pts

    def predict_ratios(self, pts):
        """pts: (N,3) array of (log_n_H2, T_cloud, log_N_dv).
        Returns dict of ratio_key -> (N,) array (NaN outside the hull),
        plus 'A_MAIN' -> (N,) interpolated main-line amplitude (needed for
        the physically_consistent absolute-brightness filter).
        """
        logamp = {a: self._interps[a](pts) for a in AMPS}
        amp = {a: 10.0 ** v for a, v in logamp.items()}
        out = {k: amp[_RATIO_NUM[k]] / amp['A_MAIN'] for k in RATIO_KEYS}
        out['A_MAIN'] = amp['A_MAIN']
        out[RATIO_21] = amp['A_MAIN_21'] / amp['A_MAIN']
        return out


def chi2_retrieve(interp, pts, ratio_grid, obs_vec, sigma_vec, ratio_keys=RATIO_KEYS):
    """Score every query point against one observed ratio vector via the
    same sigma-weighted chi^2 as scoring.weighted_chi2, vectorized over all
    points at once. Returns (best_idx, chi2_at_best, chi2_array) -- NaN rows
    (outside the hull, or where an observed ratio itself is NaN) score +inf.
    """
    model_mat = np.column_stack([ratio_grid[k] for k in ratio_keys])
    valid_obs = ~np.isnan(obs_vec)
    diff = (model_mat[:, valid_obs] - obs_vec[valid_obs]) / sigma_vec[valid_obs]
    chi2 = np.sum(diff ** 2, axis=1) / valid_obs.sum()
    chi2 = np.where(np.any(np.isnan(model_mat[:, valid_obs]), axis=1), np.inf, chi2)
    best_idx = np.argmin(chi2)
    return best_idx, chi2[best_idx], chi2
