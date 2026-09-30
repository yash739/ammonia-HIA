"""
Pure-numpy scoring for ratio-vector comparison. No Magritte / nh3_NLTE_sphere
import at module scope, deliberately -- this file must be importable and
unit-testable without a Magritte-capable environment.

Replaces cosine_similarity (ratio_screen.py) as the ranking metric for
invert_ratios.py. cosine similarity failed twice this session on these exact
kinds of ratio vectors:
  1. Ladder-ratio screening (W33 A/B): (2,2)/(1,1) dominates the vector norm,
     so candidates scored 0.98+ similarity despite (2,1)/(1,1) being SIX
     ORDERS OF MAGNITUDE off (predicted ~1e-7 vs observed ~0.1).
  2. (1,1)-internal hyperfine ratios: the four ratios move roughly
     proportionally with density (R_10_MAIN alone spans 0.225-0.769 across
     the W33_A grid), so cosine similarity (scale-invariant by construction)
     was compressed into 0.96-0.998 across the whole grid -- nearly blind to
     the one thing that actually varies.

weighted_chi2 fixes both: each residual is normalized by ITS OWN sigma, not
by a shared vector norm, so no single component can hide another's error, and
because it's not scale-invariant, it responds to the ratio *magnitude*
changing with density, not just its direction.
"""

import numpy as np


def _prepare(model_vec, obs_vec, mask=None):
    model = np.asarray(model_vec, dtype=float)
    obs = np.asarray(obs_vec, dtype=float)
    if model.shape != obs.shape:
        raise ValueError(f"model_vec and obs_vec shape mismatch: {model.shape} vs {obs.shape}")
    keep = ~np.isnan(model) & ~np.isnan(obs)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    return model, obs, keep


def weighted_chi2(model_vec, obs_vec, sigma_vec=None, mask=None, sigma_floor_frac=0.1):
    """Uncertainty-weighted mean chi-square: mean(((model_i - obs_i) / sigma_i)^2)
    over components that survive masking. Lower is better; NaN if nothing survives.

    sigma_vec: 1-sigma uncertainty on each OBSERVED component (not simulation
    noise on the model side -- that belongs in a separate convergence_ok flag,
    not folded into this number, so a badly-converged model point can't "pass"
    just because its own noise happens to be large).

    If sigma_vec is None, uses a floor of sigma_floor_frac * |obs_i| (default
    10%) as a stand-in relative uncertainty. This is an assumption, not a
    measurement -- report it as such wherever this score is surfaced without
    an explicit sigma_vec.

    NaN in either vector, or False in mask, drops that component from BOTH
    vectors before scoring (matches ratio_screen.cosine_similarity's existing
    masking convention) -- a missing/undetected component must not silently
    count as either a perfect or an infinite-error match.
    """
    model, obs, keep = _prepare(model_vec, obs_vec, mask)
    if sigma_vec is None:
        sigma = np.maximum(sigma_floor_frac * np.abs(obs), 1e-12)
    else:
        sigma = np.asarray(sigma_vec, dtype=float)
        keep = keep & ~np.isnan(sigma) & (sigma > 0)
    if not keep.any():
        return np.nan
    resid = (model[keep] - obs[keep]) / sigma[keep]
    return float(np.mean(resid ** 2))


def mean_relative_error(model_vec, obs_vec, mask=None):
    """mean(|model_i - obs_i| / |obs_i|) over unmasked components. Simpler
    companion metric to weighted_chi2 -- this is what was hand-used (without
    a sigma) to rank the Section 07 ratio-screening results before this
    module existed; kept as a plain, sigma-free number for reporting
    alongside the chi2 score, not a replacement for it."""
    model, obs, keep = _prepare(model_vec, obs_vec, mask)
    keep = keep & (obs != 0)
    if not keep.any():
        return np.nan
    return float(np.mean(np.abs(model[keep] - obs[keep]) / np.abs(obs[keep])))


def rank_candidates(rows, score_fn, score_key='score', reverse=False):
    """Attach score_fn(row) to each row dict under score_key, sort (ascending
    by default -- lower score = better fit), return the sorted list.

    score_fn receives one row dict and must return a float (NaN allowed --
    NaN-scored rows sort last regardless of `reverse`).
    """
    scored = []
    for row in rows:
        s = score_fn(row)
        row = dict(row)
        row[score_key] = s
        scored.append(row)

    def sort_key(row):
        s = row[score_key]
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return (1, 0.0)
        return (0, -s if reverse else s)

    scored.sort(key=sort_key)
    return scored
