"""Grid posterior for ratio retrievals: confidence intervals without MCMC.

The retrievals already evaluate the model at every point of a dense grid
(the 3D LUT's interpolated query grid, or the 1D model's fine grid), so the
likelihood is known everywhere and the posterior is a normalised weighted sum
over the grid. For three parameters this is exact up to the grid spacing,
has no convergence or burn-in issues, and represents the two density branches
faithfully (an MCMC sampler tends to get stuck in one of them).

Likelihood. Independent Gaussian errors on the observed ratios, with each
ratio's variance the observational error plus a fractional model error in
quadrature:
    sigma_i^2(theta) = sigma_obs,i^2 + (f_model * r_i,model(theta))^2
    -2 ln L = sum_i [ (r_i,model - r_i,obs)^2 / sigma_i^2 + ln(2 pi sigma_i^2) ]
The model term stands for interpolation error (hold-one-out medians 1.4-2.1%)
and model inadequacy (one homogeneous clump, mesh systematics), neither of
which the observational errors include; without it posteriors from small
published errors are far too narrow. Off-diagonal ratio correlations through
the shared main line were measured at <= 0.13, so errors are taken as
independent. Ratios are near-Gaussian for satellite S/N >~ 5.

Prior. Uniform in the grid coordinates (log n, T_k, log N/dv), i.e.
log-uniform in density and column and uniform in temperature, bounded by the
grid. Probability piling up at a grid edge means the interval is set by the
grid, not the data; `edge_mass` reports it.

Branches. Ratio retrievals have a high- and a low-density solution, and a
median taken over both can land between them where nothing fits. Summaries
are therefore given overall and per branch (split at log n = BRANCH_SPLIT).
"""
import numpy as np

BRANCH_SPLIT = 6.0          # log n_H2 separating the low- and high-density branches
DEFAULT_F_MODEL = 0.05      # fractional model error added in quadrature
EDGE_FRAC = 0.02            # outermost fraction of each axis range counted as "edge"


def log_likelihood(model_mat, obs_vec, sigma_obs, f_model=DEFAULT_F_MODEL):
    """model_mat: (N, k) model ratios; obs_vec, sigma_obs: (k,). NaN observed
    ratios are skipped; rows with any NaN model ratio get -inf."""
    use = ~np.isnan(obs_vec) & ~np.isnan(sigma_obs)
    m = model_mat[:, use]
    var = sigma_obs[use] ** 2 + (f_model * m) ** 2
    ll = -0.5 * np.sum((m - obs_vec[use]) ** 2 / var + np.log(2 * np.pi * var), axis=1)
    return np.where(np.any(~np.isfinite(m), axis=1), -np.inf, ll)


def weights_from_loglike(ll):
    w = np.zeros_like(ll)
    ok = np.isfinite(ll)
    if ok.any():
        w[ok] = np.exp(ll[ok] - ll[ok].max())
        w /= w.sum()
    return w


def weighted_quantiles(x, w, qs=(0.16, 0.5, 0.84)):
    keep = (w > 0) & np.isfinite(x)
    if not keep.any():
        return [np.nan] * len(qs)
    order = np.argsort(x[keep])
    xs, cw = x[keep][order], np.cumsum(w[keep][order])
    cw /= cw[-1]
    return [float(np.interp(q, cw, xs)) for q in qs]


def summarize(pts, w, extra=None, names=('log_n_H2', 'T_k', 'log_N_dv'), prefix='post_'):
    """Posterior summary: 16/50/84% marginals per parameter (and per extra
    quantity, e.g. the predicted (2,1)/(1,1)), overall and per density
    branch, the branch probabilities, and edge masses.

    pts: (N, 3) grid points; w: (N,) normalised weights; extra: dict
    name -> (N,) array of derived quantities to propagate."""
    extra = extra or {}
    out = {}
    hi = pts[:, 0] >= BRANCH_SPLIT
    p_hi = float(w[hi].sum())
    out[prefix + 'P_high_branch'] = p_hi
    quantities = {n: pts[:, i] for i, n in enumerate(names)}
    quantities.update(extra)
    for label, mask in (('', np.ones(len(w), bool)), ('hi_', hi), ('lo_', ~hi)):
        wm = np.where(mask, w, 0.0)
        tot = wm.sum()
        for n, x in quantities.items():
            q = weighted_quantiles(x, wm / tot) if tot > 0 else [np.nan] * 3
            out[f'{prefix}{label}{n}_p16'], out[f'{prefix}{label}{n}_p50'], out[f'{prefix}{label}{n}_p84'] = q
    for i, n in enumerate(names):
        lo, hi_ = pts[:, i].min(), pts[:, i].max()
        span = hi_ - lo
        out[f'{prefix}edge_mass_{n}_lo'] = float(w[pts[:, i] <= lo + EDGE_FRAC * span].sum())
        out[f'{prefix}edge_mass_{n}_hi'] = float(w[pts[:, i] >= hi_ - EDGE_FRAC * span].sum())
    return out


def predictive_z(pred_samples_q, obs, obs_err):
    """Rough tension between a posterior-predictive distribution (16/50/84%)
    and an observation with error: (median - obs) / sqrt(half-width^2 + err^2)."""
    p16, p50, p84 = pred_samples_q
    half = 0.5 * (p84 - p16)
    return (p50 - obs) / np.sqrt(half ** 2 + obs_err ** 2)


def consistency_prior(pts, T_B_theor, T_B_obs, dv_obs_kms, distance_pc, dv_clump_kms):
    """Stutzki & Winnewisser's (1985) two physical-consistency checks as a
    hard prior on every grid point (vectorised nh3hia.stutzki.physics):
    beam filling factor 0 < eta_f = T_B,obs / T_B,theor <= 1, and enough
    Jeans-mass clumps per beam, K >= dv_obs / dv_clump. Returns a boolean
    mask. This is the probabilistic version of the post-fit branch choice S85
    made by hand; it is a prior, not a measurement (see Paper I's noise test).
    """
    log_n = pts[:, 0]
    T = pts[:, 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        ef = T_B_obs / T_B_theor
        n7 = 10.0 ** log_n / 1e7
        M_c_max = 347.4 * n7 * (distance_pc / 500.0) ** 3 * np.clip(ef, 0, None) ** 1.5
        M_J = 0.103 * np.sqrt((T / 10.0) ** 3 / n7)
        K = (M_c_max / M_J) ** (2.0 / 3.0) * (dv_obs_kms / dv_clump_kms)
    return (ef > 0) & (ef <= 1.0) & (K >= dv_obs_kms / dv_clump_kms)


def posterior_block(pts, model_mat, obs_vec, sigma_obs, extra=None,
                    f_models=((DEFAULT_F_MODEL, 'post_'), (0.0, 'post0_'), (0.10, 'post10_')),
                    prior_mask=None):
    """Summaries at the default model-error floor plus 0% / 10% sensitivity
    runs, as one flat dict ready for a summary-CSV row. prior_mask: optional
    boolean array (e.g. consistency_prior) multiplied into the posterior."""
    out = {}
    for f_model, prefix in f_models:
        ll = log_likelihood(model_mat, obs_vec, sigma_obs, f_model)
        if prior_mask is not None:
            ll = np.where(prior_mask, ll, -np.inf)
        w = weights_from_loglike(ll)
        out[prefix + 'prior_mass_ok'] = float(w.sum() > 0)
        out.update(summarize(pts, w, extra=extra, prefix=prefix))
    return out
