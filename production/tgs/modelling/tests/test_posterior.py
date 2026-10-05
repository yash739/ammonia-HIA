"""Tests for the grid posterior (nh3hia.lut.posterior)."""
import numpy as np
import pytest

from nh3hia.lut import posterior as P
from nh3hia.stutzki import physics as ph


def test_gaussian_posterior_recovers_mean_and_width():
    """One parameter, one 'ratio' equal to it: the 68% interval must be +-sigma."""
    x = np.linspace(-5, 5, 20001)
    pts = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    ll = P.log_likelihood(x[:, None], np.array([0.7]), np.array([0.5]), f_model=0.0)
    q = P.weighted_quantiles(x, P.weights_from_loglike(ll))
    assert q == pytest.approx([0.7 - 0.5, 0.7, 0.7 + 0.5], abs=2e-3)


def test_uses_raw_not_reduced_chi2():
    """Two identical observations halve the variance (width / sqrt 2)."""
    x = np.linspace(-5, 5, 20001)
    ll = P.log_likelihood(np.column_stack([x, x]), np.array([0.0, 0.0]),
                          np.array([1.0, 1.0]), f_model=0.0)
    q = P.weighted_quantiles(x, P.weights_from_loglike(ll))
    assert (q[2] - q[0]) / 2 == pytest.approx(1 / np.sqrt(2), rel=2e-3)


def test_branch_probabilities():
    x = np.linspace(3.5, 8.5, 5001)
    w = np.exp(-0.5 * ((x - 4.5) / 0.1) ** 2) + 3 * np.exp(-0.5 * ((x - 7.2) / 0.1) ** 2)
    w /= w.sum()
    s = P.summarize(np.column_stack([x, np.zeros_like(x), np.zeros_like(x)]), w)
    assert s['post_P_high_branch'] == pytest.approx(0.75, abs=1e-3)
    assert s['post_hi_log_n_H2_p50'] == pytest.approx(7.2, abs=0.01)
    assert s['post_lo_log_n_H2_p50'] == pytest.approx(4.5, abs=0.01)


def test_consistency_prior_matches_scalar_filter():
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(3.5, 8.5, 300), rng.uniform(9, 48, 300), np.zeros(300)])
    T_B_theor = rng.uniform(0.5, 25, 300)
    mask = P.consistency_prior(pts, T_B_theor, 4.71, 1.47, 600.0, 0.3)
    ref = [ph.physically_consistent(T, 10 ** n, 4.71, tb, 1.47, 0.3, 600.0)[0]
           for (n, T, _), tb in zip(pts, T_B_theor)]
    assert np.array_equal(mask, np.array(ref))
