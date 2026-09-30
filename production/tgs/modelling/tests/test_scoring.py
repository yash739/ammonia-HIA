"""Unit tests for scoring.py -- pure numpy, no Magritte compute needed.
Run with: pytest tests/test_scoring.py -v
"""

import numpy as np
import pytest

from nh3hia.scoring import weighted_chi2, mean_relative_error, rank_candidates


def test_weighted_chi2_perfect_match_is_zero():
    v = [0.4, 0.5, 0.6, 0.7]
    assert weighted_chi2(v, v) == pytest.approx(0.0)


def test_weighted_chi2_catches_dominant_component_hiding_a_bad_one():
    """The actual cosine-similarity failure from Section 07: one large,
    well-matched component (2,2)-like, and one component six orders of
    magnitude off (2,1)-like). Cosine similarity scored this 0.98+; chi2 must
    not."""
    obs = [0.70, 0.0927]       # e.g. (2,2)/(1,1), (2,1)/(1,1)
    model_good = [0.699, 0.090]           # both components close
    model_bad = [0.699, 2.4e-7]           # (2,2) close, (2,1) six orders off

    chi2_good = weighted_chi2(model_good, obs)
    chi2_bad = weighted_chi2(model_bad, obs)
    assert chi2_good < 1.0
    assert chi2_bad > 100 * chi2_good, "chi2 must penalize the badly-wrong component heavily"

    # confirm this is exactly the case cosine similarity gets wrong
    def cosine(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cosine(model_bad, obs) > 0.9, "sanity check: cosine similarity really does miss this"


def test_weighted_chi2_scale_sensitivity():
    """The second cosine-similarity failure: ratios that move roughly
    proportionally (same direction, different magnitude) must NOT score as a
    near-perfect match, unlike cosine similarity."""
    obs = [0.40, 0.50, 0.45, 0.42]
    model_scaled = [0.30, 0.375, 0.3375, 0.315]  # exactly 0.75x obs -- same direction

    def cosine(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert cosine(model_scaled, obs) == pytest.approx(1.0, abs=1e-9)
    assert weighted_chi2(model_scaled, obs) > 1.0, "chi2 must penalize a uniformly-scaled mismatch"


def test_weighted_chi2_explicit_sigma():
    obs = [1.0, 1.0]
    model = [1.1, 1.1]
    sigma = [0.1, 0.1]
    # residual = 0.1/0.1 = 1 sigma on each component -> mean(1^2, 1^2) = 1.0
    assert weighted_chi2(model, obs, sigma_vec=sigma) == pytest.approx(1.0)


def test_weighted_chi2_nan_masking_drops_from_both():
    obs = [0.5, np.nan, 0.3]
    model = [0.5, 0.9, 0.3]
    # the NaN'd component must be dropped, not treated as a 0-error or inf-error match
    assert weighted_chi2(model, obs) == pytest.approx(0.0)


def test_weighted_chi2_explicit_mask():
    obs = [0.5, 0.9, 0.3]
    model = [0.5, 0.1, 0.3]  # component 1 is badly wrong
    mask = [True, False, True]
    assert weighted_chi2(model, obs, mask=mask) == pytest.approx(0.0)


def test_weighted_chi2_all_masked_returns_nan():
    assert np.isnan(weighted_chi2([1.0], [1.0], mask=[False]))


def test_mean_relative_error_hand_computed():
    obs = [1.0, 2.0, 4.0]
    model = [1.1, 1.8, 4.4]
    # |0.1|/1 + |0.2|/2 + |0.4|/4 = 0.1 + 0.1 + 0.1 -> mean = 0.1
    assert mean_relative_error(model, obs) == pytest.approx(0.1)


def test_rank_candidates_sorts_ascending_and_handles_nan():
    rows = [{'id': 'a', 'v': [10.0]}, {'id': 'b', 'v': [1.0]}, {'id': 'c', 'v': [np.nan]}]

    def score_fn(row):
        return row['v'][0]

    ranked = rank_candidates(rows, score_fn)
    assert [r['id'] for r in ranked] == ['b', 'a', 'c']
    assert ranked[0]['score'] == pytest.approx(1.0)
    assert np.isnan(ranked[-1]['score'])


def test_rank_candidates_does_not_mutate_input_rows():
    rows = [{'id': 'a', 'v': 1.0}]
    rank_candidates(rows, lambda r: r['v'])
    assert 'score' not in rows[0]
