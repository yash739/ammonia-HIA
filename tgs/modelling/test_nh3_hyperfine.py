"""Unit tests for nh3_hyperfine.py -- pure numpy, no Magritte compute.
Run: pytest tgs/modelling/test_nh3_hyperfine.py -v
"""

import os
import numpy as np
import pytest

import nh3_hyperfine as hf

LAMDA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'p-nh3@loreau.dat.txt')


# --------------------------------------------------------------------------- #
# The literal tables must not drift from the LAMDA file they were derived from
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not os.path.exists(LAMDA), reason="LAMDA file not present")
def test_11_offsets_match_lamda():
    """Re-derive the (1,1) offsets from the collision file and compare."""
    rows = hf.derive_offsets_from_lamda(LAMDA, J=1, K=1)
    # 6 F1-resolved transitions; the two within 0.1 km/s of zero form the main group
    sat = [r for r in rows if abs(r[3]) > 0.1]
    main = [r for r in rows if abs(r[3]) <= 0.1]
    assert len(sat) == 4 and len(main) == 2
    derived = sorted([r[3] for r in sat] + [0.0])
    assert np.allclose(derived, hf.NH3_11_OFFSETS_KMS, atol=1e-3)


@pytest.mark.skipif(not os.path.exists(LAMDA), reason="LAMDA file not present")
def test_22_offsets_match_lamda():
    rows = hf.derive_offsets_from_lamda(LAMDA, J=2, K=2)
    sat = [r for r in rows if abs(r[3]) > 0.1]
    assert len(sat) == 4
    derived = sorted([r[3] for r in sat] + [0.0])
    assert np.allclose(derived, hf.NH3_22_OFFSETS_KMS, atol=1e-3)


@pytest.mark.skipif(not os.path.exists(LAMDA), reason="LAMDA file not present")
def test_outer_satellite_identities_are_not_swapped():
    """The bug this module exists to fix: F1 1->0 must be at NEGATIVE velocity
    (higher frequency than the main line) and F1 0->1 at POSITIVE velocity.
    Independently corroborated by Camarata et al. (2015), who name them
    left-outer and right-outer respectively."""
    rows = hf.derive_offsets_from_lamda(LAMDA, J=1, K=1)
    by_trans = {(r[0], r[1]): r for r in rows}
    f1_10, f1_01 = by_trans[(1, 0)], by_trans[(0, 1)]
    assert f1_10[3] < 0, "F1 1->0 must sit at negative (blueshifted) velocity"
    assert f1_01[3] > 0, "F1 0->1 must sit at positive (redshifted) velocity"
    assert f1_10[2] > f1_01[2], "F1 1->0 must be at the HIGHER frequency"
    # and the table must agree with that
    tbl = dict(zip(hf.NH3_11_KEYS_BY_VELOCITY, hf.NH3_11_OFFSETS_KMS))
    assert tbl['A_10'] < 0 < tbl['A_01']


def test_matches_stutzki_figure_1_scale():
    """Stutzki & Winnewisser (1985) Fig. 1 prints the (1,1) velocity scale as
    -19.50, -7.62, 0, +7.61, +19.49 km/s."""
    published = np.array([-19.50, -7.62, 0.0, 7.61, 19.49])
    assert np.allclose(hf.NH3_11_OFFSETS_KMS, published, atol=0.05)


def test_radio_convention_sign():
    """Lower frequency -> positive (redshifted) velocity."""
    f0 = hf.FREQ_HZ['1,1']
    assert hf.freq_to_radio_velocity_kms(f0 * 0.999, f0) > 0
    assert hf.freq_to_radio_velocity_kms(f0 * 1.001, f0) < 0
    assert hf.freq_to_radio_velocity_kms(f0, f0) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Component identification
# --------------------------------------------------------------------------- #

def test_identify_round_trips_on_exact_offsets():
    a = hf.identify_components(hf.NH3_11_OFFSETS_KMS)
    assert a == {'A_10': 0, 'A_12': 1, 'A_MAIN': 2, 'A_21': 3, 'A_01': 4}


def test_identify_handles_systemic_velocity_shift():
    """A source at v_LSR = 36 km/s (like W33) must still identify correctly
    once the centres are expressed relative to that systemic velocity."""
    shifted = hf.NH3_11_OFFSETS_KMS + 36.0
    a = hf.identify_components(shifted - 36.0)
    assert a['A_01'] == 4 and a['A_10'] == 0


def test_identify_is_order_independent():
    """Assignment must not depend on the order curve_fit happens to return."""
    perm = [3, 0, 4, 2, 1]
    centres = hf.NH3_11_OFFSETS_KMS[perm]
    a = hf.identify_components(centres)
    for k, orig_i in zip(hf.NH3_11_KEYS_BY_VELOCITY, range(5)):
        assert centres[a[k]] == pytest.approx(hf.NH3_11_OFFSETS_KMS[orig_i])


def test_identify_rejects_collapsed_fit():
    """Two components landing on the same peak must raise, not silently return
    a plausible-looking mapping."""
    centres = np.array([-19.5, 0.0, 0.0, 7.6, 19.5])
    with pytest.raises(ValueError):
        hf.identify_components(centres)


def test_identify_rejects_drifted_component():
    centres = np.array([-19.5, -7.6, 0.0, 7.6, 60.0])  # outer satellite ran away
    with pytest.raises(ValueError):
        hf.identify_components(centres)


def test_identify_rejects_wrong_count():
    with pytest.raises(ValueError):
        hf.identify_components([0.0, 7.6, -7.6])


# --------------------------------------------------------------------------- #
# Fit seeding / bounding -- the "slightly free" behaviour
# --------------------------------------------------------------------------- #

def test_bounds_are_seeded_at_true_offsets():
    c, lo, hi = hf.initial_centres_and_bounds()
    assert np.allclose(c, hf.NH3_11_OFFSETS_KMS)
    assert np.all(lo < c) and np.all(c < hi)


def test_bounds_do_not_overlap():
    """The whole point: adjacent windows must be disjoint so two components
    cannot swap or collapse."""
    _, lo, hi = hf.initial_centres_and_bounds()
    assert np.all(hi[:-1] < lo[1:])


def test_bounds_track_systemic_velocity():
    c, lo, hi = hf.initial_centres_and_bounds(v_sys_kms=36.0)
    assert np.allclose(c, hf.NH3_11_OFFSETS_KMS + 36.0)
    assert np.all(hi[:-1] < lo[1:])


def test_too_wide_window_is_rejected():
    """A window wide enough to overlap must fail loudly rather than silently
    reopening the swap bug."""
    with pytest.raises(ValueError):
        hf.initial_centres_and_bounds(window_kms=hf.max_safe_window_kms() + 0.1)


def test_max_safe_window_matches_tightest_separation():
    # main <-> inner is the tightest (1,1) pair, 7.59 km/s apart
    assert hf.max_safe_window_kms() == pytest.approx(7.590 / 2, abs=1e-3)


def test_estimate_v_sys_finds_main_line():
    velos = np.linspace(-60, 60, 601) + 36.0
    tmb = np.exp(-0.5 * ((velos - 36.0) / 0.5) ** 2)
    assert hf.estimate_v_sys_kms(velos, tmb) == pytest.approx(36.0, abs=0.3)
