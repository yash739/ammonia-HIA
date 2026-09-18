"""Tests for w33_observed_ratios.py."""
import pytest
import w33_observed_ratios as wr


def test_all_five_sources_load():
    assert set(wr.DIGITIZED.keys()) >= set(wr.W33_SOURCES)
    assert 'W33_Main' not in wr.W33_SOURCES  # absorption source, excluded


def test_main_line_cross_check_within_digitization_tolerance():
    """Digitized (1,1) peak vs Table 4's fitted Tmb should agree to within
    ~10% -- a much looser check than what the pixel positions imply, since
    digitization off a printed figure is not sub-percent precise."""
    for src in wr.W33_SOURCES:
        _, _, meta = wr.observed_ratio_vector(src)
        assert meta['main_disagreement_frac'] < 0.10, src


def test_outer_anomaly_sense_holds_everywhere():
    for src in wr.W33_SOURCES:
        _, _, meta = wr.observed_ratio_vector(src)
        assert meta['outer_sense_ok'], src


def test_inner_anomaly_sense_flagged_where_reversed():
    """Documented in the module docstring: A1, B1 and Main1 do not show the
    expected inner sense. Locking this in as a test so a future re-digitization
    is compared against a known baseline, not silently assumed unchanged."""
    flipped = {s for s in wr.W33_SOURCES
               if not wr.observed_ratio_vector(s)[2]['inner_sense_ok']}
    assert flipped == {'W33_Main1', 'W33_A1', 'W33_B1'}


def test_nondetections_return_upper_limits_not_zero():
    for src in ('W33_Main1', 'W33_A1', 'W33_B1'):
        obs, err, _ = wr.observed_ratio_vector(src)
        assert obs['R_21_11'] is None
        assert err['R_21_11'] is not None and err['R_21_11'] > 0


def test_detected_21_matches_table4_for_A_and_B():
    for src, expected in (('W33_A', 0.48/5.18), ('W33_B', 0.35/3.08)):
        obs, _, _ = wr.observed_ratio_vector(src)
        assert obs['R_21_11'] == pytest.approx(expected, rel=1e-6)


def test_ratios_are_between_zero_and_one_where_detected():
    for src in wr.W33_SOURCES:
        obs, _, _ = wr.observed_ratio_vector(src)
        for k in ('R_01_MAIN', 'R_10_MAIN', 'R_12_MAIN', 'R_21_MAIN'):
            assert 0 < obs[k] < 1, (src, k)


def test_quadrant_gate_A_and_B_pass():
    for src in ('W33_A', 'W33_B'):
        g = wr.quadrant_gate(src)
        assert g['quadrant'] == 'II', src


def test_quadrant_gate_main1_a1_b1_fail():
    """Consistent with the reversed/tied inner sense already documented for
    these three sources: Main1 lands in the forbidden quadrant, A1/B1 in the
    expansion quadrant. None of the three is fittable by a static sphere at
    face value on this data."""
    assert wr.quadrant_gate('W33_Main1')['quadrant'] == 'IV'
    assert wr.quadrant_gate('W33_A1')['quadrant'] == 'I'
    assert wr.quadrant_gate('W33_B1')['quadrant'] == 'I'


def test_quadrant_gate_matches_hand_computed_HIA():
    g = wr.quadrant_gate('W33_A')
    obs, _, _ = wr.observed_ratio_vector('W33_A')
    assert g['HIA_IS'] == pytest.approx(obs['R_21_MAIN'] / obs['R_12_MAIN'])
    assert g['HIA_OS'] == pytest.approx(obs['R_01_MAIN'] / obs['R_10_MAIN'])
