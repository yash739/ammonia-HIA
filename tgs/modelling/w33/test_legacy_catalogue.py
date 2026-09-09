"""Tests for the pre-fix catalogue loader. Read-only against the CSV."""

import os
import numpy as np
import pytest

from legacy_catalogue import (LEGACY_CSV, load_legacy_catalogue,
                               anomaly_sense_fraction, assert_not_mistaken_for_lut)

pytestmark = pytest.mark.skipif(not os.path.exists(LEGACY_CSV),
                                 reason="legacy catalogue not present")


def test_swap_restores_the_published_anomaly_sense():
    """The whole justification for the swap: as labelled the outer sense is
    wrong for essentially every row, and correct for essentially every row
    afterwards. Published sense is R(0->1) > R(1->0)."""
    rows = load_legacy_catalogue(tau_range=(0.05, 20))
    frac = anomaly_sense_fraction(rows)
    assert frac['n'] > 400
    assert frac['outer'] > 0.99, "swap failed to restore the outer anomaly sense"
    assert frac['inner'] > 0.95, "inner pair should have been correct all along"


def test_raw_file_has_the_inverted_outer_sense():
    """Guards against someone 'fixing' the loader by removing the swap: the raw
    file must show the WRONG sense, which is why the swap exists."""
    import csv
    n_ok = n = 0
    with open(LEGACY_CSV, newline='') as f:
        for r in csv.DictReader(f):
            if r['Status'] != 'SUCCESS':
                continue
            tau = float(r['Main Hyperfine Optical Depth'])
            if not (0.05 <= tau <= 20) or float(r['Final Convergence']) < 90:
                continue
            n += 1
            n_ok += float(r['R_01_MAIN']) > float(r['R_10_MAIN'])
    assert n > 400
    assert n_ok / n < 0.01, "raw file no longer shows the inverted sense"


def test_amplitudes_swap_consistently_with_ratios():
    """Amplitude and ratio columns must be swapped together, not independently.
    Tolerance is set by the file itself, which stores both rounded to three
    decimals -- a tighter bound would be testing the rounding, not the swap."""
    rows = load_legacy_catalogue(tau_range=(0.05, 20))
    for r in rows[:50]:
        assert r['R_01_MAIN'] == pytest.approx(r['A_01'] / r['A_MAIN'], abs=2e-3)
        assert r['R_10_MAIN'] == pytest.approx(r['A_10'] / r['A_MAIN'], abs=2e-3)


def test_every_row_is_stamped_legacy():
    """Provenance must travel with the data so it cannot be mistaken for the
    new table downstream."""
    rows = load_legacy_catalogue()
    assert rows and all(r['is_legacy'] for r in rows)
    r = rows[0]
    assert r['nrays'] == 12 and r['resolution'] == 5
    assert r['spectrum'] == 'center' and r['estimator'] == 'peak'
    assert r['bare_clump'] is False


def test_guard_rejects_legacy_rows():
    rows = load_legacy_catalogue()
    with pytest.raises(ValueError, match="legacy"):
        assert_not_mistaken_for_lut(rows)
    assert_not_mistaken_for_lut([{'is_legacy': False}])   # must not raise


def test_convergence_and_tau_filters():
    strict = load_legacy_catalogue(min_convergence=99.0)
    loose = load_legacy_catalogue(min_convergence=0.0)
    assert 0 < len(strict) < len(loose)
    assert all(r['final_convergence'] >= 99.0 for r in strict)
    band = load_legacy_catalogue(min_convergence=0.0, tau_range=(1.0, 2.0))
    assert all(1.0 <= r['tau_main'] <= 2.0 for r in band)


def test_hia_matches_the_corrected_ratios():
    rows = load_legacy_catalogue(tau_range=(0.05, 20))
    for r in rows[:50]:
        assert r['HIA_OS'] == pytest.approx(r['R_01_MAIN'] / r['R_10_MAIN'], rel=1e-9)
        assert r['HIA_IS'] == pytest.approx(r['R_21_MAIN'] / r['R_12_MAIN'], rel=1e-9)
    # corrected catalogue should sit in the trapping quadrant on average
    assert np.median([r['HIA_OS'] for r in rows]) > 1.0
    assert np.median([r['HIA_IS'] for r in rows]) < 1.0
