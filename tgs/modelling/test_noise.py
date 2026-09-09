"""Unit tests for noise.py -- pure numpy, no Magritte compute."""

import numpy as np
import pytest

import noise

FREQ = 23.6945e9


def test_kelvin_to_intensity_roundtrip():
    """The conversion must invert nh3_NLTE_analysis.intensity_to_Tmb."""
    from nh3_NLTE_analysis import intensity_to_Tmb
    T = 0.06
    I = noise.tmb_rms_to_intensity_rms(T, FREQ)
    back = intensity_to_Tmb(np.array([0.0]), np.array([I]), FREQ)
    assert float(back[0]) == pytest.approx(T, rel=1e-9)


def test_measured_rms_matches_request():
    a = np.zeros((4, 4000))
    out = noise.add_channel_noise(a, 0.06, FREQ, seed=3)
    expected = noise.tmb_rms_to_intensity_rms(0.06, FREQ)
    assert float(np.std(out)) == pytest.approx(expected, rel=0.05)


def test_zero_and_none_are_exact_noops():
    """The default path must be byte-identical to a noiseless run, so enabling
    noise cannot perturb any existing result by accident."""
    a = np.random.default_rng(0).normal(size=(3, 50))
    assert noise.add_channel_noise(a, None, FREQ) is a
    assert noise.add_channel_noise(a, 0, FREQ) is a
    assert noise.add_channel_noise(a, 0.0, FREQ) is a


def test_input_is_not_mutated():
    a = np.zeros((2, 20))
    before = a.copy()
    noise.add_channel_noise(a, 0.1, FREQ, seed=1)
    assert np.array_equal(a, before)


def test_seed_reproducibility():
    a = np.zeros((2, 100))
    x = noise.add_channel_noise(a, 0.1, FREQ, seed=11)
    y = noise.add_channel_noise(a, 0.1, FREQ, seed=11)
    z = noise.add_channel_noise(a, 0.1, FREQ, seed=12)
    assert np.array_equal(x, y)
    assert not np.array_equal(x, z)


def test_per_line_seeds_are_independent_and_deterministic():
    assert noise.line_seed(7, '11') == noise.line_seed(7, '11')
    assert noise.line_seed(7, '11') != noise.line_seed(7, '22')
    a = np.zeros((2, 500))
    x = noise.add_channel_noise(a, 0.1, FREQ, seed=noise.line_seed(7, '11'))
    y = noise.add_channel_noise(a, 0.1, FREQ, seed=noise.line_seed(7, '22'))
    assert not np.array_equal(x, y)
    assert abs(float(np.corrcoef(x.ravel(), y.ravel())[0, 1])) < 0.1


def test_line_seed_survives_process_restart():
    """zlib.crc32, not hash(): Python randomises string hashing per process, so
    hash() would silently break reproducibility between runs."""
    import subprocess, sys, os
    here = os.path.dirname(os.path.abspath(noise.__file__))
    code = (f"import sys; sys.path.insert(0, {here!r}); import noise; "
            "print(noise.line_seed(7,'11'))")
    a = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    b = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert a.stdout.strip() == b.stdout.strip() == str(noise.line_seed(7, '11'))


def test_rms_for_target_snr():
    assert noise.rms_for_target_snr(4.713, 20) == pytest.approx(0.23565, rel=1e-4)
    with pytest.raises(ValueError):
        noise.rms_for_target_snr(1.0, 0)


def test_noise_is_gaussian_and_zero_mean():
    a = np.zeros((1, 20000))
    out = noise.add_channel_noise(a, 0.1, FREQ, seed=5)
    rms = noise.tmb_rms_to_intensity_rms(0.1, FREQ)
    assert abs(float(np.mean(out))) < 0.05 * rms
    # kurtosis of a Gaussian is 3
    z = out.ravel() / rms
    assert float(np.mean(z ** 4)) == pytest.approx(3.0, rel=0.15)
