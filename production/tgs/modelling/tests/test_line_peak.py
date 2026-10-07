"""Tests for the (2,2)/(2,1) peak estimator (nh3hia.spectrum_utils.line_peak_tmb).

The model spectra have 0.152 km/s channels and lines of 0.37-0.6 km/s FWHM
(2.4-4 channels). The old estimator (5-channel box smoothing) lowered the peak
by 25-43%; these tests pin the replacement's bias and robustness."""
import numpy as np
import pytest

from nh3hia.spectral_fit import intensity_to_Tmb, c, k_B
from nh3hia.spectrum_utils import line_peak_tmb, native_peak_tmb

FREQ = 23.8701e9
V = np.arange(-38.0, 38.0, 0.152)        # the LUT's channel grid


def make_I(profile_K, v=V):
    """Raw intensity whose conversion by intensity_to_Tmb gives profile_K."""
    return profile_K * (2 * k_B * (FREQ * (1 + 1000 * v / c)) ** 2) / c ** 2


def line(v, fwhm, centre, amp=1.0, components=((0.0, 1.0),)):
    """Sum of Gaussians (offset km/s, relative strength) sharing one width;
    returns the profile and its TRUE continuous peak."""
    sg = fwhm / 2.3548
    prof = lambda x: amp * sum(w * np.exp(-0.5 * ((x - centre - o) / sg) ** 2) for o, w in components)
    xs = np.linspace(centre - 3, centre + 3, 60001)
    return prof(v), prof(xs).max()


# (2,1)-like central group + satellites, (2,2)-like closely spaced satellites
PATTERNS = {
    'single': ((0.0, 1.0),),
    '21_like': ((0.0, 0.796), (-0.11, 0.052), (0.11, 0.052), (-0.17, 0.05), (0.17, 0.05)),
    '22_like': ((0.0, 0.7), (-0.17, 0.12), (0.17, 0.12), (-0.05, 0.03), (0.05, 0.03)),
}


@pytest.mark.parametrize('pattern', PATTERNS)
@pytest.mark.parametrize('fwhm', [0.25, 0.37, 0.43, 0.6, 1.0, 1.5])
def test_noise_free_bias_is_small_for_any_subchannel_offset(pattern, fwhm):
    rng = np.random.default_rng(7)
    for centre in 0.3 + rng.uniform(0, 0.152, 8):     # sub-channel positions
        T, true_peak = line(V, fwhm, centre, amp=0.4, components=PATTERNS[pattern])
        est = line_peak_tmb(V, make_I(T), FREQ)
        assert est == pytest.approx(true_peak, rel=0.04), (pattern, fwhm, centre)


def test_old_estimator_was_biased_low():
    """Documents why the replacement exists."""
    T, true_peak = line(V, 0.378, 0.3 + 0.07, amp=0.4)
    old = native_peak_tmb(V, make_I(T), FREQ)[0]
    assert old / true_peak < 0.75
    assert line_peak_tmb(V, make_I(T), FREQ) / true_peak > 0.97


@pytest.mark.parametrize('snr', [100, 30, 10])
def test_noisy_spectrum_unbiased_and_scatter_scales(snr):
    rng = np.random.default_rng(11)
    est = []
    for k in range(60):
        T, true_peak = line(V, 0.45, 0.3 + rng.uniform(0, 0.152), amp=1.0, components=PATTERNS['21_like'])
        Tn = T + rng.normal(0, true_peak / snr, T.shape)
        est.append(line_peak_tmb(V, make_I(Tn), FREQ) / true_peak)
    est = np.array(est)
    assert abs(est.mean() - 1) < 0.02 + 0.5 / snr          # noise-peak bias grows slowly with 1/SNR
    assert est.std() < 3.0 / snr


def test_falls_back_to_spline_for_pathological_input():
    """A single hot channel (spike) must not produce a wild fit."""
    T = np.zeros_like(V)
    T[len(V) // 2] = 1.0
    peak, info = line_peak_tmb(V, make_I(T), FREQ, return_info=True)
    assert 0.9 <= peak <= 1.6 and info['method'] in ('gauss', 'spline')


def test_zero_signal_returns_finite():
    peak = line_peak_tmb(V, make_I(np.zeros_like(V)), FREQ)
    assert np.isfinite(peak)
