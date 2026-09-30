"""Sanity tests for analysis/validation/noise_mc_w33_linewidth.py's broaden_and_rebin -- pure
numpy, no Magritte compute."""
import numpy as np
import pytest

from analysis.validation.noise_mc_w33_linewidth import broaden_and_rebin, BASE_DV_KMS


def gaussian(v, amp, center, fwhm):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return amp * np.exp(-0.5 * ((v - center) / sigma) ** 2)


def test_broadening_increases_fwhm_to_target():
    v = np.linspace(-30, 30, 2000)
    tmb = gaussian(v, 1.0, 0.0, BASE_DV_KMS)
    target = 3.0
    v2, tmb2 = broaden_and_rebin(v, tmb, target, channel_kms=0.1)
    half_max = tmb2.max() / 2.0
    above = v2[tmb2 >= half_max]
    fwhm_out = above.max() - above.min()
    assert fwhm_out == pytest.approx(target, rel=0.1)


def test_no_broadening_when_target_below_base():
    v = np.linspace(-30, 30, 2000)
    tmb = gaussian(v, 1.0, 0.0, BASE_DV_KMS)
    v2, tmb2 = broaden_and_rebin(v, tmb, BASE_DV_KMS * 0.5, channel_kms=0.1)
    # should just rebin, not broaden further -- peak stays close to original
    assert tmb2.max() == pytest.approx(tmb.max(), rel=0.05)


def test_rebin_channel_spacing_matches_request():
    v = np.linspace(-30, 30, 2000)
    tmb = gaussian(v, 1.0, 0.0, 2.0)
    v2, tmb2 = broaden_and_rebin(v, tmb, 3.0, channel_kms=0.48)
    assert v2[1] - v2[0] == pytest.approx(0.48, rel=1e-6)


def test_peak_amplitude_roughly_conserved_by_broadening():
    """Broadening a Gaussian conserves its integral, not its peak -- peak
    should drop by roughly base_fwhm/target_fwhm for a much-broadened line."""
    v = np.linspace(-40, 40, 4000)
    tmb = gaussian(v, 1.0, 0.0, BASE_DV_KMS)
    v2, tmb2 = broaden_and_rebin(v, tmb, 3.0, channel_kms=0.1)
    expected_peak = BASE_DV_KMS / 3.0
    assert tmb2.max() == pytest.approx(expected_peak, rel=0.15)
