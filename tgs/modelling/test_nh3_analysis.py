"""Regression tests for the hyperfine fit + component identification in
nh3_NLTE_analysis.py. Synthetic spectra only -- no Magritte, no FITS.

The test that matters most is test_recovers_published_anomaly_sense: it builds a
spectrum with the anomaly the literature reports (F1 0->1 enhanced, F1 1->0
suppressed) and checks the pipeline reports it that way round. The previous
index-based assignment would fail it.
"""

import numpy as np
import pytest

import nh3_hyperfine as hf
from nh3_NLTE_analysis import fit_five_gaussians, multi_gaussian


def synth_11(amps, v_sys=0.0, sigma=0.9, vmin=-60, vmax=60, n=800, noise=0.0, seed=0):
    """Build a synthetic NH3 (1,1) spectrum from per-component peak amplitudes
    given in ascending-velocity order (A_10, A_12, A_MAIN, A_21, A_01)."""
    v = np.linspace(vmin, vmax, n) + v_sys
    pars = []
    for a, off in zip(amps, hf.NH3_11_OFFSETS_KMS):
        pars.extend([a, off + v_sys, sigma])
    tmb = multi_gaussian(v, *pars)
    if noise:
        tmb = tmb + np.random.default_rng(seed).normal(0, noise, size=v.size)
    return v, tmb


def _fit_and_label(v, tmb, **kw):
    pars = fit_five_gaussians(v, tmb, 'one', **kw)
    amps, cens, sigs = pars[0:15:3], pars[1:15:3], pars[2:15:3]
    idx = hf.identify_components(cens - hf.estimate_v_sys_kms(v, tmb))
    return {k: float(amps[i]) for k, i in idx.items()}, {k: float(cens[i]) for k, i in idx.items()}


def test_recovers_published_anomaly_sense():
    """Hyperfine selective trapping enhances F1 0->1 and suppresses F1 1->0
    (Stutzki & Winnewisser 1985; Camarata et al. 2015). Build exactly that and
    confirm it is reported that way round."""
    # ascending velocity: A_10 (suppressed), A_12, A_MAIN, A_21, A_01 (enhanced)
    A, _ = _fit_and_label(*synth_11([0.18, 0.30, 1.0, 0.26, 0.27]))
    assert A['A_01'] > A['A_10'], "outer pair reported with the wrong asymmetry"
    assert A['A_12'] > A['A_21'], "inner pair reported with the wrong asymmetry"
    assert A['A_01'] == pytest.approx(0.27, rel=0.05)
    assert A['A_10'] == pytest.approx(0.18, rel=0.05)


def test_amplitudes_recovered_at_rest():
    truth = [0.20, 0.28, 1.0, 0.28, 0.20]
    A, CEN = _fit_and_label(*synth_11(truth))
    for k, t in zip(hf.NH3_11_KEYS_BY_VELOCITY, truth):
        assert A[k] == pytest.approx(t, rel=0.05)
    for k, off in zip(hf.NH3_11_KEYS_BY_VELOCITY, hf.NH3_11_OFFSETS_KMS):
        assert CEN[k] == pytest.approx(off, abs=0.2)


def test_recovers_at_w33_systemic_velocity():
    """W33 sits near v_LSR = 36 km/s; the fit must track that without the
    components wandering out of their windows."""
    truth = [0.18, 0.30, 1.0, 0.26, 0.27]
    A, CEN = _fit_and_label(*synth_11(truth, v_sys=36.0))
    assert A['A_01'] > A['A_10']
    for k, off in zip(hf.NH3_11_KEYS_BY_VELOCITY, hf.NH3_11_OFFSETS_KMS):
        assert CEN[k] == pytest.approx(off + 36.0, abs=0.3)


def test_no_component_swap_under_noise():
    """500 noisy realisations, zero mis-assignments -- the failure mode that
    unbounded centres allowed."""
    truth = [0.18, 0.30, 1.0, 0.26, 0.27]
    bad = 0
    for seed in range(500):
        v, tmb = synth_11(truth, sigma=0.9, noise=0.02, seed=seed)
        try:
            A, CEN = _fit_and_label(v, tmb)
        except ValueError:
            bad += 1
            continue
        # ordering of identified centres must stay monotonic in velocity
        cens = [CEN[k] for k in hf.NH3_11_KEYS_BY_VELOCITY]
        if not all(cens[i] < cens[i + 1] for i in range(4)):
            bad += 1
    assert bad == 0, f"{bad}/500 noisy fits mis-assigned components"


def test_anomaly_sense_survives_noise():
    """The measured asymmetry must keep its sign at realistic S/N."""
    truth = [0.18, 0.30, 1.0, 0.26, 0.27]
    wrong = 0
    for seed in range(200):
        v, tmb = synth_11(truth, noise=0.02, seed=seed + 1000)
        A, _ = _fit_and_label(v, tmb)
        if not A['A_01'] > A['A_10']:
            wrong += 1
    assert wrong == 0, f"{wrong}/200 noisy fits inverted the anomaly sense"


def test_integrated_ratio_matches_peak_ratio_for_equal_widths():
    """With all components at one width the two estimators must agree; they
    diverge only when widths differ, which is the bias Zhou et al. describe."""
    truth = [0.18, 0.30, 1.0, 0.26, 0.27]
    v, tmb = synth_11(truth)
    pars = fit_five_gaussians(v, tmb, 'one')
    amps, sigs = pars[0:15:3], pars[2:15:3]
    idx = hf.identify_components(pars[1:15:3] - hf.estimate_v_sys_kms(v, tmb))
    peak = amps[idx['A_01']] / amps[idx['A_MAIN']]
    integ = ((amps[idx['A_01']] * abs(sigs[idx['A_01']]))
             / (amps[idx['A_MAIN']] * abs(sigs[idx['A_MAIN']])))
    assert integ == pytest.approx(peak, rel=0.05)


def test_covariance_available_when_sigma_supplied():
    truth = [0.18, 0.30, 1.0, 0.26, 0.27]
    v, tmb = synth_11(truth, noise=0.02, seed=7)
    pars, pcov = fit_five_gaussians(v, tmb, 'one', sigma=np.full(v.size, 0.02),
                                     return_cov=True)
    assert pcov.shape == (15, 15)
    perr = np.sqrt(np.diag(pcov))
    assert np.all(np.isfinite(perr)) and np.all(perr > 0)
    # amplitude uncertainty should be of order the noise, not wildly off
    assert 1e-4 < perr[0] < 0.05
