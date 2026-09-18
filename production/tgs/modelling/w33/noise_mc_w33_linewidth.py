"""R3: re-run the ratio-estimator Monte Carlo at W33's actual observational
setup, not the narrow synthetic line used in the original noise_mc.py.

The original test (output_noise_mc/) built its synthetic spectrum with an
intrinsic Delta_v = 0.3 km/s (Stutzki's own clump width) sampled at the
model's native 500-channel/0.152 km/s/channel grid, and found the ratio
estimator biased and non-Gaussian below satellite S/N ~5. Applied naively to
Tursun et al. (2022)'s W33 satellite S/N estimates (2-11, from their Table 4
Tmb/rms), that flagged three of five sources as compromised -- but that
comparison mixed two different setups: W33's real lines are 2-4 km/s wide
(4-8x broader than Stutzki's clumps) and Effelsberg's real backend samples
them at 0.48 km/s/channel (3.2x coarser than the model's native grid), so a
per-channel noise realisation is averaged down far more in a real W33
spectrum's fit than in the original test's narrow, finely-sampled one.

This redoes the Monte Carlo with both corrections, per W33 source, at ZERO
extra Magritte cost: the cached truth_spectrum.npz (a real forward model,
Delta_v=0.3 km/s, log n=6.5, T=24 -- see output_noise_mc/mc.py) is Gaussian-
broadened to each source's actual observed Delta_v (Tursun Table 4, via
params.LINES), then rebinned to 0.48 km/s channels (matching the Effelsberg
NH3(1,1) backend), before per-channel noise at that source's own digitized
rms (observed_data/W33_line_heights.csv) is injected and the fit re-run.

This is NOT a claim that W33's true underlying line shape is a broadened
version of the Stutzki-clump spectrum -- it tests the ESTIMATOR (fitting +
rebinning + S/N), which is the thing R3 is actually about, using the real
observational parameters (linewidth, channel width, noise) a W33 spectrum
actually has.
"""
import os
import sys
import json
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_MODELLING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODELLING not in sys.path:
    sys.path.insert(0, _MODELLING)

import params as p
import w33_observed_ratios as wr

TRUTH_SPEC = "/home/yasho379/magritte_rebuilt/scratch/output/output_noise_mc/results/truth_spectrum.npz"
ODIR = "/home/yasho379/magritte_rebuilt/scratch/output/output_noise_mc_w33/"
M = 500
BASE_DV_KMS = 0.3          # the cached spectrum's own intrinsic linewidth
W33_CHANNEL_KMS = 0.48     # Effelsberg NH3(1,1)/(2,2)/(4,4)-(6,6) channel width (Tursun Table 1 caption)
RATIOS = ('R_10_MAIN', 'R_12_MAIN', 'R_21_MAIN', 'R_01_MAIN')


def broaden_and_rebin(v, tmb, target_dv_kms, channel_kms=W33_CHANNEL_KMS):
    """Gaussian-broaden from BASE_DV_KMS to target_dv_kms (quadrature), then
    rebin onto channel_kms-spaced channels via linear interpolation."""
    if target_dv_kms <= BASE_DV_KMS:
        extra_fwhm = 0.0
    else:
        extra_fwhm = np.sqrt(target_dv_kms ** 2 - BASE_DV_KMS ** 2)
    if extra_fwhm > 0:
        dv_chan = v[1] - v[0]
        sigma_chan = (extra_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))) / dv_chan
        tmb = gaussian_filter1d(tmb, sigma_chan, mode='nearest')
    v_new = np.arange(v.min(), v.max(), channel_kms)
    tmb_new = np.interp(v_new, v, tmb)
    return v_new, tmb_new


def fit_once(v, tmb, rms):
    from nh3_NLTE_analysis import fit_five_gaussians
    import nh3_hyperfine as hf
    pars = fit_five_gaussians(v, tmb, 'one', sigma=np.full(v.size, rms))
    amps, cens = pars[0:15:3], pars[1:15:3]
    idx = hf.identify_components(cens - hf.estimate_v_sys_kms(v, tmb))
    return {k: float(amps[i]) for k, i in idx.items()}


if __name__ == "__main__":
    d = np.load(TRUTH_SPEC)
    v0, tmb0 = d['velos'], d['tmb']
    print(f"base spectrum: Delta_v={BASE_DV_KMS} km/s, {len(v0)} channels @ {v0[1]-v0[0]:.4f} km/s\n")

    report = {}
    for src in wr.W33_SOURCES:
        line = p.LINES[src]['1,1']
        target_dv = line['dv']
        # use the source's own digitized (1,1) rms as the injected noise level
        rms = wr.DIGITIZED[src]['1,1']['rms']

        v, tmb = broaden_and_rebin(v0, tmb0, target_dv)
        peak = float(np.max(tmb))
        sat_snr_true = peak * 0.222 / rms  # LTE outer-satellite fraction, as a reference S/N

        rows, fails = [], 0
        for m in range(M):
            noisy = tmb + np.random.default_rng(2000 + m).normal(0, rms, size=tmb.size)
            try:
                A = fit_once(v, noisy, rms)
            except Exception:
                fails += 1
                continue
            r = dict(R_10_MAIN=A['A_10'] / A['A_MAIN'], R_12_MAIN=A['A_12'] / A['A_MAIN'],
                     R_21_MAIN=A['A_21'] / A['A_MAIN'], R_01_MAIN=A['A_01'] / A['A_MAIN'])
            rows.append([r[k] for k in RATIOS])

        Rm = np.array(rows)
        # "true" (noiseless) ratios for bias reference
        A_true = fit_once(v, tmb, rms * 1e-6)
        true_r = np.array([A_true['A_10'] / A_true['A_MAIN'], A_true['A_12'] / A_true['A_MAIN'],
                           A_true['A_21'] / A_true['A_MAIN'], A_true['A_01'] / A_true['A_MAIN']])
        bias_pct = 100 * (Rm.mean(axis=0) - true_r) / true_r
        skew = [float(stats.skew(Rm[:, i])) for i in range(4)]
        kurt = [float(stats.kurtosis(Rm[:, i])) for i in range(4)]

        print(f"=== {src}: Delta_v={target_dv} km/s, channel={W33_CHANNEL_KMS} km/s "
              f"({len(v)} channels), rms={rms} K, peak={peak:.3f} K, "
              f"outer-sat S/N~{sat_snr_true:.1f}")
        print(f"    {M-fails}/{M} fits ok, {fails} failed")
        print(f"    bias %: R10={bias_pct[0]:+.1f} R12={bias_pct[1]:+.1f} "
              f"R21={bias_pct[2]:+.1f} R01={bias_pct[3]:+.1f}")
        print(f"    max|skew|={max(abs(x) for x in skew):.2f}  max|excess kurt|={max(abs(x) for x in kurt):.2f}")
        report[src] = dict(target_dv=target_dv, rms=rms, peak=peak, sat_snr=sat_snr_true,
                           n_ok=M - fails, n_fail=fails, true_ratios=true_r.tolist(),
                           mean_ratios=Rm.mean(axis=0).tolist(), bias_pct=bias_pct.tolist(),
                           sd=Rm.std(axis=0).tolist(), skew=skew, excess_kurtosis=kurt)
        sys.stdout.flush()

    os.makedirs(os.path.join(ODIR, 'results'), exist_ok=True)
    with open(os.path.join(ODIR, 'results', 'noise_mc_w33.json'), 'w') as f:
        json.dump(dict(ratios=list(RATIOS), M=M, channel_kms=W33_CHANNEL_KMS,
                       by_source=report), f, indent=2)
    print(f"\nWROTE {ODIR}results/noise_mc_w33.json")
