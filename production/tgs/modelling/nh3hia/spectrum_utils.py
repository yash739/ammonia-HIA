"""
Small, general spectrum-extraction utilities shared by the active pipeline.

native_peak_tmb was originally written inside ratio_screen.py (now
relegated -- see _relegated/README.md) as a private helper; extracted here as
a public, standalone utility since invert_ratios.py still depends on it and
the module it was written in no longer represents the pipeline's approach.
"""

import os
import sys
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

EXTRACTOR_VERSION = 'v2_gaussfit'   # recorded in LUT CSVs; v1 = 5-channel smoothed peak (biased low)


def native_peak_tmb(velos, Is, freq_rest, smooth_channels=5, detection_sigma=3.0):
    """Convert a run_model-returned (velos [km/s], Is [raw intensity, NOT
    brightness temperature]) pair to Tmb and return (peak, noise, detected).

    Two issues found by hand on real output before this existed:

    1. Units: run_model's extra_spectra tuples are raw model.images[-1].I
       (~1e-13 to 1e-19 W/m^2/Hz/sr, straight from Magritte, see
       nh3_NLTE_sphere._center_beam_spectrum) -- NOT Kelvin. Only the cube
       path (return_cubes=True -> nh3_NLTE_sphere._intensity_cube_to_Tmb) is
       pre-converted; the plain (velos, Is) tuple is not. Fixed by reusing
       nh3_NLTE_analysis.intensity_to_Tmb (the same conversion A_MAIN uses),
       not re-deriving it.

    2. Baseline: for lines this faint (most of the W33 low-density
       candidates), nh3_NLTE_analysis.subtract_baseline's linear fit to the
       first/last 10% of channels breaks badly if a numerical ripple (a
       real, reproducible artifact -- confirmed by comparing raw spectra
       across candidates; a strong, well-thermalized line like Stutzki's
       OMC S4 is smooth, but these faint ones show small, roughly periodic
       dips scattered across the whole window, almost certainly a mesh/ray
       discretization effect at this level of sub-thermalization, not a
       real 18-hyperfine-component (2,1) feature) happens to land in that
       edge region -- the fit gets pulled off and np.max() afterward reads
       off noise, not signal (found: 2.4e-7 K "peak" against a 0.16 K-scale
       ripple floor in the very same spectrum). Fixed with a median baseline
       (robust to a handful of dips, since real signal -- if present at all
       -- occupies only a few of 500 channels) plus light smoothing (a real
       line, even a narrow one at these clump widths, spans >1 channel;
       single-channel noise doesn't survive a 5-channel box average).

    Returns (peak_Tmb, noise_estimate, detected) -- `detected` is False when
    peak < detection_sigma * noise, meaning the line is not just faint but
    unresolvable at this simulation's own numerical precision; callers
    should report "below noise floor", not the raw peak number, in that case.
    """
    from nh3hia.spectral_fit import intensity_to_Tmb
    velos = np.asarray(velos)
    Tmb = intensity_to_Tmb(1000 * velos, np.asarray(Is), freq_rest)
    baseline = np.median(Tmb)
    resid = Tmb - baseline
    if smooth_channels > 1:
        kernel = np.ones(smooth_channels) / smooth_channels
        resid = np.convolve(resid, kernel, mode='same')
    peak = float(np.max(resid))
    noise = float(np.std(resid))
    detected = peak >= detection_sigma * noise
    return peak, noise, detected


def line_peak_tmb(velos, Is, freq_rest, window_kms=2.0, return_info=False):
    """Unbiased peak brightness temperature of one (blended) hyperfine line.

    Replaces `native_peak_tmb` as the amplitude of the (2,2) and (2,1) lines.
    That function box-smoothed over 5 channels to suppress single-channel
    noise, but model lines are only 2-4 channels wide (0.152 km/s channels,
    0.37-0.6 km/s FWHM), so the smoothing lowered the peak by 25-43% relative
    to the (1,1) Gaussian-fit amplitude it is divided by (important_notes
    sec. 23).

    Method: remove a linear baseline from the line-free edges (as for (1,1)),
    locate the peak on a 3-channel-smoothed copy (robust to noise), and fit a
    Gaussian within +-`window_kms` with bounded centre and width. If the fit
    fails or is inconsistent with the data (amplitude outside 0.9-1.35 x the
    raw maximum, or fitted FWHM outside 0.5 channels - 4 km/s), fall back to a
    model-free peak: the maximum of a cubic-spline interpolation of the raw
    spectrum. Both are unbiased for a line sampled at >~ 2 channels per FWHM,
    for any sub-channel position of the peak.

    Returns peak [K] or, with return_info=True, (peak, info) where info has
    method ('gauss' | 'spline'), fwhm [km/s] (nan for spline), raw_peak,
    spline_peak and gauss_over_spline."""
    from nh3hia.spectral_fit import intensity_to_Tmb, subtract_baseline
    v = np.asarray(velos, dtype=float)
    T = subtract_baseline(v, intensity_to_Tmb(1000 * v, np.asarray(Is), freq_rest))
    dv = float(np.median(np.diff(v)))
    raw = float(T.max())
    fine = np.linspace(v[0], v[-1], 10 * len(v))
    spline = float(CubicSpline(v, T)(fine).max())
    i0 = int(np.argmax(np.convolve(T, np.ones(3) / 3, mode='same')))
    c0 = v[i0]
    sel = np.abs(v - c0) < window_kms
    info = dict(method='spline', fwhm=np.nan, raw_peak=raw, spline_peak=spline, gauss_over_spline=np.nan)
    peak = spline
    if sel.sum() >= 6 and raw > 0:
        g = lambda x, a, c, sg: a * np.exp(-0.5 * ((x - c) / sg) ** 2)
        try:
            po, _ = curve_fit(g, v[sel], T[sel], p0=[raw, c0, 0.2],
                              bounds=([0.0, c0 - 0.5, 0.25 * dv], [3.0 * raw, c0 + 0.5, 4.0 / 2.3548]),
                              maxfev=2000)
            a, c, sg = po
            fwhm = 2.3548 * sg
            if 0.9 * raw <= a <= 1.35 * raw and 0.5 * dv <= fwhm <= 4.0:
                peak = float(a)
                info.update(method='gauss', fwhm=float(fwhm), gauss_over_spline=float(a / spline))
        except Exception:
            pass
    return (peak, info) if return_info else peak
