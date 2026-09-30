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
