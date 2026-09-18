"""Synthetic observational noise for model spectra.

WHY THE NOISE GOES ON THE SPECTRUM, NOT ON THE RATIOS
-----------------------------------------------------
It is tempting to perturb the five hyperfine ratios directly with independent
Gaussian scatter. That produces the wrong error structure in two ways:

  1. All five ratios share `A_MAIN` as denominator, so their errors are strongly
     correlated. Treating them as independent over-counts the information and
     yields spuriously tight posteriors.
  2. A ratio whose denominator carries noise is not Gaussian-distributed
     (Marsaglia/Fieller, heavy tails). Whether the Gaussian approximation holds
     depends on the main-line S/N and must be checked per source, not assumed.

Adding noise per channel and pushing it through the *same* extraction code
reproduces both effects automatically, with no analytic propagation required.

UNITS
-----
`run_model` carries raw specific intensity (W/m^2/Hz/sr, ~1e-13 to 1e-19), not
Kelvin -- only `nh3_NLTE_analysis.intensity_to_Tmb` converts. A noise level
quoted in Kelvin (which is how observers quote it) must therefore be converted
to intensity before being added, which `tmb_rms_to_intensity_rms` does via the
inverse Rayleigh-Jeans relation I = 2 k T nu^2 / c^2.

OBSERVATIONAL NOISE IS NOT NUMERICAL NOISE
------------------------------------------
`spectrum_utils.native_peak_tmb` also reports a "noise", but that is model
discretisation ripple measured as the standard deviation over the whole
spectrum (signal included, so an upper bound) and used only as a detection
floor. It belongs to model fidelity; this module's noise belongs to the
measurement. Conflating them double-counts.
"""

import numpy as np

K_B = 1.380649e-23
C_LIGHT = 2.99792458e8


def tmb_rms_to_intensity_rms(rms_K, freq_hz):
    """Convert a main-beam brightness-temperature RMS [K] to specific-intensity
    RMS [W/m^2/Hz/sr] at `freq_hz`, inverting the Rayleigh-Jeans relation used
    by nh3_NLTE_analysis.intensity_to_Tmb.
    """
    return 2.0 * K_B * float(rms_K) * float(freq_hz) ** 2 / C_LIGHT ** 2


def add_channel_noise(intensities, rms_K, freq_hz, seed=None):
    """Add independent Gaussian noise of `rms_K` Kelvin to every channel.

    `intensities` may be a 1-D spectrum or a 2-D (npix, nfreq) image; noise is
    drawn independently per element, matching an observation where each channel
    (and each pixel) carries its own realisation.

    Returns a new array; the input is not modified. `rms_K` of 0 or None returns
    the input unchanged, so the default path is byte-identical to a noiseless run.
    """
    if not rms_K:
        return intensities
    arr = np.asarray(intensities, dtype=float)
    rms_I = tmb_rms_to_intensity_rms(rms_K, freq_hz)
    rng = np.random.default_rng(seed)
    return arr + rng.normal(0.0, rms_I, size=arr.shape)


def rms_for_target_snr(peak_tmb_K, snr):
    """RMS [K] giving a requested peak signal-to-noise on a line of the given
    peak brightness. Used to build an S/N ladder anchored on a real spectrum
    rather than on an invented absolute noise level.
    """
    if snr <= 0:
        raise ValueError("snr must be positive")
    return float(peak_tmb_K) / float(snr)


def line_seed(base_seed, label):
    """Deterministic per-line seed derived from a run's base seed.

    Each imaged line must carry an independent noise realisation, as separate
    observations would, while the whole run stays reproducible from one seed.
    Uses zlib.crc32 rather than hash(), whose string hashing is randomised per
    process and would silently break reproducibility across runs.
    """
    import zlib
    return (int(base_seed), zlib.crc32(str(label).encode()) & 0xFFFFFFFF)
