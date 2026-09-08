"""
Stage B imaging helpers: field-of-view/pixel-scale sizing to go with
nh3_NLTE_sphere.run_model's `fov_pad_factor`, and the beam convolution itself.

The bug this exists to fix: Magritte's imager sets the image field of view to the
mesh's own outer-boundary bounding box (confirmed in
Magritte/src/model/image/image.cpp::set_coordinates_projection_surface) -- with no
padding, the image is exactly source-diameter-wide, so
`scipy.ndimage.gaussian_filter(cube, sigma=..., mode='nearest')`
(as in W33_smooth_freq_dependent.py) replicates edge brightness outward instead of
convolving against real background, inflating the beam-diluted result. The fix has
two parts: pad the mesh (nh3_NLTE_sphere.run_model's `fov_pad_factor`) *and* sample
finely enough that padding doesn't trade the edge-replication bug for aliasing
(this module's `required_pixels`), *and* convolve with zero-padding, never
'nearest' (this module's `convolve_beam`).
"""

import numpy as np
from scipy.ndimage import gaussian_filter

import params as p

PX_PER_BEAM_FWHM = 5.0  # minimum spatial sampling target, per the plan's Stage B


def fov_arcsec(r_boundary_m, distance_pc=p.DISTANCE_PC):
    """Full field-of-view [arcsec] subtended by a mesh whose outer boundary is at
    physical radius `r_boundary_m` (i.e. `r_out * fov_pad_factor` from run_model)."""
    return 2.0 * p.m_to_arcsec(r_boundary_m, distance_pc=distance_pc)


def required_pixels(r_boundary_m, freq_hz, distance_pc=p.DISTANCE_PC,
                     px_per_beam=PX_PER_BEAM_FWHM):
    """Minimum (odd, for a well-defined center pixel) nx_pix=ny_pix so the padded
    FOV samples the beam at >= px_per_beam pixels per FWHM."""
    fov = fov_arcsec(r_boundary_m, distance_pc=distance_pc)
    beam = p.beam_fwhm_arcsec(freq_hz)
    n = int(np.ceil(fov * px_per_beam / beam))
    return n + 1 if n % 2 == 0 else n  # keep it odd


def pick_fov_pad_factor(theta_source_arcsec, freq_hz, beam_widths=4.0):
    """Suggest an `fov_pad_factor` so the padded FOV spans `beam_widths` beam-FWHMs
    beyond the (angular) source size -- generous enough for the convolution kernel
    to see real background at the frame edge."""
    beam = p.beam_fwhm_arcsec(freq_hz)
    target_radius_arcsec = theta_source_arcsec + beam_widths * beam
    if theta_source_arcsec <= 0:
        return beam_widths * beam  # point-source fallback: pad by absolute beam-widths
    return target_radius_arcsec / theta_source_arcsec


def convolve_beam(cube, pixel_scale_arcsec, freq_hz):
    """Convolve a (ny, nx, nfreq) brightness-temperature cube with the Effelsberg
    Gaussian beam at `freq_hz`, per frequency slice, with zero-padded edges (never
    'nearest' -- that replicates the frame edge instead of seeing real background).
    """
    beam_fwhm_px = p.beam_fwhm_arcsec(freq_hz) / pixel_scale_arcsec
    sigma_px = beam_fwhm_px / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return gaussian_filter(cube, sigma=[sigma_px, sigma_px, 0], mode='constant', cval=0.0)


def point_source_check(freq_hz, distance_pc=p.DISTANCE_PC, pixel_scale_arcsec=None,
                        nx_pix=201):
    """Sanity check for the Verification section: convolving a single delta-like
    peak (all flux in the center pixel, zero elsewhere) with `convolve_beam` should
    reproduce a 2D Gaussian of the expected beam FWHM, in pixels, to a few percent.
    Returns (measured_fwhm_px, expected_fwhm_px). Oversamples well past
    PX_PER_BEAM_FWHM and linearly interpolates the half-max crossing -- this is a
    self-test of the convolution code, not the production pixel scale, so it can
    be made far more precise than the few-px/beam target used for real images.
    """
    beam = p.beam_fwhm_arcsec(freq_hz)
    if pixel_scale_arcsec is None:
        pixel_scale_arcsec = beam / 40.0  # 40 px/beam-FWHM: sub-percent quantization
    cube = np.zeros((nx_pix, nx_pix, 1))
    cube[nx_pix // 2, nx_pix // 2, 0] = 1.0
    conv = convolve_beam(cube, pixel_scale_arcsec, freq_hz)
    profile = conv[nx_pix // 2, :, 0]
    peak = profile.max()
    half = peak / 2.0
    above = np.where(profile >= half)[0]
    lo, hi = above[0], above[-1]

    def interp_crossing(i0, i1):
        y0, y1 = profile[i0], profile[i1]
        return i0 + (half - y0) / (y1 - y0) if y1 != y0 else float(i0)

    left = interp_crossing(lo - 1, lo) if lo > 0 else float(lo)
    right = interp_crossing(hi, hi + 1) if hi < len(profile) - 1 else float(hi)
    measured_fwhm_px = right - left
    expected_fwhm_px = beam / pixel_scale_arcsec
    return measured_fwhm_px, expected_fwhm_px


if __name__ == "__main__":
    freq = p.FREQ_HZ['1,1']
    measured, expected = point_source_check(freq)
    print(f"point-source convolution check @ (1,1): measured FWHM = {measured:.2f} px, "
          f"expected = {expected:.2f} px "
          f"({100*abs(measured-expected)/expected:.1f}% difference)")

    theta_source = 5.0  # example: a 5" source (roughly W33 A1/B1's Table 1 size / 2)
    pad_factor = pick_fov_pad_factor(theta_source, freq)
    r_source_m = p.arcsec_to_m(theta_source)
    r_pad = r_source_m * pad_factor
    npix = required_pixels(r_pad, freq)
    print(f"example: {theta_source}\" source -> fov_pad_factor={pad_factor:.2f}, "
          f"padded boundary={p.m_to_arcsec(r_pad):.2f}\", FOV={fov_arcsec(r_pad):.2f}\", "
          f"nx_pix={npix} for >= {PX_PER_BEAM_FWHM:.0f} px/beam-FWHM")
