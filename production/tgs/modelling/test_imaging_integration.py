"""Validate the source-integrated spectrum against Stutzki & Winnewisser (1985)
Eq. (10) -- a closed-form check of the imaging path that needs no observation.

For a uniform sphere of source function S and radial optical depth tau_G, the
emergent intensity at impact parameter b is S(1 - e^-tau(b)) with
tau(b) = 2 tau_G sqrt(1 - (b/R)^2). Averaging that over the projected disc gives
exactly S[1 - e(2 tau_G)], which is the form Stutzki's Eq. (8) uses in place of
the single-sightline (1 - e^-tau).

Two separate things are tested here, deliberately kept apart:
  - the general disc-in-square-image averaging identity
    (_disc_average_from_image_mean), using synthetic images where the disc
    radius is chosen directly and is NOT tied to Magritte's fov_pad_factor
    semantics -- these have no HEALPix discretisation and should not go
    through the Magritte-specific 15/16 correction; and
  - that _source_integrated_spectrum correctly converts a real
    fov_pad_factor into the measured Magritte image geometry before applying
    that same identity (test_source_integrated_spectrum_wiring below).
"""
import numpy as np
import pytest

from nh3_NLTE_sphere import (stutzki_e_tau, _source_integrated_spectrum,
                              _center_beam_spectrum, _disc_average_from_image_mean,
                              MAGRITTE_IMAGE_HALF_EXTENT_FACTOR)


def synthetic_sphere_image(tau_G, nx=16, ny=16, fov_radii=1.0, S=1.0):
    """Image of a uniform sphere on an nx x ny grid spanning +/- fov_radii * R."""
    ax = (np.arange(nx) + 0.5) / nx * 2 - 1.0
    ay = (np.arange(ny) + 0.5) / ny * 2 - 1.0
    X, Y = np.meshgrid(ax * fov_radii, ay * fov_radii)
    b = np.sqrt(X**2 + Y**2)
    inside = b < 1.0
    tau = np.zeros_like(b)
    tau[inside] = 2.0 * tau_G * np.sqrt(1.0 - b[inside]**2)
    I = S * (1.0 - np.exp(-tau))
    return I.reshape(-1, 1)          # (npix, nfreq=1)


def _disc_avg(img, nx, ny, fov_radii):
    """Plain image mean -> disc average, for a synthetic image built with the
    given fov_radii (disc radius R occupies a fraction 1/fov_radii of the
    image half-width)."""
    image_mean = np.asarray(img).mean(axis=0)
    return _disc_average_from_image_mean(image_mean, disc_radius_frac=1.0 / fov_radii)


@pytest.mark.parametrize("tau_G", [0.05, 0.3, 1.0, 3.0, 10.0])
def test_disc_average_reproduces_stutzki_e_tau(tau_G):
    """The disc average must equal S[1 - e(2 tau_G)]."""
    nx = ny = 400                      # fine grid: this tests the formula, not sampling
    img = synthetic_sphere_image(tau_G, nx=nx, ny=ny, fov_radii=1.0)
    disc_mean = float(_disc_avg(img, nx, ny, fov_radii=1.0)[0])
    expected = 1.0 - float(stutzki_e_tau(np.array([2.0 * tau_G]))[0])
    assert disc_mean == pytest.approx(expected, rel=2e-3)


@pytest.mark.parametrize("tau_G", [0.3, 1.0, 3.0])
def test_centre_sightline_exceeds_disc_average(tau_G):
    """The bias the change exists to remove: the central pixel sees tau_max, the
    disc average sees less, so the centre always reports a larger (1-e^-tau)."""
    nx = ny = 400
    img = synthetic_sphere_image(tau_G, nx=nx, ny=ny)
    centre = float(_center_beam_spectrum(img, nx, ny)[0])
    disc = float(_disc_avg(img, nx, ny, fov_radii=1.0)[0])
    assert centre > disc
    assert centre == pytest.approx(1.0 - np.exp(-2.0 * tau_G), rel=1e-3)


def test_mean_chord_is_two_thirds_of_central():
    """Optically thin limit: intensity is proportional to path length, so the
    disc average must be 2/3 of the central value (mean chord 4R/3 vs 2R)."""
    nx = ny = 400
    img = synthetic_sphere_image(1e-4, nx=nx, ny=ny)
    centre = float(_center_beam_spectrum(img, nx, ny)[0])
    disc = float(_disc_avg(img, nx, ny, fov_radii=1.0)[0])
    assert disc / centre == pytest.approx(2.0 / 3.0, rel=5e-3)


def test_e_tau_limits():
    assert float(stutzki_e_tau(np.array([1e-9]))[0]) == pytest.approx(1.0, abs=1e-6)
    assert float(stutzki_e_tau(np.array([1e3]))[0]) == pytest.approx(0.0, abs=1e-5)
    # monotonically decreasing
    t = np.linspace(0.01, 20, 200)
    assert np.all(np.diff(stutzki_e_tau(t)) < 0)


def test_16x16_sampling_error_is_small():
    """The production grid is 16x16; confirm its quadrature error against a fine
    grid is well under the ~10% observational uncertainty."""
    for tau_G in (0.3, 1.0, 3.0):
        coarse = float(_disc_avg(synthetic_sphere_image(tau_G, 16, 16), 16, 16, fov_radii=1.0)[0])
        fine = float(_disc_avg(synthetic_sphere_image(tau_G, 400, 400), 400, 400, fov_radii=1.0)[0])
        assert coarse == pytest.approx(fine, rel=0.03), f"tau_G={tau_G}"


def test_edge_clipping_when_fov_equals_source_diameter():
    """fov_radii=1.0 puts the limb exactly at the image edge. Padding must
    change the recovered disc average by nothing (the area rescaling is
    already handled by disc_radius_frac), so a mismatch here is real clipped
    flux at fov_radii=1.0."""
    tau_G = 1.0
    tight = synthetic_sphere_image(tau_G, 400, 400, fov_radii=1.0)
    padded = synthetic_sphere_image(tau_G, 400, 400, fov_radii=1.15)
    tight_disc = float(_disc_avg(tight, 400, 400, fov_radii=1.0)[0])
    padded_disc = float(_disc_avg(padded, 400, 400, fov_radii=1.15)[0])
    assert padded_disc == pytest.approx(tight_disc, rel=5e-3)


# --------------------------------------------------------------------------- #
# Magritte-specific wiring: does _source_integrated_spectrum correctly turn a
# real fov_pad_factor into the MEASURED image geometry (15/16), not the
# naively-assumed one?
# --------------------------------------------------------------------------- #

def test_magritte_half_extent_factor_is_15_16():
    """Locks in the measured constant (see nh3_NLTE_sphere.py docstring for
    the direct model.images[-1].ImX/ImY measurement this came from, matched to
    5-6 significant figures at two different fov_pad_factor values)."""
    assert MAGRITTE_IMAGE_HALF_EXTENT_FACTOR == pytest.approx(15.0 / 16.0)


def test_source_integrated_spectrum_matches_disc_average_at_measured_geometry():
    """_source_integrated_spectrum(fov_pad_factor=F) must equal the pure
    disc-average identity evaluated at the MEASURED disc_radius_frac =
    1/((15/16)*F), not at the naively-assumed 1/F."""
    tau_G = 1.0
    fov_pad = 1.15
    # Build the synthetic image at fov_radii equal to what a REAL Magritte
    # image at this fov_pad_factor actually has: source radius R occupies
    # 1/((15/16)*fov_pad) of the image half-width, i.e. the image spans
    # +/- (15/16)*fov_pad in units of R.
    fov_radii = MAGRITTE_IMAGE_HALF_EXTENT_FACTOR * fov_pad
    img = synthetic_sphere_image(tau_G, 400, 400, fov_radii=fov_radii)
    via_function = float(_source_integrated_spectrum(img, 400, 400, fov_pad_factor=fov_pad)[0])
    via_pure_math = float(_disc_avg(img, 400, 400, fov_radii=fov_radii)[0])
    assert via_function == pytest.approx(via_pure_math, rel=1e-6)
    # and both should recover the true disc average (Stutzki's e(tau))
    expected = 1.0 - float(stutzki_e_tau(np.array([2.0 * tau_G]))[0])
    assert via_function == pytest.approx(expected, rel=3e-3)


def test_source_integrated_never_exceeds_centre():
    """The bug the 15/16 fix caught: the old naive fov_pad_factor*r_out
    assumption implied a source average exceeding the central sightline,
    which is impossible for a uniform sphere. At the MEASURED geometry this
    must not happen, at any tau."""
    fov_pad = 1.15
    fov_radii = MAGRITTE_IMAGE_HALF_EXTENT_FACTOR * fov_pad
    for tau_G in (0.05, 0.3, 1.0, 3.0, 10.0):
        img = synthetic_sphere_image(tau_G, 400, 400, fov_radii=fov_radii)
        centre = float(_center_beam_spectrum(img, 400, 400)[0])
        disc = float(_source_integrated_spectrum(img, 400, 400, fov_pad_factor=fov_pad)[0])
        assert disc <= centre * 1.001, f"tau_G={tau_G}: disc={disc} > centre={centre}"
