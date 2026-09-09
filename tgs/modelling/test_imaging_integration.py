"""Validate the source-integrated spectrum against Stutzki & Winnewisser (1985)
Eq. (10) -- a closed-form check of the imaging path that needs no observation.

For a uniform sphere of source function S and radial optical depth tau_G, the
emergent intensity at impact parameter b is S(1 - e^-tau(b)) with
tau(b) = 2 tau_G sqrt(1 - (b/R)^2). Averaging that over the projected disc gives
exactly S[1 - e(2 tau_G)], which is the form Stutzki's Eq. (8) uses in place of
the single-sightline (1 - e^-tau).
"""
import numpy as np
import pytest

from nh3_NLTE_sphere import (stutzki_e_tau, _source_integrated_spectrum,
                              _center_beam_spectrum)


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


@pytest.mark.parametrize("tau_G", [0.05, 0.3, 1.0, 3.0, 10.0])
def test_disc_average_reproduces_stutzki_e_tau(tau_G):
    """The integrated spectrum must equal S[1 - e(2 tau_G)].

    The image is square while the source is a disc, so the pixel mean is over
    the square; scale by the area ratio (pi/4 of the square is the unit disc)
    to recover the disc average.
    """
    nx = ny = 400                      # fine grid: this tests the formula, not sampling
    img = synthetic_sphere_image(tau_G, nx=nx, ny=ny, fov_radii=1.0)
    square_mean = float(_source_integrated_spectrum(img, nx, ny)[0])
    disc_mean = square_mean * 4.0 / np.pi
    expected = 1.0 - float(stutzki_e_tau(np.array([2.0 * tau_G]))[0])
    assert disc_mean == pytest.approx(expected, rel=2e-3)


@pytest.mark.parametrize("tau_G", [0.3, 1.0, 3.0])
def test_centre_sightline_exceeds_disc_average(tau_G):
    """The bias the change exists to remove: the central pixel sees tau_max, the
    disc average sees less, so the centre always reports a larger (1-e^-tau)."""
    nx = ny = 400
    img = synthetic_sphere_image(tau_G, nx=nx, ny=ny)
    centre = float(_center_beam_spectrum(img, nx, ny)[0])
    disc = float(_source_integrated_spectrum(img, nx, ny)[0]) * 4.0 / np.pi
    assert centre > disc
    assert centre == pytest.approx(1.0 - np.exp(-2.0 * tau_G), rel=1e-3)


def test_mean_chord_is_two_thirds_of_central():
    """Optically thin limit: intensity is proportional to path length, so the
    disc average must be 2/3 of the central value (mean chord 4R/3 vs 2R)."""
    nx = ny = 400
    img = synthetic_sphere_image(1e-4, nx=nx, ny=ny)
    centre = float(_center_beam_spectrum(img, nx, ny)[0])
    disc = float(_source_integrated_spectrum(img, nx, ny)[0]) * 4.0 / np.pi
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
        coarse = float(_source_integrated_spectrum(
            synthetic_sphere_image(tau_G, 16, 16), 16, 16)[0])
        fine = float(_source_integrated_spectrum(
            synthetic_sphere_image(tau_G, 400, 400), 400, 400)[0])
        assert coarse == pytest.approx(fine, rel=0.03), f"tau_G={tau_G}"


def test_edge_clipping_when_fov_equals_source_diameter():
    """fov_pad_factor=1.0 puts the limb exactly at the image edge. Padding must
    change the recovered disc average by only the area rescaling, so a mismatch
    here is real clipped flux."""
    tau_G = 1.0
    tight = synthetic_sphere_image(tau_G, 400, 400, fov_radii=1.0)
    padded = synthetic_sphere_image(tau_G, 400, 400, fov_radii=1.15)
    tight_disc = float(_source_integrated_spectrum(tight, 400, 400)[0]) * 4.0 / np.pi
    padded_disc = (float(_source_integrated_spectrum(padded, 400, 400)[0])
                   * 4.0 / np.pi * 1.15**2)
    assert padded_disc == pytest.approx(tight_disc, rel=5e-3)
