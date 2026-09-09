import os
import time
import warnings
import h5py
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import magritte.setup as setup
import magritte.core as magritte
import magritte.plot as plot
import magritte.mesher as mesher
import magritte.tools as tools

warnings.filterwarnings('ignore')
import yt
from yt.funcs import mylog
mylog.setLevel(40)

from astropy import constants
from astropy.io import fits
from scipy.spatial import Delaunay

import nh3_hyperfine
import noise as _noise

# -----------------------------
# Vectorized Physics Functions -- CONSTANT AMMONIA ABUNDANCE HAS BEEN ASSUMED FOR SIMPLICITY
# -----------------------------
# Fraction of the cloud's own H2 density used for the radiatively inert exterior.
# It exists only to keep the Delaunay mesh well-defined out to the boundary and
# to give the remesher a density contrast to work with. Because it SCALES with
# the cloud, that contrast is identical at every grid point -- with a fixed
# exterior density it varied by four orders of magnitude across the intended
# (n_H2 = 10^3.5 - 10^7.5) grid, so different grid points were meshed at
# different effective resolution, which would have shown up in a lookup table's
# interpolation-error map as physics rather than mesh noise.
EXTERIOR_DENSITY_FRACTION = 1e-3


def get_properties_vectorized(positions, rho_cloud, r_out, XNH3, T_cloud, vturb,
                                exterior_density_fraction=EXTERIOR_DENSITY_FRACTION):
    """Calculates all cell properties using fast, vectorized Numpy operations.

    The sphere is BARE: outside r_out the NH3 abundance is zero, so the exterior
    is radiatively inert and the geometry matches the homogeneous sphere of
    Stutzki & Winnewisser (1985) Appendix A, whose escape probability beta(r)
    (their Fig. 3) is derived for a cloud that simply ends at R with isotropic
    CMB incident. Previously the exterior was given 100 cm^-3 of H2 *and*
    nNH3 = XNH3 * nH2 *and* T_cloud, i.e. an ammonia envelope at the cloud's own
    temperature -- radiatively small (its NH3 column is <1% of the cloud's even
    at the low-density corner) but not the assumed geometry, and it silently
    fixed envelope parameters we never chose.

    Ambient/envelope emission is not thereby dropped from the physics: it belongs
    in a post-hoc two-component decomposition (anomalous clump + LTE-like
    ambient), where its filling factor is fitted and reported rather than baked
    into every model.
    """
    m_H2 = 2.01588 * constants.u.si.value

    # Radii of all points
    r = np.linalg.norm(positions, axis=1)

    outside = r > r_out

    # Constant-density sphere; the exterior carries only mesh-support H2.
    gas_density = np.where(outside, rho_cloud * exterior_density_fraction, rho_cloud)

    nH2 = gas_density / m_H2
    nNH3 = np.where(outside, 0.0, XNH3 * nH2)

    # Temperature and Turbulence (constant arrays)
    tmp = np.full(len(positions), T_cloud, dtype=np.float64)
    trb = np.full(len(positions), (vturb / constants.c.si.value) ** 2, dtype=np.float64)

    # Velocity (Assuming v_radial = 0 based on original code logic)
    # If v_radial becomes non-zero, replace 0.0 with your v_radial value
    v_radial = 0.0 
    velocity = np.zeros_like(positions)
    if v_radial != 0.0:
        safe_r = np.where(r == 0, 1.0, r) # avoid division by zero at origin
        velocity[:, 0] = v_radial * (positions[:, 0] / safe_r)
        velocity[:, 1] = v_radial * (positions[:, 1] / safe_r)
        velocity[:, 2] = v_radial * (positions[:, 2] / safe_r)
    velocity = velocity / 3e8

    return nH2, nNH3, tmp, trb, velocity

def add_carta_beams_to_fits(input_fits, default_bmaj_deg, default_bmin_deg, default_bpa_deg, overwrite=True):
    with fits.open(input_fits, mode="readonly") as hdul:
        img_hdu = next((h for h in hdul if (isinstance(h, fits.PrimaryHDU) or isinstance(h, fits.ImageHDU)) 
                        and h.data is not None and h.header.get("NAXIS", 0) >= 2), None)
        
        if img_hdu is None:
            raise RuntimeError("No image HDU with data found.")

        hdr = deepcopy(img_hdu.header)
        data = img_hdu.data

        n_spectral = int(hdr.get("NAXIS3", 1))
        bmaj_deg = float(hdr.get("BMAJ", default_bmaj_deg))
        bmin_deg = float(hdr.get("BMIN", default_bmin_deg))
        bpa_deg  = float(hdr.get("BPA",  default_bpa_deg))

        hdr["BUNIT"] = ("Jy/beam", "Brightness unit")

        col_bmaj = fits.Column(name="BMAJ", format="D", array=np.full(n_spectral, bmaj_deg, dtype=np.float64))
        col_bmin = fits.Column(name="BMIN", format="D", array=np.full(n_spectral, bmin_deg, dtype=np.float64))
        col_bpa  = fits.Column(name="BPA",  format="D", array=np.full(n_spectral, bpa_deg,  dtype=np.float64))
        beams_hdu = fits.BinTableHDU.from_columns([col_bmaj, col_bmin, col_bpa], name="BEAMS")

        beams_hdu.header.update({
            "EXTNAME": "BEAMS",
            "TTYPE1": "BMAJ", "TTYPE2": "BMIN", "TTYPE3": "BPA",
            "TFORM1": "D", "TFORM2": "D", "TFORM3": "D",
            "COMMENT": "Restoring beam(s) for each spectral channel; units are degrees."
        })

        out_hdul = fits.HDUList([fits.PrimaryHDU(data=data, header=hdr), beams_hdu])
        out_name = input_fits.replace(".fits", "_withbeams.fits")
        out_hdul.writeto(out_name, overwrite=overwrite)

    if overwrite:
        os.remove(input_fits)


def _center_beam_spectrum(image_I, nx_pix, ny_pix):
    """Average the 4 pixels nearest the image center (row-major flattened index)."""
    r0, c0 = ny_pix // 2 - 1, nx_pix // 2 - 1
    idx = [r0 * nx_pix + c0, r0 * nx_pix + c0 + 1, (r0 + 1) * nx_pix + c0, (r0 + 1) * nx_pix + c0 + 1]
    return sum(image_I[i, :] for i in idx) / 4


T_CMB_K = 2.725


def _intensity_cube_to_Tmb(intensities, freq_rest, nx_pix, ny_pix):
    """Flattened (npix, nfreq) raw intensity -> (ny_pix, nx_pix, nfreq) Tmb cube
    (Rayleigh-Jeans, CMB-subtracted) -- the layout imaging.convolve_beam expects."""
    c = constants.c.si.value
    k_B = constants.k_B.si.value
    h = constants.h.si.value
    Tmb_flat = (c**2 * intensities) / (2 * k_B * freq_rest**2)
    Tmb_flat -= (h * freq_rest / k_B) / np.expm1(h * freq_rest / (k_B * T_CMB_K))
    nfreq = intensities.shape[1]
    return Tmb_flat.reshape(ny_pix, nx_pix, nfreq)


def _image_and_save_line(model, freq_rest, label, odir, tag, nx_pix=16, ny_pix=16, save_plot=True,
                          return_cube=False, save_image_fits=False,
                          noise_rms_K=None, noise_seed=None):
    """Image one line, save the center-beam spectrum FITS/PNG, return (velos, Is)
    or (velos, Is, cube) if return_cube -- cube is the full (ny_pix, nx_pix, nfreq) Tmb
    array, for beam convolution (see imaging.py), not just the center-pixel spectrum.

    save_image_fits: off by default. tools.save_fits() re-interpolates onto its own
    hardcoded 300x300 grid regardless of nx_pix/ny_pix, so each file is ~340MB
    (300*300*500freq*8B) no matter how coarse the actual image is -- and nothing in
    this pipeline (screen.py, analyse_spectra, stageb_pilot) reads it back; the cube
    used for analysis/convolution comes from model.images[-1].I in memory. Across a
    multi-hundred-point grid this is pure disk waste (measured: 176GB for 420 Stage A
    points). Only pass True for a one-off candidate you actually want to inspect
    (e.g. in CARTA) -- and pass npix_x/npix_y matching the real resolution then, not
    the wasteful default.
    """
    model.compute_spectral_discretisation(freq_rest - 3000000.00, freq_rest + 3000000.00, 500)
    model.compute_image_new(0, nx_pix, ny_pix)

    if save_image_fits:
        img_fits = os.path.join(odir, f'fits/NLTE_nh3_image_{label}_{tag}.fits')
        tools.save_fits(model, filename=img_fits, npix_x=nx_pix, npix_y=ny_pix)

    # Standard radio/LSR convention: v = c (nu_rest - nu) / nu_rest, so positive
    # velocity is redshifted (LOWER frequency). This pipeline previously used the
    # opposite sign, c (nu - nu_rest) / nu_rest, which mirrored every spectrum
    # relative to Stutzki's Fig. 1 and to observed data while the FITS header
    # below still declared CTYPE1 = 'VELO-LSR'. See nh3_hyperfine.py.
    freqs = np.array(model.images[-1].freqs)
    velos = nh3_hyperfine.freq_to_radio_velocity_kms(freqs, freq_rest)
    intensities = np.array(model.images[-1].I)[:, :]

    # Frequencies ascend, so radio-convention velocities descend. Reverse both so
    # the returned spectrum is ascending in velocity, which is what every
    # consumer (plotting, baseline fitting, FITS CDELT1) expects.
    if len(velos) > 1 and velos[1] < velos[0]:
        velos = velos[::-1]
        intensities = intensities[:, ::-1]

    # Synthetic observational noise goes in HERE -- on the image, before the
    # centre-beam extraction and before the FITS write below. That placement
    # matters: both consumers then see the same realisation, the FITS ->
    # analyse_spectra path for the (1,1) hyperfine fit and the in-memory
    # extra_spectra path for the (2,2) peak. Perturbing after the write would
    # desynchronise them. Noise is added in intensity units, converted from the
    # Kelvin RMS an observer would quote. See noise.py.
    if noise_rms_K:
        intensities = _noise.add_channel_noise(intensities, noise_rms_K, freq_rest,
                                                seed=noise_seed)

    Is = _center_beam_spectrum(intensities, nx_pix, ny_pix)

    if save_plot:
        fig, ax = plt.subplots()
        ax.plot(velos, Is)
        fig.savefig(os.path.join(odir, f'images/NLTE_nh3_{label}_{tag}.png'))
        plt.close(fig)

    hdu = fits.PrimaryHDU(Is)
    hdu.header.update({'CRVAL1': velos[0], 'CDELT1': velos[1] - velos[0], 'CTYPE1': 'VELO-LSR',
                        'CUNIT1': 'km/s', 'NAXIS1': len(velos), 'RESTFREQ': freq_rest, 'CRPIX1': 1})
    hdu.writeto(os.path.join(odir, f'fits/NLTE_nh3_spectrum_{label}_{tag}.fits'), overwrite=True)

    if return_cube:
        cube = _intensity_cube_to_Tmb(intensities, freq_rest, nx_pix, ny_pix)
        return velos, Is, cube
    return velos, Is


def build_point_cloud(rho_cloud, r_out, r_boundary, resolution=5, fov_pad_factor=1.0,
                       exterior_density_fraction=EXTERIOR_DENSITY_FRACTION):
    """Build the Delaunay point cloud for one sphere.

    Extracted from run_model so the mesh can be exercised without running the
    NLTE solve -- the point count must be stable across a density grid for a
    lookup table's interpolation-error map to mean anything, and that is only
    checkable if this is callable on its own.

    The mesh is constructed in DIMENSIONLESS units (unit radius, unit density)
    and scaled to physical size afterwards. Only the *shape* of the density field
    should determine mesh topology, but the remesher also responds to absolute
    scale: building directly in SI gave 53-78 interior points across a 4-dex
    density sweep (r_out spanning 1e12-1e16 m), whereas normalising gives a
    byte-identical cloud at every grid point. That reproducibility is what lets a
    lookup table's interpolation-error map measure physics instead of mesh noise.

    Returns (positions_reduced, nb_boundary), positions in physical units.
    """
    scale = float(r_out)
    r_boundary = float(r_boundary) / scale
    rho_cloud = 1.0
    r_out = 1.0

    xs = np.linspace(-r_out * 1.2, +r_out * 1.2, resolution, endpoint=True)
    ys = np.linspace(-r_out * 1.2, +r_out * 1.2, resolution, endpoint=True)
    zs = np.linspace(-r_out * 1.2, +r_out * 1.2, resolution, endpoint=True)
    Xs, Ys, Zs = np.meshgrid(xs, ys, zs)

    position = np.column_stack((Xs.ravel(), Ys.ravel(), Zs.ravel()))

    if fov_pad_factor > 1.2:
        # Coarse background-density fringe out to r_boundary, so the imager's FOV
        # (== the mesh's outer boundary) extends several beam-widths past the
        # source instead of stopping exactly at its edge. Low resolution is fine:
        # this shell is uniform low density, so it carries little information and
        # the density-aware remesher below will thin it further on its own.
        pad_res = 5
        pxs = np.linspace(-r_boundary, +r_boundary, pad_res, endpoint=True)
        PXs, PYs, PZs = np.meshgrid(pxs, pxs, pxs)
        pad_position = np.column_stack((PXs.ravel(), PYs.ravel(), PZs.ravel()))
        pad_position = pad_position[np.linalg.norm(pad_position, axis=1) > r_out * 1.2]
        position = np.vstack((position, pad_position))

    # Rough density map for remesher
    r_dist = np.linalg.norm(position, axis=1)
    # Same scaled exterior as get_properties_vectorized, so the density contrast
    # the remesher sees is identical at every point of a density grid.
    rhos_ravel = np.where(r_dist > r_out,
                           rho_cloud * exterior_density_fraction, rho_cloud)

    positions_reduced, nb_boundary = mesher.remesh_point_cloud(
        position, rhos_ravel, max_depth=5, threshold=2e-1, hullorder=3
    )

    origin = np.array([0.0, 0.0, 0.0]).T
    positions_reduced, nb_boundary = mesher.point_cloud_add_spherical_inner_boundary(
        positions_reduced, nb_boundary, 0.01 * r_out, healpy_order=3, origin=origin
    )
    positions_reduced, nb_boundary = mesher.point_cloud_add_spherical_outer_boundary(
        positions_reduced, nb_boundary, r_boundary, healpy_order=3, origin=origin
    )
    npoints = len(positions_reduced)
    return positions_reduced * scale, nb_boundary


def run_model(wdir, odir, XNH3=1e-7, numberdensity=1e8, vturb=100, T_cloud=35, max_NLTE=20, radius_sphere=1e16,
              image_lines=None, fov_pad_factor=1.0, nx_pix=16, ny_pix=16, resolution=10,
              nrays=48, return_cubes=False,
              save_image_fits=False, noise_rms_K=None, noise_seed=None):
    """
    fov_pad_factor: ratio of the CMB (outer) boundary radius to r_out (the emitting
    sphere's own radius). Default 1.0 reproduces the original behaviour exactly:
    boundary at r_out, no sky padding -- this is what Magritte's imager then uses
    as the image field of view (see image.cpp set_coordinates_projection_surface),
    so an unpadded image is exactly source-diameter-wide with no room for a beam
    convolution kernel to see real background. Pass e.g. 5.0 to place the boundary
    (and a coarse background-density fringe) 5x further out, giving a convolution
    real sky to sample instead of replicating source-edge brightness.
    """

    model_file = os.path.join(wdir, f'model_files/NLTE_analytic_sphere_nh3_{XNH3}_{numberdensity:.2e}_{radius_sphere:.2e}_{vturb}_{T_cloud}_pad{fov_pad_factor}_res{resolution}_nr{nrays}.hdf5')
    lamda_file = os.path.join(wdir, 'p-nh3@loreau.dat.txt')

    m_H2 = 2.01588 * constants.u.si.value
    rho_cloud = numberdensity * 1.0E6 * m_H2
    r_out = radius_sphere / 100
    r_boundary = r_out * fov_pad_factor

    positions_reduced, nb_boundary = build_point_cloud(
        rho_cloud, r_out, r_boundary, resolution=resolution,
        fov_pad_factor=fov_pad_factor)
    npoints = len(positions_reduced)

    delaunay = Delaunay(positions_reduced)
    indptr, indices = delaunay.vertex_neighbor_vertices
    neighbors = [indices[indptr[k]:indptr[k + 1]] for k in range(npoints)]
    nbs = [n for sublist in neighbors for n in sublist]
    n_nbs = [len(sublist) for sublist in neighbors]

    # Get vectorized physical properties
    nH2, nNH3, tmp, trb, velocity = get_properties_vectorized(
        positions_reduced, rho_cloud, r_out, XNH3, T_cloud, vturb
    )
    zeros = np.zeros(npoints)

    model = magritte.Model() 
    model.parameters.set_model_name(model_file)
    model.parameters.set_dimension(3)
    model.parameters.set_npoints(npoints)
    # HEALPix nside=2 -> 48 directions. The previous hardcoded 12 (nside=1) is
    # the coarsest quadrature available, and the level populations are driven by
    # the mean intensity, an angular integral.
    model.parameters.set_nrays(nrays)
    model.parameters.set_nspecs(3)
    model.parameters.set_nlspecs(1)
    model.parameters.set_nquads(20)
    model.parameters.sum_opacity_emissivity_over_all_lines = True
    model.parameters.pop_prec = 1.0e-6

    model.geometry.points.position.set(positions_reduced)
    model.geometry.points.velocity.set(velocity)
    model.geometry.points.neighbors.set(nbs)
    model.geometry.points.n_neighbors.set(n_nbs)

    model.chemistry.species.abundance = np.column_stack((nNH3, nH2, zeros))
    model.chemistry.species.symbol = ['p-NH3', 'H2', 'e-']

    model.thermodynamics.temperature.gas.set(tmp)
    model.thermodynamics.turbulence.vturb2.set(trb)

    model.parameters.set_nboundary(nb_boundary)
    model.geometry.boundary.boundary2point.set(np.arange(nb_boundary))

    setup.set_uniform_rays(model)
    setup.set_boundary_condition_CMB(model)
    setup.set_linedata_from_LAMDA_file(model, lamda_file)
    setup.set_quadrature(model)

    max_write_attempts = 3
    write_success = False
        
    for attempt in range(1, max_write_attempts + 1):
        try:
            model.write(model_file)
            write_success = True
            break
        except Exception as e:
            if attempt < max_write_attempts:
                time.sleep(2)
            else:
                print(f"CRITICAL ERROR: Failed to write model file: {e}")
                raise

    if write_success and os.path.exists(model_file):
        try:
            with h5py.File(model_file, 'r') as f:
                pass # Check valid read
        except Exception as e:
            raise RuntimeError(f"CRITICAL ERROR: File is corrupt! {e}")
    else:
        raise FileNotFoundError("File was not found!")

    # -----------------------------
    # Execute Model
    # -----------------------------
    model = magritte.Model(model_file)
    model.compute_spectral_discretisation()
    model.compute_inverse_line_widths()
    model.compute_LTE_level_populations()
    info = model.compute_level_populations_sparse(True, max_NLTE)
    
    # -----------------------------
    # (1,1) and (2,2) processing -- filenames/behaviour unchanged from before the refactor
    # -----------------------------
    tag = f'{XNH3}_{numberdensity:.2e}_{radius_sphere:.2e}_{vturb}_{T_cloud}'
    fcen_1 = 23694494829.874
    fcen_2 = model.lines.lineProducingSpecies[0].linedata.frequency[6]

    # -----------------------------
    # Every imaged line's (velos, Is[, cube]) -- '11'/'22' always included alongside
    # any additional lines requested (e.g. (2,1), (4,4)) -- label -> rest freq [Hz].
    # A single unified loop (not '11'/'22' imaged separately, then again here) --
    # imaging is a real compute cost, imaging the same line twice would double it.
    # Nothing outside this session's own code reads this dict, so including '11'/'22'
    # here is not a compatibility break.
    # -----------------------------
    all_lines = {'11': fcen_1, '22': fcen_2, **(image_lines or {})}
    extra_spectra = {}
    for label, freq_rest in all_lines.items():
        # Distinct per-line seed so the (1,1), (2,2) and (2,1) spectra carry
        # independent noise realisations, as separate observations would, while
        # the run as a whole stays reproducible from `noise_seed`.
        line_seed = None if noise_seed is None else _noise.line_seed(noise_seed, label)
        extra_spectra[label] = _image_and_save_line(model, freq_rest, label, odir, tag,
                                                      nx_pix=nx_pix, ny_pix=ny_pix,
                                                      return_cube=return_cubes,
                                                      save_image_fits=save_image_fits,
                                                      noise_rms_K=noise_rms_K,
                                                      noise_seed=line_seed)

    #tau estimate
    # # Apply Beams
    # default_bmaj_deg = default_bmin_deg = 0.00027778 * 2      
    # add_carta_beams_to_fits(img1_fits, default_bmaj_deg, default_bmin_deg, 0.0, overwrite=True)
    # add_carta_beams_to_fits(img2_fits, default_bmaj_deg, default_bmin_deg, 0.0, overwrite=True)

    model.compute_image_optical_depth_new(0, 16, 16)

    # Extract latest optical depth image
    image = model.images[-1]

    # Optical depth array: shape (npixels, nfreqs)
    tau = np.array(image.I)

    # Image plane coordinates
    ImX = np.array(image.ImX)
    ImY = np.array(image.ImY)

    # --- 1. Find main hyperfine frequency index (global max τ) ---
    f_main = np.argmax(np.max(tau, axis=0))
    
    # --- 2. Extract τ at main hyperfine ---
    tau_main_flat = tau[:, f_main]

    # --- 3. Reshape into 2D image ---
    nx = len(np.unique(ImX))
    ny = len(np.unique(ImY))

    tau_main = tau_main_flat[120]

    return info + (tau_main, extra_spectra)

if __name__ == "__main__":
    run_model(wdir="./", odir="./", XNH3=1e-7, numberdensity=1e8, vturb=100, T_cloud=35, max_NLTE=20, radius_sphere=1e16)