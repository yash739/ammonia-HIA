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

from tqdm import tqdm
from astropy import units, constants
from astropy.io import fits
from scipy.spatial import Delaunay, cKDTree
from scipy.signal import savgol_filter

def density(rr, rho_cloud, r_in, r_out):
    """
    Calculates gas density. Automatically handles both scalar and array inputs.
    """
    # Calculate radial distance from origin
    radius = np.sqrt(rr[0]**2 + rr[1]**2 + rr[2]**2)
    
    # Calculate the background density constant once
    m_H2 = 2.01588 * constants.u.si.value
    background_density = 1e2 * 1e6 * m_H2
    
    # np.where(condition, true_value, false_value)
    # If radius > r_out, use background_density. Otherwise, use rho_cloud.
    gas_density = np.where(radius > r_out, background_density, rho_cloud)
    
    return gas_density

def abn_nH2(rr, rho_cloud, r_in, r_out):
    """
    Calculates H2 number density.
    """
    m_H2 = 2.01588 * constants.u.si.value
    return density(rr, rho_cloud, r_in, r_out) / m_H2

def abn_nNH3(rr, XNH3, rho_cloud, r_in, r_out):
    return XNH3 * abn_nH2(rr, rho_cloud, r_in, r_out)


def temperature(rr, T_cloud):
    return T_cloud


def turbulence(rr, vturb):
    return (vturb / constants.c.si.value) ** 2


def velocity_f(rr):
    x, y, z = rr[0], rr[1], rr[2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    v_radial = 0   
    if hasattr(r, "__len__"):
        v = np.zeros((3,) + r.shape)
        v[0] = v_radial * (x / r)
        v[1] = v_radial * (y / r)
        v[2] = v_radial * (z / r)
    else:
        v = np.array([v_radial * (x / r), v_radial * (y / r), v_radial * (z / r)])
    return v / (3e8)

def add_carta_beams_to_fits(input_fits, default_bmaj_deg, default_bmin_deg, default_bpa_deg, overwrite=True):
    with fits.open(input_fits, mode="readonly") as hdul:
        img_hdu = None
        for h in hdul:
            if isinstance(h, fits.PrimaryHDU) or isinstance(h, fits.ImageHDU):
                if h.data is not None and h.header.get("NAXIS", 0) >= 2:
                    img_hdu = h
                    break
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

        beams_hdu.header["EXTNAME"] = "BEAMS"
        beams_hdu.header["TTYPE1"]  = "BMAJ"
        beams_hdu.header["TTYPE2"]  = "BMIN"
        beams_hdu.header["TTYPE3"]  = "BPA"
        beams_hdu.header["TFORM1"]  = "D"
        beams_hdu.header["TFORM2"]  = "D"
        beams_hdu.header["TFORM3"]  = "D"
        beams_hdu.header["COMMENT"] = "Restoring beam(s) for each spectral channel; units are degrees."

        primary = fits.PrimaryHDU(data=data, header=hdr)
        out_hdul = fits.HDUList([primary, beams_hdu])

        out_name = input_fits.replace(".fits", "_withbeams.fits")
        out_hdul.writeto(out_name, overwrite=overwrite)

        if overwrite:
            os.remove(input_fits)
        
    print(f" Wrote: {out_name}")
    print("  Added BEAMS table and set BUNIT='Jy/beam'.")

def run_model(wdir, odir, XNH3=1e-7, numberdensity=1e8, vturb=100, T_cloud=35, max_NLTE=20, radius_sphere=1e16):
    """
    Main function to build and run the NH3 NLTE sphere model.
    """

    model_file = os.path.join(wdir, 'NLTE_analytic_sphere_nh3.hdf5')
    lamda_file = os.path.join(wdir, 'p-nh3@loreau.dat.txt')

    G = constants.G.si.value
    kb = constants.k_B.si.value
    m_H2 = 2.01588 * constants.u.si.value

    rho_cloud = numberdensity * 1.0E6 * m_H2   
    r_out = radius_sphere/100  
    r_in = 0 * constants.au.si.value
    resolution = 4

    xs = np.linspace(-r_out * 1.2, +r_out * 1.2, resolution, endpoint=True)
    ys = np.linspace(-r_out * 1.2, +r_out * 1.2, resolution, endpoint=True)
    zs = np.linspace(-r_out * 1.2, +r_out * 1.2, resolution, endpoint=True)
    (Xs, Ys, Zs) = np.meshgrid(xs, ys, zs)

    position = np.array((Xs.ravel(), Ys.ravel(), Zs.ravel())).T

    rhos = density([Xs, Ys, Zs], rho_cloud, r_in, r_out)

    positions_reduced, nb_boundary = mesher.remesh_point_cloud(
        position, rhos.ravel(), max_depth=5, threshold=2e-1, hullorder=3
    )

    healpy_order = 5
    origin = np.array([0.0, 0.0, 0.0]).T
    positions_reduced, nb_boundary = mesher.point_cloud_add_spherical_inner_boundary(
        positions_reduced, nb_boundary, 0.01 * r_out, healpy_order=healpy_order, origin=origin
    )
    print("number of points in reduced grid: ", len(positions_reduced))
    print("number of boundary points: ", nb_boundary)

    healpy_order = 15
    positions_reduced, nb_boundary = mesher.point_cloud_add_spherical_outer_boundary(
        positions_reduced, nb_boundary, r_out, healpy_order=healpy_order, origin=origin
    )
    print("number of points in reduced grid: ", len(positions_reduced))

    npoints = len(positions_reduced)

    delaunay = Delaunay(positions_reduced)
    (indptr, indices) = delaunay.vertex_neighbor_vertices
    neighbors = [indices[indptr[k]:indptr[k + 1]] for k in range(npoints)]
    nbs = [n for sublist in neighbors for n in sublist]
    n_nbs = [len(sublist) for sublist in neighbors]

    zeros = np.zeros(npoints)
    ones = np.ones(npoints)

    position = positions_reduced
    velocity = np.array([velocity_f(rr) for rr in positions_reduced])
    nH2 = np.array([abn_nH2(rr, rho_cloud, r_in, r_out) for rr in positions_reduced])
    nNH3 = np.array([abn_nNH3(rr, XNH3, rho_cloud, r_in, r_out) for rr in positions_reduced])
    tmp = np.array([temperature(rr, T_cloud) for rr in positions_reduced])
    trb = np.array([turbulence(rr, vturb) for rr in positions_reduced])

    model = magritte.Model() 

    model.parameters.set_model_name(model_file)
    model.parameters.set_dimension(3)
    model.parameters.set_npoints(npoints)
    model.parameters.set_nrays(12 * 2 * 2)
    model.parameters.set_nspecs(3)
    model.parameters.set_nlspecs(1)
    model.parameters.set_nquads(20)
    model.parameters.sum_opacity_emissivity_over_all_lines = True

    print(model.parameters.sum_opacity_emissivity_over_all_lines)

    model.geometry.points.position.set(position)
    model.geometry.points.velocity.set(velocity)
    model.geometry.points.neighbors.set(nbs)
    model.geometry.points.n_neighbors.set(n_nbs)

    model.chemistry.species.abundance = np.array((nNH3, nH2, zeros)).T
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
            print(f" Wrote model file: {model_file}")
            write_success = True
            break
        except Exception as e:
            print(f"Write attempt {attempt}/{max_write_attempts} failed: {e}")
            if attempt < max_write_attempts:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print("CRITICAL ERROR: Failed to write model file after all attempts.")
                exit(1)

    if write_success and os.path.exists(model_file):
        print(f"File exists. Size: {os.path.getsize(model_file) / (1024*1024):.2f} MB")
        try:
            with h5py.File(model_file, 'r') as f:
                print("Keys in file:", list(f.keys()))
                print("File is VALID HDF5.")
        except Exception as e:
            print(f"CRITICAL ERROR: File is corrupt! {e}")
            exit(1)
    else:
        print("File was not found!")
        exit(1)

    model = magritte.Model(model_file)
    model.compute_spectral_discretisation()
    model.compute_inverse_line_widths()
    model.compute_LTE_level_populations()
    info = model.compute_level_populations_sparse(True, max_NLTE)
    
    fcen = model.lines.lineProducingSpecies[0].linedata.frequency[0]
    vpix = 1e+3
    dd = vpix * (model.parameters.nfreqs() - 1) / 2 / magritte.CC
    fmax = fcen + fcen * dd
    fmin = 2 * fcen - fmax

    fcen = 23694494829.874
    fmin = fcen - 3000000.00
    fmax = fcen + 3000000.00
    ray_nr = 0

    model.compute_spectral_discretisation(fmin, fmax, 500)
    model.compute_image_new(ray_nr, 16, 16)

    tools.save_fits(model, filename=os.path.join(odir, f'fits/NLTE_nh3_image_11_{XNH3}_{numberdensity:.2e}_{radius_sphere:.2e}_{vturb}_{T_cloud}.fits'))

    fig = plot.image_channel(model, [110, 200, 249, 300, 360], [1, 5])

    velos = (np.array(model.images[-1].freqs) - fcen) / fcen * 3e8 / 1000
    intensities = np.array(model.images[-1].I)[:, :]
    pixel_coords_x, pixel_coords_y = np.array(model.images[-1].ImX), np.array(model.images[-1].ImY)
    Is = (intensities[119, :] + intensities[120, :] + intensities[135, :] + intensities[136, :]) / 4

    velos1 = velos
    plt.plot(velos1, Is)
    plt.savefig(os.path.join(odir, f'images/NLTE_nh3_11_{XNH3}_{numberdensity:.2e}_{radius_sphere:.2e}_{vturb}_{T_cloud}.png'))

    hdu = fits.PrimaryHDU(Is)
    hdu.header['CRVAL1'] = velos1[0]
    hdu.header['CDELT1'] = velos1[1] - velos1[0]
    hdu.header['CTYPE1'] = 'VELO-LSR'
    hdu.header['CUNIT1'] = 'km/s'
    hdu.header['NAXIS1'] = len(velos1)
    hdu.header['RESTFREQ'] = fcen
    hdu.header['CRPIX1'] = 1

    fits_file = os.path.join(odir, f'fits/NLTE_nh3_spectrum_11_{XNH3}_{numberdensity:.2e}_{radius_sphere:.2e}_{vturb}_{T_cloud}.fits')
    hdu.writeto(fits_file, overwrite=True)

    fcen = model.lines.lineProducingSpecies[0].linedata.frequency[6]
    fmin = fcen - 3000000.00
    fmax = fcen + 3000000.00
    ray_nr = 0

    model.compute_spectral_discretisation(fmin, fmax, 500)
    model.compute_image_new(ray_nr, 16, 16)

    fig = plot.image_channel(model, [110, 200, 249, 300, 360], [1, 5])
    
    tools.save_fits(model, filename=os.path.join(odir, f'fits/NLTE_nh3_image_22_{XNH3}_{numberdensity:.2e}_{radius_sphere:.2e}_{vturb}_{T_cloud}.fits'))

    velos = (np.array(model.images[-1].freqs) - fcen) / fcen * 3e8 / 1000
    intensities = np.array(model.images[-1].I)[:, :]
    Is = (intensities[119, :] + intensities[120, :] + intensities[135, :] + intensities[136, :]) / 4

    velos2 = velos
    plt.plot(velos2, Is)
    plt.savefig(os.path.join(odir, f'images/NLTE_nh3_22_{XNH3}_{numberdensity:.2e}_{radius_sphere:.2e}_{vturb}_{T_cloud}.png'))

    hdu = fits.PrimaryHDU(Is)
    hdu.header['CRVAL1'] = velos2[0]
    hdu.header['CDELT1'] = velos2[1] - velos2[0]
    hdu.header['CTYPE1'] = 'VELO-LSR'
    hdu.header['CUNIT1'] = 'km/s'
    hdu.header['NAXIS1'] = len(velos2)
    hdu.header['RESTFREQ'] = fcen
    hdu.header['CRPIX1'] = 1

    fits_file = os.path.join(odir, f'fits/NLTE_nh3_spectrum_22_{XNH3}_{numberdensity:.2e}_{radius_sphere:.2e}_{vturb}_{T_cloud}.fits')
    hdu.writeto(fits_file, overwrite=True)

    print("Model run complete.")

    default_bmaj_deg = 0.00027778 * 2      
    default_bmin_deg = 0.00027778 * 2
    default_bpa_deg  = 0.0                 
    overwrite = True
    
    input_fits = os.path.join(odir, f'fits/NLTE_nh3_image_11_{XNH3}_{numberdensity:.2e}_{radius_sphere:.2e}_{vturb}_{T_cloud}.fits')
    add_carta_beams_to_fits(input_fits, default_bmaj_deg, default_bmin_deg, default_bpa_deg, overwrite)
    input_fits = os.path.join(odir, f'fits/NLTE_nh3_image_22_{XNH3}_{numberdensity:.2e}_{radius_sphere:.2e}_{vturb}_{T_cloud}.fits')
    add_carta_beams_to_fits(input_fits, default_bmaj_deg, default_bmin_deg, default_bpa_deg, overwrite)
    return info

if __name__ == "__main__":
    # Provides dummy paths so it can still run standalone if needed
    run_model(wdir="./", odir="./")