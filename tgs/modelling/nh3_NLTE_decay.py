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

# -----------------------------
# Vectorized Physics Functions
# -----------------------------
def get_properties_vectorized(positions, rho_center, r_core, decay_index, XNH3, T_cloud, vturb):
    """Calculates cell properties using a core plateau + power-law envelope."""
    m_H2 = 2.01588 * constants.u.si.value
    background_density = 1e2 * 1e6 * m_H2

    # Radii of all points
    r = np.linalg.norm(positions, axis=1)
    
    # 1. Inside the core: r_eff = r_core
    # 2. Outside the core: r_eff = r
    r_eff = np.maximum(r, r_core)
    
    # Calculate density: constant inside core, decaying outside
    gas_density = rho_center * (r_eff / r_core)**(-decay_index)

    # Floor the density at the ambient background
    gas_density = np.maximum(gas_density, background_density)

    nH2 = gas_density / m_H2
    nNH3 = XNH3 * nH2

    # Temperature and Turbulence (constant arrays)
    tmp = np.full(len(positions), T_cloud, dtype=np.float64)
    trb = np.full(len(positions), (vturb / constants.c.si.value) ** 2, dtype=np.float64)

    # Velocity (Static)
    velocity = np.zeros_like(positions)

    return nH2, nNH3, tmp, trb, velocity

def add_carta_beams_to_fits(input_fits, default_bmaj_deg, default_bmin_deg, default_bpa_deg, overwrite=True):
    # [Keep existing add_carta_beams_to_fits function exactly as it was]
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


def run_model(wdir, odir, XNH3=1e-7, numberdensity=1e8, vturb=100, T_cloud=35, max_NLTE=20, decay_index=1.5, r_core_au=5000):
    
    # Updated all file naming to use decay_index instead of radius
    model_file = os.path.join(wdir, f'model_files/NLTE_analytic_sphere_nh3_{XNH3}_{numberdensity:.2e}_{decay_index:.2f}_{vturb}_{T_cloud}.hdf5')
    lamda_file = os.path.join(wdir, 'p-nh3@loreau.dat.txt')

    m_H2 = 2.01588 * constants.u.si.value
    rho_center = numberdensity * 1.0E6 * m_H2   
    background_density = 1e2 * 1e6 * m_H2
    r_core = r_core_au * constants.au.si.value

    # Calculate outer boundary (r_out) where density = 3 * background
    target_density = 3.0 * background_density
    
    if rho_center > target_density and decay_index > 0:
        r_out = r_core * (rho_center / target_density)**(1.0 / decay_index)
    else:
        # Fallback to prevent meshing errors if central density is very low
        r_out = r_core * 1.5 
        
    resolution = 10

    xs = np.linspace(-r_out * 1.2, +r_out * 1.2, resolution, endpoint=True)
    ys = np.linspace(-r_out * 1.2, +r_out * 1.2, resolution, endpoint=True)
    zs = np.linspace(-r_out * 1.2, +r_out * 1.2, resolution, endpoint=True)
    Xs, Ys, Zs = np.meshgrid(xs, ys, zs)

    position = np.column_stack((Xs.ravel(), Ys.ravel(), Zs.ravel()))

    # Rough density map for remesher
    r_dist = np.linalg.norm(position, axis=1)
    r_eff_rough = np.maximum(r_dist, r_core)
    rhos_ravel = rho_center * (r_eff_rough / r_core)**(-decay_index)
    rhos_ravel = np.maximum(rhos_ravel, background_density)

    positions_reduced, nb_boundary = mesher.remesh_point_cloud(
        position, rhos_ravel, max_depth=5, threshold=2e-1, hullorder=3
    )

    origin = np.array([0.0, 0.0, 0.0]).T
    positions_reduced, nb_boundary = mesher.point_cloud_add_spherical_inner_boundary(
        positions_reduced, nb_boundary, 0.01 * r_out, healpy_order=3, origin=origin
    )
    positions_reduced, nb_boundary = mesher.point_cloud_add_spherical_outer_boundary(
        positions_reduced, nb_boundary, r_out, healpy_order=3, origin=origin
    )
    npoints = len(positions_reduced)

    delaunay = Delaunay(positions_reduced)
    indptr, indices = delaunay.vertex_neighbor_vertices
    neighbors = [indices[indptr[k]:indptr[k + 1]] for k in range(npoints)]
    nbs = [n for sublist in neighbors for n in sublist]
    n_nbs = [len(sublist) for sublist in neighbors]

    # Get vectorized physical properties
    nH2, nNH3, tmp, trb, velocity = get_properties_vectorized(
        positions_reduced, rho_center, r_core, decay_index, XNH3, T_cloud, vturb
    )
    zeros = np.zeros(npoints)

    model = magritte.Model() 
    model.parameters.set_model_name(model_file)
    model.parameters.set_dimension(3)
    model.parameters.set_npoints(npoints)
    model.parameters.set_nrays(12 * 1 * 1)
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
    # 1st Line Processing
    # -----------------------------
    fcen_1 = 23694494829.874
    model.compute_spectral_discretisation(fcen_1 - 3000000.00, fcen_1 + 3000000.00, 500)
    model.compute_image_new(0, 16, 16)

    img1_fits = os.path.join(odir, f'fits/NLTE_nh3_image_11_{XNH3}_{numberdensity:.2e}_{decay_index:.2f}_{vturb}_{T_cloud}.fits')
    tools.save_fits(model, filename=img1_fits)

    velos1 = (np.array(model.images[-1].freqs) - fcen_1) / fcen_1 * 3e8 / 1000
    intensities1 = np.array(model.images[-1].I)[:, :]
    Is1 = (intensities1[119, :] + intensities1[120, :] + intensities1[135, :] + intensities1[136, :]) / 4

    fig, ax = plt.subplots()
    ax.plot(velos1, Is1)
    fig.savefig(os.path.join(odir, f'images/NLTE_nh3_11_{XNH3}_{numberdensity:.2e}_{decay_index:.2f}_{vturb}_{T_cloud}.png'))
    plt.close(fig)

    hdu1 = fits.PrimaryHDU(Is1)
    hdu1.header.update({'CRVAL1': velos1[0], 'CDELT1': velos1[1] - velos1[0], 'CTYPE1': 'VELO-LSR', 
                        'CUNIT1': 'km/s', 'NAXIS1': len(velos1), 'RESTFREQ': fcen_1, 'CRPIX1': 1})
    hdu1.writeto(os.path.join(odir, f'fits/NLTE_nh3_spectrum_11_{XNH3}_{numberdensity:.2e}_{decay_index:.2f}_{vturb}_{T_cloud}.fits'), overwrite=True)

    # -----------------------------
    # 2nd Line Processing
    # -----------------------------
    fcen_2 = model.lines.lineProducingSpecies[0].linedata.frequency[6]
    model.compute_spectral_discretisation(fcen_2 - 3000000.00, fcen_2 + 3000000.00, 500)
    model.compute_image_new(0, 16, 16)

    img2_fits = os.path.join(odir, f'fits/NLTE_nh3_image_22_{XNH3}_{numberdensity:.2e}_{decay_index:.2f}_{vturb}_{T_cloud}.fits')
    tools.save_fits(model, filename=img2_fits)

    velos2 = (np.array(model.images[-1].freqs) - fcen_2) / fcen_2 * 3e8 / 1000
    intensities2 = np.array(model.images[-1].I)[:, :]
    Is2 = (intensities2[119, :] + intensities2[120, :] + intensities2[135, :] + intensities2[136, :]) / 4

    fig, ax = plt.subplots()
    ax.plot(velos2, Is2)
    fig.savefig(os.path.join(odir, f'images/NLTE_nh3_22_{XNH3}_{numberdensity:.2e}_{decay_index:.2f}_{vturb}_{T_cloud}.png'))
    plt.close(fig)

    hdu2 = fits.PrimaryHDU(Is2)
    hdu2.header.update({'CRVAL1': velos2[0], 'CDELT1': velos2[1] - velos2[0], 'CTYPE1': 'VELO-LSR', 
                        'CUNIT1': 'km/s', 'NAXIS1': len(velos2), 'RESTFREQ': fcen_2, 'CRPIX1': 1})
    hdu2.writeto(os.path.join(odir, f'fits/NLTE_nh3_spectrum_22_{XNH3}_{numberdensity:.2e}_{decay_index:.2f}_{vturb}_{T_cloud}.fits'), overwrite=True)

    model.compute_image_optical_depth_new(0, 16, 16)

    # Apply Beams
    default_bmaj_deg = default_bmin_deg = 0.00027778 * 2      
    add_carta_beams_to_fits(img1_fits, default_bmaj_deg, default_bmin_deg, 0.0, overwrite=True)
    add_carta_beams_to_fits(img2_fits, default_bmaj_deg, default_bmin_deg, 0.0, overwrite=True)

    image = model.images[-1]
    tau = np.array(image.I)
    ImX = np.array(image.ImX)
    ImY = np.array(image.ImY)

    f_main = np.argmax(np.max(tau, axis=0))
    tau_main_flat = tau[:, f_main]

    nx = len(np.unique(ImX))
    ny = len(np.unique(ImY))
    tau_main = tau_main_flat[120]
    
    return info + (tau_main,)

if __name__ == "__main__":
    run_model(wdir="./", odir="./", XNH3=1e-10, numberdensity=1e6, vturb=300, T_cloud=35, max_NLTE=100, decay_index=1.5, r_core_au=5000)
    