import matplotlib.pyplot as plt                   # Plotting
import magritte.setup    as setup                  # Model setup
import magritte.core     as magritte               # Core functionality
import magritte.plot     as plot                   # Plotting
import magritte.mesher   as mesher                 # Mesher
import magritte.tools    as tools                  # Save fits
import numpy             as np                     # Data structures
import warnings                                     # Hide warnings
import numpy as np
from copy import deepcopy
warnings.filterwarnings('ignore')                  # especially for yt
import yt                                            # 3D plotting
import os
from tqdm                import tqdm               # Progress bars
from astropy             import units, constants   # Unit conversions
from astropy.io          import fits
from scipy.spatial       import Delaunay, cKDTree  # Finding neighbors
from scipy.signal        import savgol_filter
from yt.funcs            import mylog              # To avoid yt output
mylog.setLevel(40)                                 # as error messages
import astropy.io.fits as fits                     # For FITS output


# Hardcoded working and output directories
wdir = "/home/yasho379/magritte_rebuilt/tgs/"
odir = "/home/yasho379/magritte_rebuilt/output"


def density(rr, rho_cloud, r_in, r_out):
    """
    Density profile function.
    """
    rho_0 = rho_cloud  # central density in si units
    r_0 = r_in         # inner flat part radius in AU
    alpha = 0          # power-law index
    r_max = r_out      # outer cutoff radius in AU

    # Compute radial distance
    Radius = (rr[0] ** 2. + rr[1] ** 2. + rr[2] ** 2.) ** 0.5

    # Calculate density profile
    GasDensity = rho_0 / (1.0 + (Radius / r_0) ** alpha)

    # Apply cutoff beyond r_out
    if hasattr(Radius, "__len__"):
        GasDensity[Radius > r_max] = 0.0
    else:
        if Radius > r_max:
            GasDensity = 0.0

    return GasDensity


def abn_nH2(rr, rho_cloud, r_in, r_out):
    """
    H2 number density function.
    """
    return density(rr, rho_cloud, r_in, r_out) / (2.01588 * constants.u.si.value)


def abn_nNH3(rr, XNH3, rho_cloud, r_in, r_out):
    """
    NH3 number density function.
    """
    return XNH3 * abn_nH2(rr, rho_cloud, r_in, r_out)


def temperature(rr, T_cloud):
    return T_cloud


def turbulence(rr, vturb):
    """
    !!! Peculiar Magritte thing...
    Square turbulent speed as fraction of the speed of light.
    """
    return (vturb / constants.c.si.value) ** 2


def velocity_f(rr):
    """
    Inward radial velocity.
    """
    x, y, z = rr[0], rr[1], rr[2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    v_radial = 0   # [m/s], inward radial velocity
    if hasattr(r, "__len__"):
        v = np.zeros((3,) + r.shape)
        v[0] = v_radial * (x / r)
        v[1] = v_radial * (y / r)
        v[2] = v_radial * (z / r)
    else:
        v = np.array([v_radial * (x / r), v_radial * (y / r), v_radial * (z / r)])
    return v / (3e8)

def add_carta_beams_to_fits(input_fits, default_bmaj_deg, default_bmin_deg, default_bpa_deg, overwrite=True):
        """
        Add CARTA BEAMS table to FITS cube.
        """
        # ================================
        with fits.open(input_fits, mode="readonly") as hdul:
            # Find the image HDU (typically primary)
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

            # Get cube size, expect NAXIS3 present for spectral cubes
            n_spectral = int(hdr.get("NAXIS3", 1))

            # Pull existing beam from header if present; else use defaults
            bmaj_deg = float(hdr.get("BMAJ", default_bmaj_deg))
            bmin_deg = float(hdr.get("BMIN", default_bmin_deg))
            bpa_deg  = float(hdr.get("BPA",  default_bpa_deg))

            # Ensure BUNIT is beam-based (CARTA is case-sensitive in some versions)
            hdr["BUNIT"] = ("Jy/beam", "Brightness unit")

            # Create BEAMS BINTABLE with one row per channel (constant beam)
            # Columns must be named exactly BMAJ/BMIN/BPA and are in degrees.
            col_bmaj = fits.Column(name="BMAJ", format="D", array=np.full(n_spectral, bmaj_deg, dtype=np.float64))
            col_bmin = fits.Column(name="BMIN", format="D", array=np.full(n_spectral, bmin_deg, dtype=np.float64))
            col_bpa  = fits.Column(name="BPA",  format="D", array=np.full(n_spectral, bpa_deg,  dtype=np.float64))
            beams_hdu = fits.BinTableHDU.from_columns([col_bmaj, col_bmin, col_bpa], name="BEAMS")

            # It can help to record the axes the table spans:
            beams_hdu.header["EXTNAME"] = "BEAMS"
            beams_hdu.header["TTYPE1"]  = "BMAJ"
            beams_hdu.header["TTYPE2"]  = "BMIN"
            beams_hdu.header["TTYPE3"]  = "BPA"
            beams_hdu.header["TFORM1"]  = "D"
            beams_hdu.header["TFORM2"]  = "D"
            beams_hdu.header["TFORM3"]  = "D"
            beams_hdu.header["COMMENT"] = "Restoring beam(s) for each spectral channel; units are degrees."

            # Build output HDUList:
            # - primary HDU holds the image and updated header (still keep BMAJ/BMIN/BPA for other tools)
            primary = fits.PrimaryHDU(data=data, header=hdr)
            out_hdul = fits.HDUList([primary, beams_hdu])

            out_name = input_fits.replace(".fits", "_withbeams.fits")
            out_hdul.writeto(out_name, overwrite=overwrite)

        print(f" Wrote: {out_name}")
        print("  Added BEAMS table and set BUNIT='Jy/beam'.")

def run_model(XNH3=1e-7, numberdensity=1e8, vturb=100, T_cloud=35, max_NLTE=20):
    """
    Main function to build and run the NH3 NLTE sphere model.
    """

    # Define file names
    model_file = os.path.join(wdir, 'NLTE_analytic_sphere_nh3.hdf5')
    lamda_file = os.path.join(wdir, 'p-nh3@loreau.dat.txt')
    bmesh_name = os.path.join(wdir, 'analytic_sphere')

    ### Model parameters
    G = constants.G.si.value
    kb = constants.k_B.si.value
    m_H2 = 2.01588 * constants.u.si.value

    rho_cloud = numberdensity * 1.0E6 * m_H2   # [kg/m^3] for magritte purposes
    r_out = 5000.0 * constants.au.si.value
    r_in = 100.0 * constants.au.si.value
    resolution = 20

    # Define the desired background mesh, a (75 x 75 x 75) cube.
    xs = np.linspace(-r_out * 1.2, +r_out * 1.2, resolution, endpoint=True)
    ys = np.linspace(-r_out * 1.2, +r_out * 1.2, resolution, endpoint=True)
    zs = np.linspace(-r_out * 1.2, +r_out * 1.2, resolution, endpoint=True)
    (Xs, Ys, Zs) = np.meshgrid(xs, ys, zs)

    # Extract positions of points in background mesh
    position = np.array((Xs.ravel(), Ys.ravel(), Zs.ravel())).T

    # Evaluate the density on the cube
    rhos = density([Xs, Ys, Zs], rho_cloud, r_in, r_out)
    rhos_min = np.min(rhos[rhos != 0.0])
    rhos += rhos_min

    # Now we remesh this model using the new remesher
    positions_reduced, nb_boundary = mesher.remesh_point_cloud(
        position, rhos.ravel(), max_depth=5, threshold=2e-1, hullorder=3
    )

    # We add a spherical inner boundary at 0.01*r_out
    healpy_order = 5
    origin = np.array([0.0, 0.0, 0.0]).T
    positions_reduced, nb_boundary = mesher.point_cloud_add_spherical_inner_boundary(
        positions_reduced, nb_boundary, 0.01 * r_out, healpy_order=healpy_order, origin=origin
    )
    print("number of points in reduced grid: ", len(positions_reduced))
    print("number of boundary points: ", nb_boundary)

    # We add a spherical outer boundary at r_out
    healpy_order = 15
    positions_reduced, nb_boundary = mesher.point_cloud_add_spherical_outer_boundary(
        positions_reduced, nb_boundary, r_out, healpy_order=healpy_order, origin=origin
    )
    print("number of points in reduced grid: ", len(positions_reduced))

    npoints = len(positions_reduced)

    # Extract Delaunay vertices (= Voronoi neighbors)
    delaunay = Delaunay(positions_reduced)
    (indptr, indices) = delaunay.vertex_neighbor_vertices
    neighbors = [indices[indptr[k]:indptr[k + 1]] for k in range(npoints)]
    nbs = [n for sublist in neighbors for n in sublist]
    n_nbs = [len(sublist) for sublist in neighbors]

    # Convenience arrays
    zeros = np.zeros(npoints)
    ones = np.ones(npoints)

    # Convert model functions to arrays based the model mesh.
    position = positions_reduced
    velocity = np.array([velocity_f(rr) for rr in positions_reduced])
    nH2 = np.array([abn_nH2(rr, rho_cloud, r_in, r_out) for rr in positions_reduced])
    nNH3 = np.array([abn_nNH3(rr, XNH3, rho_cloud, r_in, r_out) for rr in positions_reduced])
    tmp = np.array([temperature(rr, T_cloud) for rr in positions_reduced])
    trb = np.array([turbulence(rr, vturb) for rr in positions_reduced])

    model = magritte.Model()  # Create model object

    model.parameters.set_model_name(model_file)
    model.parameters.set_dimension(3)
    model.parameters.set_npoints(npoints)
    model.parameters.set_nrays(12 * 1 * 1)
    model.parameters.set_nspecs(3)
    model.parameters.set_nlspecs(1)
    model.parameters.set_nquads(20)
    model.parameters.sum_opacity_emissivity_over_all_lines = True

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

    model.write()

    # --- YT Visualization ---
    ds = yt.load_unstructured_mesh(
        connectivity=delaunay.simplices.astype(np.int64),
        coordinates=delaunay.points.astype(np.float64) * 100.0,
        node_data={('connect1', 'n'): nNH3[delaunay.simplices].astype(np.float64)}
    )

    sl = yt.SlicePlot(ds, 'z', ('connect1', 'n'))
    sl.set_cmap(('connect1', 'n'), 'magma')
    sl.zoom(0.9)

    # --- NLTE Computation ---
    model = magritte.Model(model_file)
    model.compute_spectral_discretisation()
    model.compute_inverse_line_widths()
    model.compute_LTE_level_populations()
    model.compute_level_populations_sparse(True, max_NLTE)

    # --- Frequency Range ---
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

    tools.save_fits(model, filename=f'{odir}/fits/NLTE_nh3_11_{XNH3}_{numberdensity}_{vturb}_{T_cloud}.fits')

    fig = plot.image_channel(model, [110, 200, 249, 300, 360], [1, 5])

    velos = (np.array(model.images[-1].freqs) - fcen) / fcen * 3e8 / 1000
    intensities = np.array(model.images[-1].I)[:, :]
    pixel_coords_x, pixel_coords_y = np.array(model.images[-1].ImX), np.array(model.images[-1].ImY)
    Is = (intensities[119, :] + intensities[120, :] + intensities[135, :] + intensities[136, :]) / 4

    velos1 = velos
    plt.plot(velos1, Is)
    plt.savefig(f'{odir}/images/NLTE_nh3_11_{XNH3}_{numberdensity}_{vturb}_{T_cloud}.png')

    # Save spectrum FITS
    hdu = fits.PrimaryHDU(Is)
    hdu.header['CRVAL1'] = velos1[0]
    hdu.header['CDELT1'] = velos1[1] - velos1[0]
    hdu.header['CTYPE1'] = 'VELO-LSR'
    hdu.header['CUNIT1'] = 'km/s'
    hdu.header['NAXIS1'] = len(velos1)
    hdu.header['RESTFREQ'] = fcen
    hdu.header['CRPIX1'] = 1

    fits_file = os.path.join(wdir, f'{odir}/fits/NLTE_nh3_spectrum_11_{XNH3}_{numberdensity:.0e}_{vturb}_{T_cloud}.fits')
    hdu.writeto(fits_file, overwrite=True)

    # --- Repeat for second line ---
    fcen = model.lines.lineProducingSpecies[0].linedata.frequency[6]
    fmin = fcen - 3000000.00
    fmax = fcen + 3000000.00
    ray_nr = 0

    model.compute_spectral_discretisation(fmin, fmax, 500)
    model.compute_image_new(ray_nr, 16, 16)

    fig = plot.image_channel(model, [110, 200, 249, 300, 360], [1, 5])
    tools.save_fits(model, filename=f'{odir}/fits/NLTE_nh3_image_22_{XNH3}_{vturb}_{T_cloud}.fits')

    velos = (np.array(model.images[-1].freqs) - fcen) / fcen * 3e8 / 1000
    intensities = np.array(model.images[-1].I)[:, :]
    Is = (intensities[119, :] + intensities[120, :] + intensities[135, :] + intensities[136, :]) / 4

    velos2 = velos
    plt.plot(velos2, Is)
    plt.savefig(f'{odir}/images/NLTE_nh3_22_{XNH3}_{numberdensity}_{vturb}_{T_cloud}.png')

    # Save second FITS
    hdu = fits.PrimaryHDU(Is)
    hdu.header['CRVAL1'] = velos2[0]
    hdu.header['CDELT1'] = velos2[1] - velos2[0]
    hdu.header['CTYPE1'] = 'VELO-LSR'
    hdu.header['CUNIT1'] = 'km/s'
    hdu.header['NAXIS1'] = len(velos2)
    hdu.header['RESTFREQ'] = fcen
    hdu.header['CRPIX1'] = 1

    fits_file = os.path.join(wdir, f'{odir}/fits/NLTE_nh3_spectrum_22_{XNH3}_{numberdensity:.0e}_{vturb}_{T_cloud}.fits')
    hdu.writeto(fits_file, overwrite=True)

    print("✅ Model run complete.")
    """
    Make a FITS cube CARTA-friendly by:
    - ensuring BUNIT='Jy/beam'
    - adding a BEAMS BINTABLE extension with columns BMAJ/BMIN/BPA (degrees)
    with one row per spectral channel (constant beam across channels).

    Input:  a 3D FITS cube with spatial WCS and spectral axis
    Output: <name>_withbeams.fits
    """
    # ======== USER SETTINGS ======== # 
    # We need to discuss how to optimize this, but I have set it to some reasonable value based on your existing header BMAJ/BMIN/BP
    default_bmaj_deg = 0.00027778 * 2      # ~2" as an example -> 2 * (1"/deg) in deg
    default_bmin_deg = 0.00027778 * 2
    default_bpa_deg  = 0.0                 # degrees (E of N)
    overwrite = True
    #usage on above two cubes
    input_fits = f'{odir}/fits/NLTE_nh3_image_11_{XNH3}_{numberdensity}_{vturb}_{T_cloud}.fits'
    add_carta_beams_to_fits(input_fits, default_bmaj_deg, default_bmin_deg, default_bpa_deg, overwrite)
    input_fits = f'{odir}/fits/NLTE_nh3_image_22_{XNH3}_{numberdensity}_{vturb}_{T_cloud}.fits'
    add_carta_beams_to_fits(input_fits, default_bmaj_deg, default_bmin_deg, default_bpa_deg, overwrite)

# Allow standalone run
if __name__ == "__main__":
    run_model()

from astropy import units as u
import astropy.io.fits as fits
import numpy as np
from scipy.optimize import curve_fit, brentq

def analyse_spectra(XNH3, numberdensity, vturb, T_cloud):
    """
    Analyse the NH3 (1,1) and (2,2) spectra produced by Magritte NLTE model.

    Preserves all original logic, comments, and math.
    Writes results to CSV with NH3 hyperfine amplitude and ratio columns.
    Saves all plots in original and subfolder locations.
    """

    # Subfolder for this model's results
    subfolder = f"X{XNH3}_n{numberdensity:.0e}_v{vturb}_T{T_cloud}"
    image_subdir = os.path.join(odir, "images", subfolder)
    os.makedirs(image_subdir, exist_ok=True)

    # Spectra FITS filenames
    filenames = {
        'oneone': os.path.join(odir, f'fits/NLTE_nh3_spectrum_11_{XNH3}_{numberdensity:.0e}_{vturb}_{T_cloud}.fits'),
        'twotwo': os.path.join(odir, f'fits/NLTE_nh3_spectrum_22_{XNH3}_{numberdensity:.0e}_{vturb}_{T_cloud}.fits')
    }

    # Load the spectra from FITS files
    spec1 = fits.getdata(filenames['oneone'])
    spec2 = fits.getdata(filenames['twotwo'])

    # Get velocity axes from FITS headers
    with fits.open(filenames['oneone']) as hdul:
        hdr1 = hdul[0].header
        velos1 = hdr1['CRVAL1'] + np.arange(hdr1['NAXIS1']) * hdr1['CDELT1']

    with fits.open(filenames['twotwo']) as hdul:
        hdr2 = hdul[0].header
        velos2 = hdr2['CRVAL1'] + np.arange(hdr2['NAXIS1']) * hdr2['CDELT1']

    # Get rest frequencies from headers
    freq1 = hdr1['RESTFREQ'] 
    freq2 = hdr2['RESTFREQ']

    # Convert intensity (assumed in W/(m^2 Hz sr)) to main beam temperature (K)
    h  = 6.62607015e-34          # Planck  [J s]
    k_B  = 1.380649e-23          # Boltzmann [J K⁻¹]
    c  = 2.99792458e8            # speed of light [m s⁻¹]

    def intensity_to_Tmb(v,I, freq):
        Tmb = (c**2 * I) / (2 * k_B * (freq*(1+v/c))**2)
        return Tmb

    Tmb1 = intensity_to_Tmb(1000*velos1, spec1, freq1)
    Tmb2 = intensity_to_Tmb(1000*velos2, spec2, freq2)

    # Plot raw spectra
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(velos1, Tmb1, label='NH3 (1,1)', color='blue')
    plt.title('NH3 (1,1) Spectrum')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Tmb (K)')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(velos2, Tmb2, label='NH3 (2,2)', color='red')
    plt.title('NH3 (2,2) Spectrum')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Tmb (K)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{odir}/images/NLTE_nh3_1122_{XNH3}_{numberdensity:.0e}_{vturb}_{T_cloud}.png')
    plt.savefig(os.path.join(image_subdir, f'NLTE_nh3_1122_{XNH3}_{numberdensity:.0e}_{vturb}_{T_cloud}.png'))
    plt.close()

    # Subtract baseline
    def subtract_baseline(velos, intensities, edge_fraction=0.1):
        n = len(velos)
        edge_n = int(n * edge_fraction)
        edge_velos = np.concatenate((velos[:edge_n], velos[-edge_n:]))
        edge_intensities = np.concatenate((intensities[:edge_n], intensities[-edge_n:]))
        coeffs = np.polyfit(edge_velos, edge_intensities, 1)
        baseline = np.polyval(coeffs, velos)
        return intensities - baseline

    Tmb1_corrected = subtract_baseline(velos1, Tmb1)
    Tmb2_corrected = subtract_baseline(velos2, Tmb2)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(velos1, Tmb1_corrected, label='NH3 (1,1) Corrected', color='blue')
    plt.title('NH3 (1,1) Spectrum after Baseline Subtraction')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Tmb (K)')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(velos2, Tmb2_corrected, label='NH3 (2,2) Corrected', color='red')
    plt.title('NH3 (2,2) Spectrum after Baseline Subtraction')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Tmb (K)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{odir}/images/NLTE_nh3_1122_{XNH3}_{numberdensity:.0e}_{vturb}_{T_cloud}_corrected.png')
    plt.savefig(os.path.join(image_subdir, f'NLTE_nh3_1122_{XNH3}_{numberdensity:.0e}_{vturb}_{T_cloud}_corrected.png'))
    plt.close()

    # ---------------------------
    # NH3 (1,1) and (2,2) analysis
    # ---------------------------
    h  = 6.62607015e-34
    k  = 1.380649e-23
    c  = 2.99792458e8
    T_BG  = 2.73
    NU_11 = 23.6944955e9
    NU_22 = 23.7226333e9
    DELTA_E_K = 42.32

    def gaussian(v, amp, cen, sig):
        return amp * np.exp(-0.5 * ((v - cen)/sig)**2)

    def multi_gaussian(v,*pars):
        n = len(pars)//3
        out = np.zeros_like(v)
        for i in range(n):
            a, c, s = pars[3*i :3*i+3]
            out += gaussian(v, a, c, s)
        return out

    def fit_five_gaussians(v, tmb, number, p0=None):
        if p0 is None:
            idx_max = np.argmax(tmb)
            vpk, amp_pk = v[idx_max], tmb[idx_max]
            width = (v[-1] - v[0]) / 40
            if number == 'one':
                centres = np.linspace((v[0] + v[-1])/2 - 10*width, (v[0] + v[-1])/2 + 10*width, 5)
            else:
                centres = np.linspace((v[0] + v[-1])/2 - 15*width, (v[0] + v[-1])/2 + 15*width, 5)
            p0 = []
            for c_ in centres:
                p0 += [max(amp_pk/3, 1e-3), c_, width]
            lower_bounds = []
            for i in range(15):
                if i % 3 == 0:
                    lower_bounds.append(0.0001)
                elif i % 3 == 2:
                    lower_bounds.append(1e-6)
                else:
                    lower_bounds.append(-np.inf)
            upper_bounds = [np.inf] * 15
        pars, _ = curve_fit(lambda vv, *pp: multi_gaussian(vv, *pp),
                            v, tmb, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=200000)
        return pars

    def comp_areas(pars):
        amps  = pars[0::3]
        sigs  = pars[2::3]
        return amps * sigs * np.sqrt(2*np.pi)

    def J_nu(T, nu):
        return (h*nu/k) / (np.exp(h*nu/(k*T)) - 1)

    def tau_from_sat(Ts, Tm, a_s=0.22):
        f = lambda tau: (1 - np.exp(-a_s*tau))/(1 - np.exp(-tau)) - Ts/Tm
        return brentq(f, 1e-4, 100)

    def tex_from_tau(Tmb_peak, tau, nu):
        Jbg = J_nu(T_BG, nu)
        f = lambda Tex: (J_nu(Tex, nu) - Jbg)*(1 - np.exp(-tau)) - Tmb_peak
        return brentq(f, 1e-4, 100)

    def T_rot(int11, int22):
        ratio = (int11/int22) * (9/5)
        return DELTA_E_K / np.log(ratio)

    def analyse_pair(v11, t11, v22, t22,
                     main_idx11=2, sat_idx11=4, a_s=0.03):
        p11 = fit_five_gaussians(v11, t11, 'one')
        p22 = fit_five_gaussians(v22, t22, 'two')

        amps11  = p11[0::3]
        amps22  = p22[0::3]

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(v11, t11, label='NH3 (1,1) Data', color='blue')
        plt.plot(v11, multi_gaussian(v11, *p11), label='Fit (1,1)', color='orange')
        plt.subplot(2, 1, 2)
        plt.plot(v22, t22, label='NH3 (2,2) Data', color='red')
        plt.plot(v22, multi_gaussian(v22, *p22), label='Fit (2,2)', color='green')
        plt.tight_layout()
        plt.savefig(f'{odir}/images/NLTE_nh3_1122_{XNH3}_{numberdensity:.0e}_{vturb}_{T_cloud}_fit.png')
        plt.savefig(os.path.join(image_subdir, f'NLTE_nh3_1122_{XNH3}_{numberdensity:.0e}_{vturb}_{T_cloud}_fit.png'))
        plt.close()

        I11 = np.sum(comp_areas(p11))
        I22 = np.sum(comp_areas(p22))
        tau_11 = tau_from_sat(amps11[sat_idx11], amps11[main_idx11], a_s)
        Tex_11 = tex_from_tau(amps11[main_idx11], tau_11, NU_11)
        Trot = T_rot(I11, I22)
        Tkin = Trot /(1-(Trot/42)*np.log(1+1.1*np.exp(-16/Trot)))
        return dict(I11=I11, I22=I22, tau_11=tau_11, Tex_11=Tex_11,
                    Trot=Trot, Tkin=Tkin, fit_params_11=p11, fit_params_22=p22)

    res = analyse_pair(velos1, Tmb1_corrected, velos2, Tmb2_corrected)

    amps11 = res['fit_params_11'][0:15:3]

    results_file = f'{odir}/results/NLTE_nh3_results.csv'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    header = (
        "XNH3,numberdensity,vturb,T_cloud,"
        "A_OSL,A_ISL,A_MAIN,A_ISH,A_OSH,"
        "R_OSH_MAIN,R_ISH_ISL,R_ISL_MAIN,R_ISL_ISH\n"
    )

    if not os.path.exists(results_file) or os.stat(results_file).st_size == 0:
        with open(results_file, 'w') as f:
            f.write(header)

    with open(results_file, 'a') as f:
        f.write(
            f"{XNH3},{numberdensity},{vturb},{T_cloud},"
            f"{amps11[0]:.3f},{amps11[1]:.3f},{amps11[2]:.3f},{amps11[3]:.3f},{amps11[4]:.3f},"
            f"{amps11[4]/amps11[2]:.3f},{amps11[3]/amps11[1]:.3f},{amps11[1]/amps11[2]:.3f},{amps11[1]/amps11[3]:.3f}\n"
        )

    print(f"Results written to {results_file}")