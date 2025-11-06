
import matplotlib.pyplot as plt                   # Plotting
import magritte.setup    as setup                   # Model setup
import magritte.core     as magritte                # Core functionality
import magritte.plot     as plot                       # Plotting
import magritte.mesher   as mesher                  # Mesher
import magritte.tools    as tools      # Save fits

# print("Threads avail:", magritte.pcmt_n_threads_avail())
# magritte.pcmt_set_n_threads_avail(8)
# print("After set:", magritte.pcmt_n_threads_avail())

import numpy             as np                      # Data structures
import warnings                                     # Hide warnings
warnings.filterwarnings('ignore')                   # especially for yt
import yt                                           # 3D plotting
import argparse                                     # Command line arguments
import os

from tqdm                import tqdm                # Progress bars
from astropy             import units, constants    # Unit conversions
from scipy.spatial       import Delaunay, cKDTree   # Finding neighbors
from scipy.signal        import savgol_filter
from yt.funcs            import mylog               # To avoid yt output 
mylog.setLevel(40)                                  # as error messages

wdir = "/home/yasho379/magritte_rebuilt/tgs/"
odir= "/home/yasho379/magritte_rebuilt/output"
# Define file names

model_file = os.path.join(wdir, 'NLTE_analytic_sphere_nh3.hdf5')   # Resulting Magritte model
lamda_file = os.path.join(wdir, 'p-nh3@loreau.dat.txt'                  )   # Line data file
bmesh_name = os.path.join(wdir, 'analytic_sphere'           )   # bachground mesh name (no extension!)

### Model parameters

# The functions below describe the disk structure, based on the Magritte application presented in [De Ceuster et al. (2019)](https://doi.org/10.1093/mnras/stz3557).
G      =           constants.G.si.value
kb     =           constants.k_B.si.value
m_H2   = 2.01588 * constants.u.si.value

# Parse command line arguments
parser = argparse.ArgumentParser(description="Set model parameters.")
parser.add_argument("--XNH3", type=float, default=1e-7, help="Ammonia abundance (default: 1e-8)")
parser.add_argument("--numberdensity", type=float, default=1e8, help="Hydrogen Number Density in /cm^3(default: 5e6)")
parser.add_argument("--vturb", type=float, default=100, help="Turbulent velocity in m/s (default: 100)")
parser.add_argument("--T_cloud", type=float, default=35, help="Cloud temperature in K (default: 35)")
parser.add_argument("--max_NLTE", type=int, default=20, help="Maximum number of NLTE iterations (default: 10)")

"""
prototype command to run the script with custom parameters:
python nh3_NLTE_sphere.py --XNH3 1e-7 --numberdensity 1e+8 --vturb 100 --T_cloud 35 --max_NLTE 20

"""
args = parser.parse_args()

XNH3 = args.XNH3
vturb = args.vturb
T_cloud = args.T_cloud
max_NLTE_iterations = args.max_NLTE
numberdensity = args.numberdensity
rho_cloud = numberdensity* 1.0E6 * m_H2   # [kg/m^3] for magritte purposes

r_out  =   5000.0 * constants.au.si.value
r_in   =    100.0 * constants.au.si.value
resolution = 20

def density(rr):
    rho_0 = rho_cloud  # central density in si units
    r_0 = r_in # inner flat part radius in AU
    alpha = 0  # power-law index
    r_max = r_out  # outer cutoff radius in AU

    # Compute radial distance
    Radius = (rr[0]**2. + rr[1]**2. + rr[2]**2.)**0.5

    # Calculate density profile
    GasDensity = rho_0 / (1.0 + (Radius / r_0)**alpha)

    # Apply cutoff beyond r_out
    if hasattr(Radius, "__len__"):
        GasDensity[Radius > r_max] = 0.0
    else:
        if Radius > r_max:
            GasDensity = 0.0

    return GasDensity

def abn_nH2(rr):
    """
    H2 number density function.
    """
    return density(rr) / (2.01588 * constants.u.si.value)

def abn_nNH3(rr):
    """
    NH3 number density function.
    """
    return XNH3 * abn_nH2(rr)


def temperature(rr):
    return T_cloud
    
        
def turbulence(rr):
    """
    !!! Peculiar Magritte thing...
    Square turbulent speed as fraction of the speed of light.
    """
    return (vturb/constants.c.si.value)**2


def velocity_f(rr):
    """
    Inward radial velocity.
    """
    x, y, z = rr[0], rr[1], rr[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    v_radial = 0   # [m/s], inward radial velocity
    if hasattr(r, "__len__"):
        v = np.zeros((3,) + r.shape)
        v[0] = v_radial * (x / r)
        v[1] = v_radial * (y / r)
        v[2] = v_radial * (z / r)
    else:
        v = np.array([v_radial * (x / r), v_radial * (y / r), v_radial * (z / r)])
    return v/ (3e8)

# Define the desired background mesh, a ($75 \times 75 \times 75$) cube.

xs = np.linspace(-r_out*1.2, +r_out*1.2, resolution, endpoint=True)
ys = np.linspace(-r_out*1.2, +r_out*1.2, resolution, endpoint=True)
zs = np.linspace(-r_out*1.2, +r_out*1.2, resolution, endpoint=True)

(Xs, Ys, Zs) = np.meshgrid(xs, ys, zs)

# Extract positions of points in background mesh
position = np.array((Xs.ravel(), Ys.ravel(), Zs.ravel())).T

# Evaluate the density on the cube
rhos      = density([Xs, Ys, Zs])
rhos_min  = np.min(rhos[rhos!=0.0])
rhos     += rhos_min

# Now we remesh this model using the new remesher
positions_reduced, nb_boundary = mesher.remesh_point_cloud(position, rhos.ravel(), max_depth=5, threshold= 2e-1, hullorder=3)

# We add a spherical inner boundary at 0.01*r_out

healpy_order = 5 #using healpy to determine where to place the 12*healpy_order**2 boundary points on the sphere
origin = np.array([0.0,0.0,0.0]).T #the origin of the inner boundary sphere
positions_reduced, nb_boundary = mesher.point_cloud_add_spherical_inner_boundary(positions_reduced, nb_boundary, 0.01*r_out, healpy_order=healpy_order, origin=origin)
print("number of points in reduced grid: ", len(positions_reduced))
print("number of boundary points: ", nb_boundary)

# We add a spherical outer boundary at r_out

healpy_order = 15 #using healpy to determine where to place the 12*healpy_order**2 boundary points on the sphere
origin = np.array([0.0,0.0,0.0]).T #the origin of the inner boundary sphere
positions_reduced, nb_boundary = mesher.point_cloud_add_spherical_outer_boundary(positions_reduced, nb_boundary, r_out, healpy_order=healpy_order, origin=origin)
print("number of points in reduced grid: ", len(positions_reduced))

npoints = len(positions_reduced)

# Extract Delaunay vertices (= Voronoi neighbors)
delaunay = Delaunay(positions_reduced)
(indptr, indices) = delaunay.vertex_neighbor_vertices
neighbors = [indices[indptr[k]:indptr[k+1]] for k in range(npoints)]
nbs       = [n for sublist in neighbors for n in sublist]
n_nbs     = [len(sublist) for sublist in neighbors]

# Convenience arrays
zeros = np.zeros(npoints)
ones  = np.ones (npoints)

# Convert model functions to arrays based the model mesh.

position = positions_reduced
velocity = np.array([velocity_f (rr) for rr in positions_reduced])
nH2      = np.array([abn_nH2    (rr) for rr in positions_reduced])
nNH3      = np.array([abn_nNH3   (rr) for rr in positions_reduced])
tmp      = np.array([temperature(rr) for rr in positions_reduced])
trb      = np.array([turbulence (rr) for rr in positions_reduced])

model = magritte.Model ()                              # Create model object

model.parameters.set_model_name         (model_file)   # Magritte model file
model.parameters.set_dimension          (3)            # This is a 3D model
model.parameters.set_npoints            (npoints)      # Number of points
model.parameters.set_nrays              (12*1*1)            # Number of rays  
model.parameters.set_nspecs             (3)            # Number of species (min. 5)
model.parameters.set_nlspecs            (1)            # Number of line species
model.parameters.set_nquads             (20)           # Number of quadrature points
model.parameters.sum_opacity_emissivity_over_all_lines = True
# model.parameters.one_line_approximation = False

model.geometry.points.position.set(position)
model.geometry.points.velocity.set(velocity)

model.geometry.points.  neighbors.set(  nbs)
model.geometry.points.n_neighbors.set(n_nbs)

model.chemistry.species.abundance = np.array((nNH3, nH2, zeros)).T
model.chemistry.species.symbol    = ['p-NH3', 'H2', 'e-']

model.thermodynamics.temperature.gas  .set(tmp)
model.thermodynamics.turbulence.vturb2.set(trb)

model.parameters.set_nboundary(nb_boundary)
model.geometry.boundary.boundary2point.set(np.arange(nb_boundary))

# direction = np.array([[+1,0,0], [-1,0,0]])            # Comment out to use all directions
# model.geometry.rays.direction.set(direction)          # Comment out to use all directions
# model.geometry.rays.weight   .set(0.5 * np.ones(2))   # Comment out to use all directions

setup.set_uniform_rays            (model)   # Uncomment to use all directions

setup.set_boundary_condition_CMB  (model)
#setup.set_linedata_from_LAMDA_file(model, lamda_file, {'considered transitions': [0,1]})  # Consider only CO lines
setup.set_linedata_from_LAMDA_file(model, lamda_file)   # Consider all transitions
setup.set_quadrature              (model)

model.write()

ds = yt.load_unstructured_mesh(
    connectivity = delaunay.simplices.astype(np.int64),
    coordinates  = delaunay.points.astype(np.float64) * 100.0, # yt expects cm not m 
    node_data    = {('connect1', 'n'): nNH3[delaunay.simplices].astype(np.float64)}
)

sl = yt.SlicePlot (ds, 'z', ('connect1', 'n'))
sl.set_cmap       (('connect1', 'n'), 'magma')
sl.zoom           (0.9)

model = magritte.Model(model_file)

model.compute_spectral_discretisation ()
model.compute_inverse_line_widths     ()
model.compute_LTE_level_populations   ()
model.compute_level_populations_sparse (True, max_NLTE_iterations) #with param args: whether to use Ng-acceleration for faster convergence (True) and max number of NLTE iterations (20)

fcen = model.lines.lineProducingSpecies[0].linedata.frequency
print(fcen)

fcen = model.lines.lineProducingSpecies[0].linedata.frequency[0]
vpix = 1e+3   # velocity pixel size [m/s] 
dd   = vpix * (model.parameters.nfreqs()-1)/2 / magritte.CC
fmax = fcen + fcen*dd
fmin = 2*fcen - fmax  # same as fmin = fcen - fcen*dd

fcen=23694494829.874
fmin=fcen- 3000000.00
fmax=fcen+ 3000000.00
# Ray orthogonal to image plane
ray_nr = 0

model.compute_spectral_discretisation (fmin, fmax, 500)#bins the frequency spectrum [fmin, fmax] into model.parameters.nfreqs bins.
# model.compute_spectral_discretisation (fmin, fmax, 31)#bins using the specified amount of frequency bins (31). Can be any integer >=1

model.compute_image_new               (ray_nr,16,16) #using a resolution of 512x512 for the image. 
#Instead of definining a ray index [0, nrays-1], you can also define a ray direction for the imager 
#model.compute_image_new              (rx, ry, rz, 512, 512)#in which (rx, ry, rz) is the (normalized) ray direction

tools.save_fits(model,filename=f'{odir}/fits/NLTE_nh3_11_{XNH3}_{numberdensity}_{vturb}_{T_cloud}.fits')

# plot.image_channel(model,[3243,3244],[1,2])
fig=plot.image_channel(model,[110,200,249,300,360],[1,5])

#frequencies and intensities can be extracted from the image object
velos = (np.array(model.images[-1].freqs) - fcen)/fcen*3e8/1000 #convert frequency to velocity in km/s
intensities = np.array(model.images[-1].I)[:,:] #intensities [pixel index, frequency index]
print(intensities.shape) #(nrays, nfreqs)
#see ImX, ImY in model.images[-1] for the pixel coordinates; will need to be multiplied with the (unit) direction vectors: image_direction_x, image_direction_y
pixel_coords_x, pixel_coords_y = np.array(model.images[-1].ImX), np.array(model.images[-1].ImY)

# Is = np.sum(intensities, axis=0) * (np.max(pixel_coords_x) - np.min(pixel_coords_x)) * (np.max(pixel_coords_y) - np.min(pixel_coords_y)) / (len(pixel_coords_x))
Is=(intensities[119,:]+intensities[120,:]+intensities[135,:]+intensities[136,:])/4

print(velos,Is)
np.size(velos), np.size(Is)

velos1=velos
plt.plot(velos1, Is)
plt.savefig(f'{odir}/images/NLTE_nh3_11_{XNH3}_{numberdensity}_{vturb}_{T_cloud}.png')

# Save the image to a FITS file conforming to radio 1D format
import astropy.io.fits as fits

hdu = fits.PrimaryHDU(Is)
print(len(velos1))
hdu.header['CRVAL1'] = velos1[0]
hdu.header['CDELT1'] = velos1[1] - velos1[0]
hdu.header['CTYPE1'] = 'VELO-LSR'
hdu.header['CUNIT1'] = 'km/s'
hdu.header['NAXIS1'] = len(velos1)
hdu.header['RESTFREQ'] = fcen  # Rest frequency of the line
hdu.header['CRPIX1'] = 1  # FITS convention for first pixel

fits_file = os.path.join(wdir, f'{odir}/fits/NLTE_nh3_spectrum_11_{XNH3}_{numberdensity:.0e}_{vturb}_{T_cloud}.fits')
hdu.writeto(fits_file, overwrite=True)

fcen = model.lines.lineProducingSpecies[0].linedata.frequency[6]
fmin=fcen- 3000000.00
fmax=fcen+ 3000000.00
# Ray orthogonal to image plane
ray_nr = 0

model.compute_spectral_discretisation (fmin, fmax,500)#bins the frequency spectrum [fmin, fmax] into model.parameters.nfreqs bins.
# model.compute_spectral_discretisation (fmin, fmax, 31)#bins using the specified amount of frequency bins (31). Can be any integer >=1

model.compute_image_new               (ray_nr, 16, 16)#using a resolution of 512x512 for the image. 
#Instead of definining a ray index [0, nrays-1], you can also define a ray direction for the imager 

fig=plot.image_channel(model,[110,200,249,300,360],[1,5])
tools.save_fits(model, filename=f'{odir}/fits/NLTE_nh3_image_22_{XNH3}_{vturb}_{T_cloud}.fits')

#frequencies and intensities can be extracted from the image object
velos = (np.array(model.images[-1].freqs) - fcen)/fcen*3e8/1000 #convert frequency to velocity in km/s
intensities = np.array(model.images[-1].I)[:,:] #intensities [pixel index, frequency index]
print(intensities.shape) #(nrays, nfreqs)
#see ImX, ImY in model.images[-1] for the pixel coordinates; will need to be multiplied with the (unit) direction vectors: image_direction_x, image_direction_y
pixel_coords_x, pixel_coords_y = np.array(model.images[-1].ImX), np.array(model.images[-1].ImY)
# Is = np.sum(intensities, axis=0) * (np.max(pixel_coords_x) - np.min(pixel_coords_x)) * (np.max(pixel_coords_y) - np.min(pixel_coords_y)) / (len(pixel_coords_x))
Is=(intensities[119,:]+intensities[120,:]+intensities[135,:]+intensities[136,:])/4

velos2=velos
print(velos2)
#shift velo to have the centre at 0
plt.plot(velos2, Is)
plt.savefig(f'{odir}/images/NLTE_nh3_22_{XNH3}_{numberdensity}_{vturb}_{T_cloud}.png')

# Save the image to a FITS file conforming to radio 1D format
import astropy.io.fits as fits

hdu = fits.PrimaryHDU(Is)

hdu.header['CRVAL1'] = velos2[0]
hdu.header['CDELT1'] = velos2[1] - velos2[0]
hdu.header['CTYPE1'] = 'VELO-LSR'
hdu.header['CUNIT1'] = 'km/s'
hdu.header['NAXIS1'] = len(velos2)
hdu.header['RESTFREQ'] = fcen  # Rest frequency of the line
hdu.header['CRPIX1'] = 1  # FITS convention for first pixel
fits_file = os.path.join(wdir, f'{odir}/fits/NLTE_nh3_spectrum_22_{XNH3}_{numberdensity:.0e}_{vturb}_{T_cloud}.fits')
hdu.writeto(fits_file, overwrite=True)

import pyspeckit
# The ammonia fitting wrapper requires a dictionary specifying the transition name
# (one of the four specified below) and the filename.  Alternately, you can have the
# dictionary values be pre-loaded Spectrum instances
filenames = {'oneone':os.path.join(odir, f'fits/NLTE_nh3_spectrum_11_{XNH3}_{numberdensity:.0e}_{vturb}_{T_cloud}.fits'),
    'twotwo': os.path.join(odir, f'fits/NLTE_nh3_spectrum_22_{XNH3}_{numberdensity:.0e}_{vturb}_{T_cloud}.fits')}

# Fit the ammonia spectrum with some reasonable initial guesses.  It is
# important to crop out extraneous junk and to smooth the data to make the
# fit proceed at a reasonable pace.
# spdict1,spectra1 = pyspeckit.wrappers.fitnh3.fitnh3tkin(filenames,crop=False,npeaks=2,guessline='twotwo',rebase=True,tkin=5.65,tex=4.49,column=15.5,fortho=0.3,verbose=True, smooth=False,dobaseline=True,doplot=True,fittype='ammonia')

from astropy import units as u
import astropy.io.fits as fits
import numpy as np

h  = 6.62607015e-34          # Planck  [J s]
k_B  = 1.380649e-23            # Boltzmann [J K⁻¹]
c  = 2.99792458e8            # speed of light [m s⁻¹]

# Load the spectra from the FITS files
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
def intensity_to_Tmb(v,I, freq):
    Tmb = (c**2 * I) / (2 * k_B * (freq*(1+v/c))**2)
    return Tmb

Tmb1 = intensity_to_Tmb(1000*velos1,spec1, freq1)
Tmb2 = intensity_to_Tmb(1000*velos2,spec2, freq2)
# now plot the results
import matplotlib.pyplot as plt
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

#subtract baseline from the spectra by subtracting a straight line fitted to the edges of the spectra

def subtract_baseline(velos, intensities, edge_fraction=0.1):
    n = len(velos)
    edge_n = int(n * edge_fraction)
    
    # Select edge points
    edge_velos = np.concatenate((velos[:edge_n], velos[-edge_n:]))
    edge_intensities = np.concatenate((intensities[:edge_n], intensities[-edge_n:]))
    
    # Fit a linear baseline to the edge points
    coeffs = np.polyfit(edge_velos, edge_intensities, 1)
    baseline = np.polyval(coeffs, velos)
    
    # Subtract the baseline from the original intensities
    corrected_intensities = intensities - baseline
    
    return corrected_intensities
Tmb1_corrected = subtract_baseline(velos1, Tmb1)
Tmb2_corrected = subtract_baseline(velos2, Tmb2)
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

"""
NH3 (1,1) and (2,2) line analysis: Gaussian fitting, τ, T_ex, T_rot
"""
import numpy as np
from scipy.optimize import curve_fit, brentq

# --------------------------- constants ---------------------------
h  = 6.62607015e-34          # Planck  [J s]
k  = 1.380649e-23            # Boltzmann [J K⁻¹]
c  = 2.99792458e8            # speed of light [m s⁻¹]
T_BG  = 2.73                 # CMB [K]
NU_11 = 23.6944955e9         # NH3 (1,1) [Hz]
NU_22 = 23.7226333e9         # NH3 (2,2) [Hz]
DELTA_E_K = 42.32            # (2,2)–(1,1) energy gap [K]

# --------------------------- models ---------------------------
def gaussian(v, amp, cen, sig):
    """Single Gaussian."""
    return amp * np.exp(-0.5 * ((v - cen)/sig)**2)

def multi_gaussian(v,*pars):
    """Sum of N Gaussians; pars = [amp1,cen1,sig1, …, ampN,cenN,sigN]."""
    n = len(pars)//3
    out = np.zeros_like(v)
    for i in range(n):
        a, c, s = pars[3*i :3*i+3]
        out += gaussian(v, a, c, s)
    return out 

# --------------------------- fitting ---------------------------
def fit_five_gaussians(v, tmb, number, p0=None):
    """Fit 5 Gaussians; returns best-fit parameters (len=15) with positive amplitudes."""
    if p0 is None:  # crude automatic seed
        idx_max = np.argmax(tmb)
        vpk, amp_pk = v[idx_max], tmb[idx_max]
        width = (v[0] - v[-1]) / 40
        centres = np.linspace((v[0] + v[-1])/2 - 10*width, (v[0] + v[-1])/2 + 10*width, 5)
        p0 = []
        for c in centres:
            p0 += [max(amp_pk/3, 1e-3), c, width]  # ensure positive initial amplitude
            # Set bounds: offset free, amplitudes >=0, widths > 0, centers free
            lower_bounds = [min(tmb), -np.inf]
            for i in range(15):
                if i % 3 == 0:      # amplitude
                    lower_bounds.append(0.0001)
                elif i % 3 == 2:    # width (sigma)
                    lower_bounds.append(1e-6)
                else:               # center
                    lower_bounds.append(-np.inf)

            upper_bounds = [np.inf] * 15

    pars, _ = curve_fit(
        lambda vv, *pp: multi_gaussian(vv, *pp),
        v, tmb, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=200000
    )
    return pars

def comp_areas(pars):
    """Return arrays of areas (K km/s) for each Gaussian component."""
    amps  = pars[0::3]
    sigs  = pars[2::3]
    return amps * sigs * np.sqrt(2*np.pi)

# ------------------------ line diagnostics -----------------------
def J_nu(T, nu):
    return (h*nu/k) / (np.exp(h*nu/(k*T)) - 1)

def tau_from_sat(Ts, Tm, a_s=0.22):
    """Solve τ from T_s/T_m ratio (inner satellite)."""
    f = lambda tau: (1 - np.exp(-a_s*tau))/(1 - np.exp(-tau)) - Ts/Tm
    return brentq(f, 1e-4, 100)

def tex_from_tau(Tmb_peak, tau, nu, method= 'brentq'):
    if method == 'brentq':
        """Solve T_ex from radiative-transfer equation."""
        Jbg = J_nu(T_BG, nu)
        f = lambda Tex: (J_nu(Tex, nu) - Jbg)*(1 - np.exp(-tau)) - Tmb_peak
        #plot f at a range of values
        import matplotlib.pyplot as plt
        Tex_vals = np.linspace(0, 100, 100)
        plt.plot(Tex_vals, f(Tex_vals), label='f(Tex)')
        plt.xlabel('T_ex (K)')
        plt.ylabel('f(T_ex)')
        plt.title('Function for T_ex')
        plt.show()
        return brentq(f, 1e-4, 100)
    elif method == 'simple':
        return Tmb_peak/(1-np.exp(-tau)) + T_BG

def T_rot(int11, int22):
    ratio = (int11/int22) * (9/5)          # statistical weight factor
    return DELTA_E_K / np.log(ratio)

# ------------------------------- main ---------------------------
def analyse_pair(v11, t11, v22, t22,
                 main_idx11=2, sat_idx11=1, a_s=0.22):
    # ----- fit spectra
    p11 = fit_five_gaussians(v11, t11,'one')
    p22 = fit_five_gaussians(v22, t22,'two')
    print(p11)
    print(p22)
    #plot the fitted spectra
    amps11  = p11[0::3]; cents11=p11[1::3];  sigs11 = p11[2::3]*1000
    amps22  = p22[0::3]; cents22=p22[1::3];  sigs22 = p22[2::3]*1000

    #plot the fitted spectra
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(v11, t11, label='NH3 (1,1) Data', color='blue')
    plt.plot(v11, multi_gaussian(v11, *p11), label='Fit (1,1)', color='orange')
    plt.title('NH3 (1,1) Spectrum with Fit')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Tmb (K)')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(v22, t22, label='NH3 (2,2) Data', color='red')
    plt.plot(v22, multi_gaussian(v22, *p22), label='Fit (2,2)', color='green')
    plt.title('NH3 (2,2) Spectrum with Fit')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Tmb (K)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{odir}/images/NLTE_nh3_1122_{XNH3}_{numberdensity:.0e}_{vturb}_{T_cloud}_fit.png')

    # ----- integrated intensities (K km/s)
    I11 = np.sum(comp_areas(p11))
    I22 = np.sum(comp_areas(p22))

    # ----- optical depth via satellite ratio
    tau_11 = tau_from_sat(amps11[sat_idx11], amps11[main_idx11], a_s)

    # ----- excitation temperature
    Tex_11 = tex_from_tau(amps11[main_idx11], tau_11, NU_11, method='brentq')

    # ----- rotational temperature
    Trot = T_rot(I11, I22)

    # ----- kinetic temperature
    Tkin = Trot /(1-(Trot/42)*np.log(1+1.1*np.exp(-16/Trot)))

    return dict(I11=I11, I22=I22, tau_11=tau_11, Tex_11=Tex_11,
                Trot=Trot, Tkin=Tkin, fit_params_11=p11, fit_params_22=p22)

res = analyse_pair(velos1, Tmb1_corrected, velos2, Tmb2_corrected,
                       main_idx11=2,
                       sat_idx11=4,
                       a_s=0.03)

print(f"∫Tmb dv (1,1): {res['I11']:.3f} K km/s")
print(f"∫Tmb dv (2,2): {res['I22']:.3f} K km/s")
print(f"τ(1,1)       : {res['tau_11']:.2f}")
print(f"T_ex (1,1)   : {res['Tex_11']:.2f} K")
print(f"T_rot        : {res['Trot']:.2f} K")
print(f"T_kin        : {res['Tkin']:.2f} K")