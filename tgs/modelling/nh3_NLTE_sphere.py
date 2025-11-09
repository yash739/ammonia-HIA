
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


"""
Make a FITS cube CARTA-friendly by:
 - ensuring BUNIT='Jy/beam'
 - adding a BEAMS BINTABLE extension with columns BMAJ/BMIN/BPA (degrees)
   with one row per spectral channel (constant beam across channels).

Input:  a 3D FITS cube with spatial WCS and spectral axis
Output: <name>_withbeams.fits
"""

from astropy.io import fits
import numpy as np
from copy import deepcopy

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

    print(f"✅ Wrote: {out_name}")
    print("   Added BEAMS table and set BUNIT='Jy/beam'.")

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


