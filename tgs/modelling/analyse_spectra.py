import matplotlib.pyplot as plt                   # Plotting
import magritte.setup    as setup                   # Model setup
import magritte.core     as magritte                # Core functionality
import magritte.plot     as plot                       # Plotting
import magritte.mesher   as mesher                  # Mesher
import magritte.tools    as tools      # Save fits


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

wdir = "/home/yasho379/magritte_rebuilt/tgs"
odir= "/home/yasho379/magritte_rebuilt/output"
/home/yasho379/magritte_rebuilt/tgs/NLTE_nh3_spectrum_11_1e-07_100.0_50.0_normalised.fits
/home/yasho379/magritte_rebuilt/tgs/NLTE_nh3_spectrum_11_1e-07_100.0_100.0_50.0_unnormalised.fits
# Parse command line arguments
parser = argparse.ArgumentParser(description="Set model parameters.")
parser.add_argument("--XNH3", type=float, default=1e-8, help="Ammonia abundance (default: 1e-8)")
parser.add_argument("--numberdensity", type=float, default=5*1E6, help="Hydrogen Number Density in /cm^3(default: 5e6)")
parser.add_argument("--vturb", type=float, default=100, help="Turbulent velocity in m/s (default: 100)")
parser.add_argument("--T_cloud", type=float, default=35, help="Cloud temperature in K (default: 35)")
parser.add_argument("--max_NLTE", type=int, default=10, help="Maximum number of NLTE iterations (default: 10)")

"""
prototype command to run the script with custom parameters:
python analyse_spectra.py --XNH3 1e-7 --numberdensity 1e7 --vturb 200 --T_cloud 50 --max_NLTE 15
"""
args = parser.parse_args()

XNH3 = args.XNH3
vturb = args.vturb
T_cloud = args.T_cloud
max_NLTE_iterations = args.max_NLTE
numberdensity = args.numberdensity


import pyspeckit

# The ammonia fitting wrapper requires a dictionary specifying the transition name
# (one of the four specified below) and the filename.  Alternately, you can have the
# dictionary values be pre-loaded Spectrum instances
filenames = {'oneone':os.path.join(wdir, f'NLTE_nh3_spectrum_11_{XNH3}_{numberdensity}_{vturb}_{T_cloud}_unnormalised.fits'),
    'twotwo': os.path.join(wdir, f'NLTE_nh3_spectrum_22_{XNH3}_{numberdensity}_{vturb}_{T_cloud}_unnormalised.fits')}

# Fit the ammonia spectrum with some reasonable initial guesses.  It is
# important to crop out extraneous junk and to smooth the data to make the
# fit proceed at a reasonable pace.
spdict1,spectra1 = pyspeckit.wrappers.fitnh3.fitnh3tkin(filenames,crop=False,npeaks=2,guessline='twotwo',rebase=True,tkin=5.65,tex=4.49,column=15.5,fortho=0.3,verbose=True, smooth=False,dobaseline=True,doplot=True,fittype='ammonia')


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
plt.savefig(f'{odir}/images/NLTE_nh3_1122_{XNH3}_{numberdensity}_{vturb}_{T_cloud}.png')

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
plt.savefig(f'{odir}/images/NLTE_nh3_1122_{XNH3}_{numberdensity}_{vturb}_{T_cloud}_corrected.png')
