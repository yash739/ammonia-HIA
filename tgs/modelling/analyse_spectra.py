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
# /home/yasho379/magritte_rebuilt/tgs/NLTE_nh3_spectrum_11_1e-07_100.0_50.0_normalised.fits
# /home/yasho379/magritte_rebuilt/tgs/NLTE_nh3_spectrum_11_1e-07_100.0_100.0_50.0_unnormalised.fits
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
        width = (v[-1] - v[-0]) / 40
        if number == 'one':
            centres = np.linspace((v[0] + v[-1])/2 - 10*width, (v[0] + v[-1])/2 + 10*width, 5)
        else:
            centres = np.linspace((v[0] + v[-1])/2 - 15*width, (v[0] + v[-1])/2 + 15*width, 5)
        p0 = []
        for c in centres:
            p0 += [max(amp_pk/3, 1e-3), c, width]  # ensure positive initial amplitude
            # Set bounds: offset free, amplitudes >=0, widths > 0, centers free
            lower_bounds = []
            for i in range(15):
                if i % 3 == 0:      # amplitude
                    lower_bounds.append(0.0001)
                elif i % 3 == 2:    # width (sigma)
                    lower_bounds.append(1e-6)
                else:               # center
                    lower_bounds.append(-np.inf)

            upper_bounds = [np.inf] * 15

    print(p0,lower_bounds, upper_bounds)
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

amps11=res['fit_params_11'][0:3]
#write the ratio of the amplitudes of the main and inner satellite components to a results text file
with open(f'{odir}/results/NLTE_nh3_results.csv', 'a') as f:
    f.write(f"{XNH3},{numberdensity},{vturb},{T_cloud},{amps11[0]:.3f},{amps11[1]:.3f},{amps11[2]:.3f},{amps11[3]:.3f},{amps11[4]:.3f},{amps11[4]/amps11[0]:.3f},{amps11[3]/amps11[1]:.3f},{amps11[2]/amps11[0]:.3f},{amps11[2]/amps11[1]:.3f},")

print("Results written to ", f'{odir}/results/NLTE_nh3_results.csv')