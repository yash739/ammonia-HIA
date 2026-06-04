import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit

# -----------------------------
# Module-Level Constants
# -----------------------------
h = 6.62607015e-34          
k_B = 1.380649e-23          
c = 2.99792458e8            
T_BG = 2.73

# -----------------------------
# Fast, Vectorized Helper Functions
# -----------------------------
def intensity_to_Tmb(v, I, freq):
    return (c**2 * I) / (2 * k_B * (freq * (1 + v / c))**2)

def escape_probability(vturb, radius_sphere, nH2, XNH3):
    return nH2 * XNH3 * radius_sphere / vturb 

def subtract_baseline(velos, intensities, edge_fraction=0.1):
    n = len(velos)
    edge_n = max(1, int(n * edge_fraction)) # Ensure at least 1 pixel is grabbed
    edge_velos = np.concatenate((velos[:edge_n], velos[-edge_n:]))
    edge_intensities = np.concatenate((intensities[:edge_n], intensities[-edge_n:]))
    coeffs = np.polyfit(edge_velos, edge_intensities, 1)
    return intensities - np.polyval(coeffs, velos)

def multi_gaussian(v, *pars):
    """
    Vectorized multi-gaussian evaluation for blazing-fast curve fitting.
    Replaces the slow Python 'for' loop.
    """
    pars = np.array(pars).reshape(-1, 3)
    amps = pars[:, 0, np.newaxis]
    cens = pars[:, 1, np.newaxis]
    sigs = pars[:, 2, np.newaxis]
    return np.sum(amps * np.exp(-0.5 * ((v - cens) / sigs)**2), axis=0)

def fit_five_gaussians(v, tmb, number):
    idx_max = np.argmax(tmb)
    amp_pk = tmb[idx_max]
    width = (v[0] - v[-1]) / 40.0
    
    multiplier = 10 if number == 'one' else 15
    centres = np.linspace((v[0] + v[-1])/2 - multiplier*width, (v[0] + v[-1])/2 + multiplier*width, 5)
    
    p0 = []
    for c_ in centres:
        p0.extend([max(amp_pk/3, 1e-3), c_, width])
        
    lower_bounds = []
    for i in range(15):
        if i % 3 == 0:
            lower_bounds.append(1e-4)   # Amplitude > 0
        elif i % 3 == 2:
            lower_bounds.append(1e-6)   # Sigma > 0
        else:
            lower_bounds.append(-np.inf) # Center unbound
            
    upper_bounds = [np.inf] * 15
    
    # Dropped maxfev to 50000. If it hasn't converged by then, it's spinning its wheels.
    pars, _ = curve_fit(multi_gaussian, v, tmb, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=50000)
    return pars

# -----------------------------
# Main Analysis Function
# -----------------------------
def analyse_spectra(odir, XNH3, numberdensity, vturb, T_cloud, decay_index, max_NLTE=100, save_plots=False):
    """
    Analyse the NH3 (1,1) and (2,2) spectra produced by Magritte NLTE model.
    Returns a dictionary of extracted parameters to be written safely by the main thread.
    """
    try:
        filenames = {
            'oneone': os.path.join(odir, f'fits/NLTE_nh3_spectrum_11_{XNH3}_{numberdensity:.2e}_{decay_index:.2f}_{vturb}_{T_cloud}.fits'),
            'twotwo': os.path.join(odir, f'fits/NLTE_nh3_spectrum_22_{XNH3}_{numberdensity:.2e}_{decay_index:.2f}_{vturb}_{T_cloud}.fits')
        }

        # Open FITS files ONCE to get both data and headers
        with fits.open(filenames['oneone']) as hdul1:
            spec1 = hdul1[0].data
            hdr1 = hdul1[0].header
            velos1 = (hdr1['CRVAL1'] + np.arange(hdr1['NAXIS1']) * hdr1['CDELT1'])[::-1]
            freq1 = hdr1['RESTFREQ']

        with fits.open(filenames['twotwo']) as hdul2:
            spec2 = hdul2[0].data
            hdr2 = hdul2[0].header
            velos2 = (hdr2['CRVAL1'] + np.arange(hdr2['NAXIS1']) * hdr2['CDELT1'])[::-1]
            freq2 = hdr2['RESTFREQ']

        # Convert to Tmb and Baseline Subtract
        Tmb1_corrected = subtract_baseline(velos1, intensity_to_Tmb(1000 * velos1, spec1, freq1))
        Tmb2_corrected = subtract_baseline(velos2, intensity_to_Tmb(1000 * velos2, spec2, freq2))

        # Perform curve fitting
        p11 = fit_five_gaussians(velos1, Tmb1_corrected, 'one')
        p22 = fit_five_gaussians(velos2, Tmb2_corrected, 'two')

        # -----------------------------
        # Conditional Plotting (For massive speedups)
        # -----------------------------
        if save_plots:
            subfolder = f"X{XNH3}_n{numberdensity:.2e}_{decay_index:.2f}_v{vturb}_T{T_cloud}"
            image_subdir = os.path.join(odir, "images", subfolder)
            os.makedirs(image_subdir, exist_ok=True)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
            ax1.plot(velos1, Tmb1_corrected, label='NH3 (1,1) Data', color='blue')
            ax1.plot(velos1, multi_gaussian(velos1, *p11), label='Fit (1,1)', color='orange')
            ax1.set_title('NH3 (1,1) Fit')
            ax1.legend()

            ax2.plot(velos2, Tmb2_corrected, label='NH3 (2,2) Data', color='red')
            ax2.plot(velos2, multi_gaussian(velos2, *p22), label='Fit (2,2)', color='green')
            ax2.set_title('NH3 (2,2) Fit')
            ax2.legend()

            plt.tight_layout()
            fig.savefig(os.path.join(image_subdir, f'NLTE_nh3_1122_{XNH3}_{numberdensity:.2e}_{decay_index:.2f}_{vturb}_{T_cloud}_fit.png'))
            plt.close(fig) # Critical to prevent memory leaks

        # Extract Amplitudes
        amps11 = p11[0:15:3]
        
        # Verify valid data
        if len(amps11) != 5:
            raise RuntimeError("Expected 5 hyperfine amplitudes.")
        if np.isclose(amps11[2], 0):
            print(f"WARNING: Main HF component amplitude is zero for T={T_cloud}, n={numberdensity:.2e}. Ratios may be invalid.")

        # Delete the FITS files after analysis is complete
        for filename in filenames.values():
            if os.path.exists(filename):
                os.remove(filename)
                
        # Return the payload dictionary to the main thread delivery system
        return {
            'A_10': amps11[0], 'A_21': amps11[1], 'A_MAIN': amps11[2], 'A_12': amps11[3], 'A_01': amps11[4],
            'R_01_MAIN': amps11[4]/amps11[2] if amps11[2] != 0 else np.nan,
            'R_10_MAIN': amps11[0]/amps11[2] if amps11[2] != 0 else np.nan,
            'R_21_MAIN': amps11[1]/amps11[2] if amps11[2] != 0 else np.nan,
            'R_12_MAIN': amps11[3]/amps11[2] if amps11[2] != 0 else np.nan
        }

    except Exception as e:
        raise RuntimeError(f"Spectral analysis failed: {e}")

if __name__ == "__main__":
    # Example usage for a single set of parameters (for testing)
    odir = "/home/yasho379/magritte_rebuilt/output_test_decaying/"
    from nh3_NLTE_decay import run_model
    info =  run_model(
        wdir="/home/yasho379/magritte_rebuilt/tgs/",
        odir=odir,
        XNH3=1e-8,
        numberdensity=1e6,
        vturb=100.0,
        decay_index=1.5,
        T_cloud=18.0,
        max_NLTE=20)
    result = analyse_spectra(odir, XNH3=1e-8, numberdensity=1e6, vturb=100, T_cloud=18.0, decay_index=1.5, save_plots=True)
    print(result)