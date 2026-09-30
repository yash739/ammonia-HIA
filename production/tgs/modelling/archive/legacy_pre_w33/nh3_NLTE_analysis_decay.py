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

def subtract_baseline(velos, intensities, edge_fraction=0.05):
    n = len(velos)
    edge_n = max(1, int(n * edge_fraction))
    edge_velos = np.concatenate((velos[:edge_n], velos[-edge_n:]))
    edge_intensities = np.concatenate((intensities[:edge_n], intensities[-edge_n:]))
    coeffs = np.polyfit(edge_velos, edge_intensities, 1)
    return intensities - np.polyval(coeffs, velos)

def multi_gaussian(v, *pars):
    """Fallback structural evaluator preserving the 15-parameter flat signature."""
    pars = np.array(pars).reshape(-1, 3)
    amps = pars[:, 0, np.newaxis]
    cens = pars[:, 1, np.newaxis]
    sigs = pars[:, 2, np.newaxis]
    return np.sum(amps * np.exp(-0.5 * ((v - cens) / sigs)**2), axis=0)

def fit_five_gaussians(v, tmb, number):
    """
    FIX 5: Enforces a physically tied forward model using laboratory hyperfine structural 
    offsets to prevent self-absorbed profiles from tracking as detached false emission peaks.
    """
    if number == 'one':
        # NH3 (1,1) rigid laboratory offsets in km/s relative to center
        offsets = np.array([-19.34, -7.46, 0.0, 7.46, 19.34])
    else:
        # NH3 (2,2) rigid laboratory offsets in km/s relative to center
        offsets = np.array([-26.03, -16.41, 0.0, 16.41, 26.03])
        
    def tied_hyperfine_model(vel, v0, sigma, a0, a1, a2, a3, a4):
        amps = [a0, a1, a2, a3, a4]
        res = np.zeros_like(vel)
        for a, off in zip(amps, offsets):
            res += a * np.exp(-0.5 * ((vel - (v0 + off)) / sigma)**2)
        return res

    idx_max = np.argmax(tmb)
    amp_pk = tmb[idx_max]
    v0_guess = velos_center = np.median(v)
    sigma_guess = 0.4 
    
    # Parameter map: [v0, sigma, a0, a1, a2, a3, a4]
    p0 = [v0_guess, sigma_guess, amp_pk/3, amp_pk/2, amp_pk, amp_pk/2, amp_pk/3]
    
    lower_bounds = [-np.inf, 1e-3, 0, 0, 0, 0, 0]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    
    popt, _ = curve_fit(tied_hyperfine_model, v, tmb, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=50000)
    
    v0_fit, sigma_fit = popt[0], popt[1]
    amps_fit = popt[2:]
    
    # Reconstruct the 15-element list to seamlessly feed downstream code blocks
    flat_pars = []
    for i in range(5):
        flat_pars.extend([amps_fit[i], v0_fit + offsets[i], sigma_fit])
        
    return np.array(flat_pars)

# -----------------------------
# Main Analysis Function
# -----------------------------
def analyse_spectra(odir, vturb=100, max_NLTE=20, 
                    n0=2.1e6, r0_n_arcsec=14.0, alpha_n=2.5,
                    X0=8e-9, alpha_X=0.16,
                    T_out=12.0, T_in=5.5, r0_T_arcsec=18.0,
                    distance_pc=140.0, save_plots=False):
    try:
        filenames = {
            'oneone': os.path.join(odir, f'fits/NLTE_nh3_spectrum_11_{n0:.2e}_{alpha_n:.2f}_{vturb}.fits'),
            'twotwo': os.path.join(odir, f'fits/NLTE_nh3_spectrum_22_{n0:.2e}_{alpha_n:.2f}_{vturb}.fits')
        }

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

        Tmb1_corrected = subtract_baseline(velos1, intensity_to_Tmb(1000 * velos1, spec1, freq1))
        Tmb2_corrected = subtract_baseline(velos2, intensity_to_Tmb(1000 * velos2, spec2, freq2))

        p11 = fit_five_gaussians(velos1, Tmb1_corrected, 'one')
        p22 = fit_five_gaussians(velos2, Tmb2_corrected, 'two')

        if save_plots:
            subfolder = f"X{X0:.2e}_n{n0:.2e}_{alpha_n:.2f}_v{vturb}_T{T_out:.1f}"
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
            fig.savefig(os.path.join(image_subdir, f'NLTE_nh3_1122_{n0:.2e}_{alpha_n:.2f}_{vturb}_fit.png'))
            plt.close(fig)

        amps11 = p11[0:15:3]
        
        if len(amps11) != 5:
            raise RuntimeError("Expected 5 hyperfine amplitudes.")
        if np.isclose(amps11[2], 0):
            print(f"WARNING: Main HF component amplitude is zero for T_out={T_out}, n0={n0:.2e}.")

        for filename in filenames.values():
            if os.path.exists(filename):
                os.remove(filename)
                
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
    from nh3_NLTE_decay import run_model
    
    run_model(
        wdir="./", 
        odir="./", 
        vturb=300, 
        max_NLTE=100,
        n0=2.1e6, 
        r0_n_arcsec=14.0, 
        alpha_n=2.5,
        X0=8e-9, 
        alpha_X=0.16,
        T_out=12.0, 
        T_in=5.5, 
        r0_T_arcsec=18.0,
        distance_pc=140.0
    )
    result = analyse_spectra(
        odir="./", 
        vturb=300, 
        max_NLTE=100,
        n0=2.1e6, 
        r0_n_arcsec=14.0, 
        alpha_n=2.5,
        X0=8e-9, 
        alpha_X=0.16,
        T_out=12.0, 
        T_in=5.5, 
        r0_T_arcsec=18.0,
        distance_pc=140.0,
        save_plots=True
    )
    print(result)