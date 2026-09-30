"""Hyperfine fitting of NH3 (1,1)/(2,2) spectra.

`fit_five_gaussians` fits the five (1,1) hyperfine groups (main + four
satellites) with bounded centres and returns amplitudes in the order of
`nh3hia.hyperfine.NH3_11_COMPONENTS`, identified by fitted centre rather
than by parameter position. `analyse_spectra` runs the fit on a saved model
spectrum and forms the satellite/main ratios; `intensity_to_Tmb` and
`subtract_baseline` are the shared conversions.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit

import nh3hia.hyperfine as hf

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

def fit_five_gaussians(v, tmb, number, sigma=None, return_cov=False,
                        amp_min=1e-4, sigma_max_kms=3.0):
    """Fit the five hyperfine groups with centres seeded at their true offsets
    and bounded so they cannot swap.

    Previously the centres were seeded uniformly across the window and left
    completely unbounded (-inf, +inf), with components then identified by their
    position in the parameter vector. That is unsafe: under noise a component
    can drift, cross a neighbour, or collapse onto the same peak, silently
    permuting the index -> component mapping. Here each centre is seeded at its
    known offset (shifted by the observed systemic velocity) and confined to a
    window narrower than half the smallest component separation, so the mapping
    cannot permute -- and `analyse_spectra` still verifies it by centre rather
    than trusting order.

    `sigma`: per-channel uncertainty, passed through to curve_fit. Supply it
    (e.g. the injected noise RMS) whenever `return_cov` is used -- without it
    curve_fit rescales the covariance by the residual variance, which on a
    noiseless model spectrum is numerical junk rather than an uncertainty.
    """
    offsets = hf.NH3_11_OFFSETS_KMS if number == 'one' else hf.NH3_22_OFFSETS_KMS
    v = np.asarray(v, dtype=float)
    tmb = np.asarray(tmb, dtype=float)

    v_sys = hf.estimate_v_sys_kms(v, tmb)
    centres, c_lo, c_hi = hf.initial_centres_and_bounds(offsets, v_sys_kms=v_sys)

    amp_pk = float(np.max(tmb))
    width0 = min(max((v[-1] - v[0]) / 40.0, 1e-3), sigma_max_kms * 0.5)

    p0, lower, upper = [], [], []
    for c, clo, chi in zip(centres, c_lo, c_hi):
        p0.extend([max(amp_pk / 3, 1e-3), c, width0])
        lower.extend([amp_min, clo, 1e-6])
        upper.extend([np.inf, chi, sigma_max_kms])

    pars, pcov = curve_fit(multi_gaussian, v, tmb, p0=p0, bounds=(lower, upper),
                            sigma=sigma, absolute_sigma=sigma is not None, maxfev=50000)
    if return_cov:
        return pars, pcov
    return pars


# -----------------------------
# Main Analysis Function
# -----------------------------
def analyse_spectra(odir, XNH3, numberdensity, vturb, T_cloud, radius_sphere, max_NLTE=100, save_plots=False):
    """
    Analyse the NH3 (1,1) and (2,2) spectra produced by Magritte NLTE model.
    Returns a dictionary of extracted parameters to be written safely by the main thread.
    """
    try:
        filenames = {
            'oneone': os.path.join(odir, f'fits/NLTE_nh3_spectrum_11_{XNH3}_{numberdensity:.2e}_{radius_sphere:.2e}_{vturb}_{T_cloud}.fits'),
            'twotwo': os.path.join(odir, f'fits/NLTE_nh3_spectrum_22_{XNH3}_{numberdensity:.2e}_{radius_sphere:.2e}_{vturb}_{T_cloud}.fits')
        }

        # Open FITS files ONCE to get both data and headers
        with fits.open(filenames['oneone']) as hdul1:
            spec1 = hdul1[0].data
            hdr1 = hdul1[0].header
            velos1 = hdr1['CRVAL1'] + np.arange(hdr1['NAXIS1']) * hdr1['CDELT1']
            freq1 = hdr1['RESTFREQ']

        with fits.open(filenames['twotwo']) as hdul2:
            spec2 = hdul2[0].data
            hdr2 = hdul2[0].header
            velos2 = hdr2['CRVAL1'] + np.arange(hdr2['NAXIS1']) * hdr2['CDELT1']
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
            subfolder = f"X{XNH3}_n{numberdensity:.2e}_{radius_sphere:.2e}_v{vturb}_T{T_cloud}"
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
            ax2.annotate(f'N_NH3 = {escape_probability(vturb/1000, radius_sphere, numberdensity, XNH3):.2e} m^-2', xy=(0.05, 0.05), xycoords='axes fraction')
            ax2.legend()

            plt.tight_layout()
            fig.savefig(os.path.join(image_subdir, f'NLTE_nh3_1122_{XNH3}_{numberdensity:.2e}_{radius_sphere:.2e}_{vturb}_{T_cloud}_fit.png'))
            plt.close(fig) # Critical to prevent memory leaks

        # -----------------------------
        # Identify components by fitted CENTRE, not by parameter index.
        # See nh3hia/hyperfine.py: index-based assignment mislabelled the outer
        # satellite pair (A_01 / A_10 swapped), which inverts the very asymmetry
        # the hyperfine anomaly consists of.
        # -----------------------------
        amps11_raw = p11[0:15:3]
        cens11_raw = p11[1:15:3]
        sigs11_raw = p11[2:15:3]
        idx = hf.identify_components(cens11_raw - hf.estimate_v_sys_kms(velos1, Tmb1_corrected))

        A = {k: float(amps11_raw[i]) for k, i in idx.items()}
        CEN = {k: float(cens11_raw[i]) for k, i in idx.items()}
        SIG = {k: float(sigs11_raw[i]) for k, i in idx.items()}
        # Integrated intensity of a Gaussian [K km/s]. Zhou et al. (2020) show
        # the peak-ratio anomaly estimator is biased along the velocity-dispersion
        # axis and recommend integrated intensities; both are reported here so the
        # two can be compared rather than one silently replacing the other.
        I = {k: A[k] * abs(SIG[k]) * np.sqrt(2.0 * np.pi) for k in A}

        # A component pinned at the amplitude lower bound while others carry real
        # flux means the fit has collapsed rather than measured anything. This
        # used to pass silently; record it so downstream can reject the row.
        AMP_FLOOR = 1e-4
        floor_pinned = int(sum(1 for k in A if A[k] <= AMP_FLOOR * 1.01))

        if np.isclose(A['A_MAIN'], 0):
            print(f"WARNING: Main HF component amplitude is zero for T={T_cloud}, n={numberdensity:.2e}. Ratios may be invalid.")

        # Delete the FITS files after analysis is complete
        for filename in filenames.values():
            if os.path.exists(filename):
                os.remove(filename)
                
        # Return the payload dictionary to the main thread delivery system
        def _ratio(num, den):
            return num / den if den != 0 else np.nan

        out = {
            # Peak amplitudes -- same keys and meaning as before, but now
            # correctly identified (the outer pair used to be swapped).
            'A_10': A['A_10'], 'A_21': A['A_21'], 'A_MAIN': A['A_MAIN'],
            'A_12': A['A_12'], 'A_01': A['A_01'],
            'R_01_MAIN': _ratio(A['A_01'], A['A_MAIN']),
            'R_10_MAIN': _ratio(A['A_10'], A['A_MAIN']),
            'R_21_MAIN': _ratio(A['A_21'], A['A_MAIN']),
            'R_12_MAIN': _ratio(A['A_12'], A['A_MAIN']),
            # Integrated intensities [K km/s] and their ratios -- the less biased
            # estimator (Zhou et al. 2020).
            'I_10': I['A_10'], 'I_21': I['A_21'], 'I_MAIN': I['A_MAIN'],
            'I_12': I['A_12'], 'I_01': I['A_01'],
            'RI_01_MAIN': _ratio(I['A_01'], I['A_MAIN']),
            'RI_10_MAIN': _ratio(I['A_10'], I['A_MAIN']),
            'RI_21_MAIN': _ratio(I['A_21'], I['A_MAIN']),
            'RI_12_MAIN': _ratio(I['A_12'], I['A_MAIN']),
            # Fitted centres and widths, so a bad fit is diagnosable after the fact.
            'fit_floor_pinned': floor_pinned,
            'CEN_MAIN': CEN['A_MAIN'], 'SIG_MAIN': SIG['A_MAIN'],
            'FWHM_MAIN': 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(SIG['A_MAIN']),
            # Hyperfine intensity anomaly, redshifted/blueshifted, integrated
            # (Zhou et al. 2020; Wu et al. 2024 convention). Static hyperfine
            # selective trapping predicts HIA_IS < 1 and HIA_OS > 1 (quadrant II);
            # infall gives both < 1, expansion both > 1. A source outside
            # quadrant II cannot be reproduced by a static constant-density model.
            'HIA_IS': _ratio(I['A_21'], I['A_12']),
            'HIA_OS': _ratio(I['A_01'], I['A_10']),
            'N_NH3': escape_probability(vturb/1000, radius_sphere, numberdensity, XNH3)
        }
        return out

    except Exception as e:
        raise RuntimeError(f"Spectral analysis failed: {e}")

if __name__ == "__main__":
    odir = "./"
    from nh3hia.model3d.sphere import run_model
    run_model(wdir="./", odir="./", XNH3=1e-9, 
              numberdensity=1e6, vturb=300, T_cloud=35, 
              max_NLTE=100, radius_sphere=1e16)
    
    results = analyse_spectra(
        odir=odir, 
        XNH3=1e-9, 
        numberdensity=1e6, 
        vturb=300, 
        T_cloud=35, 
        radius_sphere=1e16,
        max_NLTE=100,
        save_plots=True
    )
    print(results)