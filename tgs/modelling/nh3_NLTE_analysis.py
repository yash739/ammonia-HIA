import os
import time
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy import units as u
from scipy.optimize import curve_fit, brentq


def analyse_spectra(odir, results_csv, XNH3, numberdensity, vturb, T_cloud, radius_sphere, max_NLTE=100):
    """
    Analyse the NH3 (1,1) and (2,2) spectra produced by Magritte NLTE model.
    """

    print("\n--- Starting spectral analysis ---")

    try:

        # Subfolder for this model's results
        subfolder = f"X{XNH3}_n{numberdensity:.2e}_{radius_sphere:.2e}_v{vturb}_T{T_cloud}"
        image_subdir = os.path.join(odir, "images", subfolder)
        os.makedirs(image_subdir, exist_ok=True)

        # Spectra FITS filenames
        filenames = {
            'oneone': os.path.join(odir, f'fits/NLTE_nh3_spectrum_11_{XNH3}_{numberdensity:.2e}_{radius_sphere:.2e}_{vturb}_{T_cloud}.fits'),
            'twotwo': os.path.join(odir, f'fits/NLTE_nh3_spectrum_22_{XNH3}_{numberdensity:.2e}_{radius_sphere:.2e}_{vturb}_{T_cloud}.fits')
        }

        spec1 = fits.getdata(filenames['oneone'])
        spec2 = fits.getdata(filenames['twotwo'])

        with fits.open(filenames['oneone']) as hdul:
            hdr1 = hdul[0].header
            velos1 = hdr1['CRVAL1'] + np.arange(hdr1['NAXIS1']) * hdr1['CDELT1']

        with fits.open(filenames['twotwo']) as hdul:
            hdr2 = hdul[0].header
            velos2 = hdr2['CRVAL1'] + np.arange(hdr2['NAXIS1']) * hdr2['CDELT1']

        freq1 = hdr1['RESTFREQ'] 
        freq2 = hdr2['RESTFREQ']

        h  = 6.62607015e-34          
        k_B  = 1.380649e-23          
        c  = 2.99792458e8            

        def intensity_to_Tmb(v,I, freq):
            Tmb = (c**2 * I) / (2 * k_B * (freq*(1+v/c))**2)
            return Tmb

        def escape_probability(vturb,radius_sphere, nH2, XNH3):
            N_NH3 = nH2 * XNH3 * radius_sphere /vturb 
            return N_NH3

        Tmb1 = intensity_to_Tmb(1000*velos1, spec1, freq1)
        Tmb2 = intensity_to_Tmb(1000*velos2, spec2, freq2)

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
        plt.savefig(os.path.join(image_subdir, f'NLTE_nh3_1122_{XNH3}_{numberdensity:.2e}_{radius_sphere:.2e}_{vturb}_{T_cloud}_corrected.png'))
        plt.close()

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
            plt.annotate(f'N_NH3 = {escape_probability(vturb/1000, radius_sphere, numberdensity, XNH3):.2e} m^-2', xy=(0, -5), xycoords='axes points')
            plt.tight_layout()
            plt.savefig(os.path.join(image_subdir, f'NLTE_nh3_1122_{XNH3}_{numberdensity:.2e}_{radius_sphere:.2e}_{vturb}_{T_cloud}_fit.png'))
            plt.close()

            return dict(fit_params_11=p11, fit_params_22=p22)

        res = analyse_pair(velos1, Tmb1_corrected, velos2, Tmb2_corrected)

        amps11 = res['fit_params_11'][0:15:3]
        print("NH3 (1,1) hyperfine amplitudes:", amps11)
        
        # Ensure the directory for the CSV exists
        os.makedirs(os.path.dirname(results_csv), exist_ok=True)
        header = (
            "XNH3,numberdensity,vturb,T_cloud,"
            "A_10,A_21,A_MAIN,A_12,A_01,"
            "R_01_MAIN,R_10_MAIN,R_21_MAIN,R_12_MAIN,"
            "N_NH3, max_NLTE\n"
        )

        if not os.path.exists(results_csv) or os.stat(results_csv).st_size == 0:
            with open(results_csv, 'w') as f:
                f.write(header)

        with open(results_csv, 'a') as f:
            f.write(
                f"{XNH3},{numberdensity},{vturb},{T_cloud},"
                f"{amps11[0]:.3f},{amps11[1]:.3f},{amps11[2]:.3f},{amps11[3]:.3f},{amps11[4]:.3f},"
                f"{amps11[4]/amps11[2]:.3f},{amps11[0]/amps11[2]:.3f},{amps11[1]/amps11[2]:.3f},{amps11[3]/amps11[2]:.3f},"
                f"{escape_probability(vturb/1000, radius_sphere, numberdensity, XNH3):.3e},{max_NLTE}\n"
            )
        print(f"Results written to {results_csv}")

    except Exception as e:
        print("\n!!! ERROR during spectral analysis !!!")
        print("Type:", type(e).__name__)
        print("Message:", e)
        import traceback
        traceback.print_exc()
        print("Analysis aborted before CSV writing.")
        return

    try:
        if 'res' not in locals():
            raise RuntimeError("Result dictionary 'res' not created.")

        if 'fit_params_11' not in res:
            raise RuntimeError("fit_params_11 missing from results.")

        amps11 = res['fit_params_11'][0:15:3]

        if len(amps11) != 5:
            raise RuntimeError("Expected 5 hyperfine amplitudes.")

        if np.isclose(amps11[2], 0):
            print("WARNING: Main hyperfine component amplitude is zero.")
            print("Ratios may be invalid.")

    except Exception as e:
        print("Post-analysis validation failed:", e)
        return

    try:
        print("Verifying CSV file location...")
        print("Absolute path:", os.path.abspath(results_csv))

        if not os.path.exists(results_csv):
            print("WARNING: CSV file does not exist after write attempt.")
        else:
            print("CSV file size:", os.path.getsize(results_csv), "bytes")

    except Exception as e:
        print("CSV verification failed:", e)

    print("--- Spectral analysis complete ---\n")