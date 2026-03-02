"""
Grid runner for NH3 NLTE sphere models + automatic spectral analysis.
Densely samples the log(N_NH3 / dv) vs log(n) plane by reverse-engineering radius.
Handles unit conversion between Stutzki plots (km/s) and internal code (m/s).
"""

from nh3_NLTE_sphere import run_model, analyse_spectra
import numpy as np

def run_model_grid():
    """
    Run NH3 NLTE models by iterating over the TARGET plot coordinates
    (Density vs N_NH3/dv) and back-calculating the required sphere radius.
    """

    max_NLTE = 100 

    # -----------------------------
    # 1. Define Fixed Physics Parameters
    # -----------------------------
    T_cloud_values = [36,18]       # K 
    vturb_values = [300]        # m/s (Keep this in m/s as required by your code)
    XNH3_values = [1e-8]        # Abundance

    # -----------------------------
    # 2. Define Target Grid (The Stutzki Plot Axes)
    # -----------------------------
    # X-axis: Number Density (3.5 to 8.5)
    log_n_start, log_n_end = 3.5, 8.5
    n_steps = 10
    numberdensity_values = np.logspace(log_n_start, log_n_end, n_steps)

    # Y-axis: log(N_NH3 / dv) (13 to 16)
    # Standard Unit: [cm^-2 / (km/s)]
    log_N_dv_start, log_N_dv_end = 14.0, 16.0
    y_steps = 10
    target_log_N_dv_values = np.linspace(log_N_dv_start, log_N_dv_end, y_steps)

    total_runs = (
        len(T_cloud_values) *
        len(vturb_values) *
        len(XNH3_values) *
        len(numberdensity_values) *
        len(target_log_N_dv_values)
    )

    print(f"\nStarting Grid of {total_runs} runs.")
    print(f"Sampling Density: {n_steps} x {y_steps} grid points.\n")

    count = 0

    for T_cloud in T_cloud_values:
        for vturb in vturb_values:
            for XNH3 in XNH3_values:
                for numberdensity in numberdensity_values:
                    for target_log_y in target_log_N_dv_values:
                        count += 1
                        
                        # --- REVERSE ENGINEERING RADIUS ---
                        
                        # 1. Retrieve the target value from the loop (in Stutzki units)
                        #    Target = N / dv_kms
                        target_val_kms = 10**target_log_y
                        
                        # 2. Convert target to "SI-like" units for the internal formula
                        #    We want to find N such that N / v_ms gives the correct magnitude.
                        #    Since v_ms = 1000 * v_kms, we need N to be 1000x larger 
                        #    to maintain the ratio if we were dividing by a larger number.
                        #    HOWEVER, your formula divides by vturb (m/s).
                        #    
                        #    Let's derive:
                        #    Wanted Ratio (Stutzki) = N / v_kms
                        #    Your Formula Output    = N / v_ms
                        #    
                        #    If we want Your Output == Wanted Ratio:
                        #       N / v_ms = Target
                        #       N = Target * v_ms
                        #    
                        #    So we substitute this N into the density equation:
                        #    N = n * X * R  =>  Target * v_ms = n * X * R
                        #
                        #    Therefore:
                        #    R = (Target * v_ms) / (n * X)
                        
                        radius_req = (target_val_kms * vturb) / (numberdensity * XNH3)

                        # Note: This logic assumes your code wants to reproduce the MAGNITUDE 
                        # of the Stutzki plot. If vturb=100 m/s, it simply multiplies 
                        # the target ratio by 100 to get the required column density.

                        # -----------------------------
                        # Run Simulation
                        # -----------------------------
                        try:
                            run_model(
                                XNH3=float(XNH3),
                                numberdensity=float(numberdensity),
                                vturb=float(vturb),
                                T_cloud=float(T_cloud),
                                radius_sphere=float(radius_req),
                                max_NLTE=max_NLTE,
                            )
                        except Exception as e:
                            print(f" Run FAILED: {e}")
                            continue

                        # -----------------------------
                        # Analyse Spectra
                        # -----------------------------
                        try:
                            analyse_spectra(
                                XNH3=float(XNH3),
                                numberdensity=float(numberdensity),
                                T_cloud=float(T_cloud),
                                radius_sphere=float(radius_req),
                                vturb=float(vturb),
                                max_NLTE=max_NLTE,
                            )
                        except Exception as e:
                            print(f" Analysis FAILED: {e}")

    print("\nGrid complete.")

if __name__ == "__main__":
    run_model_grid()
    