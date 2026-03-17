import os
import numpy as np
from nh3_NLTE_sphere import run_model
from nh3_NLTE_analysis import analyse_spectra

def run_model_grid():
    """
    Grid runner for NH3 NLTE sphere models + automatic spectral analysis.
    Densely samples the log(N_NH3 / dv) vs log(n) plane by reverse-engineering radius.
    """
    
    # -----------------------------
    # 0. Define Directories & Files
    # -----------------------------
    wdir = "/home/yasho379/magritte_rebuilt/tgs/"
    odir = "/home/yasho379/magritte_rebuilt/output_test_1e-6/"
    results_csv = os.path.join(odir, "results", "NLTE_nh3_v4.csv")
    
    # Ensure necessary output subdirectories exist
    os.makedirs(os.path.join(odir, "fits"), exist_ok=True)
    os.makedirs(os.path.join(odir, "images"), exist_ok=True)
    os.makedirs(os.path.join(odir, "results"), exist_ok=True)

    max_NLTE = 100 

    # -----------------------------
    # 1. Define Fixed Physics Parameters
    # -----------------------------
    T_cloud_values = [36, 18]       # K 
    vturb_values = [300]            # m/s (Keep this in m/s as required by your code)
    XNH3_values = [1e-8]            # Abundance

    # -----------------------------
    # 2. Define Target Grid (The Stutzki Plot Axes)
    # -----------------------------
    log_n_start, log_n_end = 3.5, 8.5
    n_steps = 3
    numberdensity_values = np.logspace(log_n_start, log_n_end, n_steps)

    log_N_dv_start, log_N_dv_end = 14.0, 16.0
    y_steps = 3
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
                        
                        target_val_kms = 10**target_log_y
                        radius_req = (target_val_kms * vturb) / (numberdensity * XNH3)

                        # -----------------------------
                        # Run Simulation
                        # -----------------------------
                        run_model(
                                wdir=wdir,
                                odir=odir,
                                XNH3=float(XNH3),
                                numberdensity=float(numberdensity),
                                vturb=float(vturb),
                                T_cloud=float(T_cloud),
                                radius_sphere=float(radius_req),
                                max_NLTE=max_NLTE,
                            )
                        # except Exception as e:
                        #     print(f" Run FAILED: {e}")
                        #     continue

                        # -----------------------------
                        # Analyse Spectra
                        # -----------------------------
                        try:
                            analyse_spectra(
                                odir=odir,
                                results_csv=results_csv,
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