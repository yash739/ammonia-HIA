"""
Grid runner for NH3 NLTE sphere models + automatic spectral analysis.

For each point in the parameter grid:
    1. Run Magritte NLTE model → produces FITS + image files
    2. Run analyse_spectra(...) → fits NH3 spectra, saves plots + CSV results
"""

from nh3_NLTE_sphere import run_model, analyse_spectra
import numpy as np
import itertools

def run_model_grid():
    """
    Run NH3 NLTE models in a grid fashion over selected parameters
    and automatically analyse spectra afterwards.
    """

    max_NLTE = 100  # constant for all runs

    # -----------------------------
    # Generate parameter combinations
    # -----------------------------
    grid = itertools.product(T_cloud_values, XNH3_values, numberdensity_values, radius_values, vturb_values)

    total = (
        len(T_cloud_values)
        * len(XNH3_values)
        * len(numberdensity_values)
        * len(radius_values)
        * len(vturb_values)
    )

    print(f"\nStarting grid of {total} NH3 NLTE model runs...\n")

    # -----------------------------
    # Loop through the grid
    # -----------------------------
    for i, (T_cloud, XNH3, numberdensity, radius_value, vturb) in enumerate(grid, 1):

        print(f"========== Model {i}/{total} ==========")
        print(f"T_cloud       = {T_cloud:.2f} K")
        print(f"XNH3          = {XNH3:.2e}")
        print(f"numberdensity = {numberdensity:.2e} cm^-3")
        print(f"radius        = {radius_value:.2e} cm")
        print(f"vturb         = {vturb:.2f} m/s")
        print("----------------------------------------")

        try:
            # Run the physical simulation
            run_model(
                XNH3=float(XNH3),
                numberdensity=float(numberdensity),
                vturb=float(vturb),
                T_cloud=float(T_cloud),
                radius_sphere=float(radius_value),
                max_NLTE=max_NLTE,
            )
        except Exception as e:
            print(f" Model FAILED at:")
            print(f" XNH3={XNH3:.2e}, n={numberdensity:.2e}, vturb={vturb:.2f}, T={T_cloud:.2f}")
            print(f" Error: {e}\n")
            continue  # skip analysis if model output is missing
        else:
            print("✅ Model completed successfully.")

        # Analyse the spectra
        try:
            analyse_spectra(
                XNH3=float(XNH3),
                numberdensity=float(numberdensity),
                T_cloud=float(T_cloud),
                radius_sphere=float(radius_value),
                vturb=float(vturb),
            )
        except Exception as e:
            print(f"Analysis FAILED at:")
            print(f"XNH3={XNH3:.2e}, n={numberdensity:.2e}, vturb={vturb:.2f}, T={T_cloud:.2f}")
            print(f"Error: {e}\n")
        else:
            print("Spectral analysis complete.\n")

    print("\nAll grid models + analyses finished.\n")

# -----------------------------
# Define parameter ranges via logspace
# -----------------------------
T_cloud_values = [36,18]     # K
XNH3_values = [1e-8,1e-9]            # abundance
numberdensity_values = np.logspace(3.5,8.5,5, base=10)   # cm^-3
radius_values = np.logspace(17,22,10,base=10)          # cm
vturb_values = [100]            # m/s

if __name__ == "__main__":
    run_model_grid()
