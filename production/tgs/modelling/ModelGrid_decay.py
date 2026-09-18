import os
import csv
import numpy as np
import multiprocessing
from tqdm import tqdm

def worker_task(params):
    """
    Worker function to execute a single model and its analysis.
    Acts as a 'delivery driver' - it runs the math and returns a dictionary payload.
    """
    # 1. Set environment variables FIRST to prevent CPU thrashing
    os.environ["OMP_NUM_THREADS"] = "3"     
    os.environ["OPENBLAS_NUM_THREADS"] = "3"
    os.environ["MKL_NUM_THREADS"] = "3"

    # 2. Local imports to ensure threading limits are respected
    from nh3_NLTE_decay import run_model
    from nh3_NLTE_analysis_decay import analyse_spectra

    try:
        # -----------------------------
        # Step 1: Run Magritte Model
        # -----------------------------
        convergence_info = run_model(
            wdir=params['wdir'],
            odir=params['odir'],
            vturb=params['vturb'],
            max_NLTE=params['max_NLTE'],
            n0=params['n0'],
            r0_n_arcsec=params['r0_n_arcsec'],
            alpha_n=params['alpha_n'],
            X0=params['X0'],
            alpha_X=params['alpha_X'],
            T_out=params['T_out'],
            T_in=params['T_in'],
            r0_T_arcsec=params['r0_T_arcsec'],
            distance_pc=params['distance_pc']
        )

        # -----------------------------
        # Step 2: Analyse Spectra
        # -----------------------------
        analysis_data = analyse_spectra(
            odir=params['odir'],
            vturb=params['vturb'],
            max_NLTE=params['max_NLTE'],
            n0=params['n0'],
            r0_n_arcsec=params['r0_n_arcsec'],
            alpha_n=params['alpha_n'],
            X0=params['X0'],
            alpha_X=params['alpha_X'],
            T_out=params['T_out'],
            T_in=params['T_in'],
            r0_T_arcsec=params['r0_T_arcsec'],
            distance_pc=params['distance_pc'],
            save_plots=True 
        )
        print(f"Completed analysis for X0={params['X0']:.2e}, n0={params['n0']:.2e}, alpha_n={params['alpha_n']:.2f}")
        
        # -----------------------------
        # Step 3: Package Payload
        # -----------------------------
        return {
            "status": "SUCCESS",
            "params": params,
            "convergence": convergence_info,
            "analysis": analysis_data
        }

    except Exception as e:
        print(str(e))
        return {
            "status": "FAILED",
            "params": params,
            "error": str(e)
        }

def run_model_grid():
    """
    Grid runner for NH3 NLTE sphere models + automatic spectral analysis.
    Safely writes ALL results to a single CSV file.
    """
    wdir = "/home/yasho379/magritte_rebuilt/production/tgs/"
    odir = "/home/yasho379/magritte_rebuilt/scratch/output/output_test_decaying_crapsi/"
    results_csv = os.path.join(odir, "results", "NLTE_nh3_decaying_crapsi.csv")
    
    os.makedirs(os.path.join(odir, "fits"), exist_ok=True)
    os.makedirs(os.path.join(odir, "images"), exist_ok=True)
    os.makedirs(os.path.join(odir, "results"), exist_ok=True)

    max_NLTE = 200 

    T_cloud_values = [5.5, 12]       
    vturb_values = [100, 300, 500]            
    XNH3_values = [1e-8, 5e-9, 1e-9]            

    log_n_start, log_n_end = 4.5, 6.5
    n_steps = 5
    numberdensity_values = np.logspace(log_n_start, log_n_end, n_steps)

    tasks = []
    for vturb in vturb_values:
        for T_cloud in T_cloud_values:
            for XNH3 in XNH3_values:
                for numberdensity in numberdensity_values:
                                                                
                        decay_index = 1.5
                        tasks.append({
                            'wdir': wdir,
                            'odir': odir,
                            'vturb': float(vturb),
                            'max_NLTE': max_NLTE,
                            # Structural Crapsi Mapping Explicit Keys:
                            'n0': float(numberdensity),
                            'r0_n_arcsec': 14.0,
                            'alpha_n': float(decay_index),
                            'X0': float(XNH3),
                            'alpha_X': 0.16,
                            'T_out': float(T_cloud),
                            'T_in': 5.5,
                            'r0_T_arcsec': 18.0,
                            'distance_pc': 140.0
                        })

    # -----------------------------
    # Initialize the CSV File
    # -----------------------------
    # Expanded header incorporating all direct Crapsi parameter variables
    master_header = [
        'Status', 
        'n0', 'r0_n_arcsec', 'alpha_n', 'X0', 'alpha_X', 
        'T_out', 'T_in', 'r0_T_arcsec', 'distance_pc', 'vturb', 'max_NLTE',
        'A_10', 'A_21', 'A_MAIN', 'A_12', 'A_01', 
        'R_01_MAIN', 'R_10_MAIN', 'R_21_MAIN', 'R_12_MAIN', 
        'Main Hyperfine Optical Depth', 'Halting Iteration', 'Final Convergence', 'Error_Msg'
    ]
    
    if not os.path.exists(results_csv):
        with open(results_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(master_header)
        print(f"Created new results file: {results_csv}")
    else:
        print(f"Found existing results file. Appending new runs to {results_csv}")
    
    print(f"\nStarting Grid of {len(tasks)} runs.")
    print("Executing with 3 concurrent workers...\n")
    
    # -----------------------------
    # Execute with Multiprocessing
    # -----------------------------
    with multiprocessing.Pool(processes=5, maxtasksperchild=1) as pool:
        
        for result in tqdm(pool.imap_unordered(worker_task, tasks), total=len(tasks)):
            
            p = result['params']
            
            with open(results_csv, mode='a', newline='') as f:
                writer = csv.writer(f)
                
                if result['status'] == "SUCCESS":
                    a = result['analysis']
                    writer.writerow([
                        "SUCCESS", 
                        f"{p['n0']:.2e}", f"{p['r0_n_arcsec']:.1f}", f"{p['alpha_n']:.2f}", 
                        f"{p['X0']:.2e}", f"{p['alpha_X']:.2f}", f"{p['T_out']:.1f}", 
                        f"{p['T_in']:.1f}", f"{p['r0_T_arcsec']:.1f}", f"{p['distance_pc']:.1f}", 
                        f"{p['vturb']:.1f}", int(p['max_NLTE']),
                        f"{a['A_10']:.3f}", f"{a['A_21']:.3f}", f"{a['A_MAIN']:.3f}", f"{a['A_12']:.3f}", f"{a['A_01']:.3f}",
                        f"{a['R_01_MAIN']:.3f}", f"{a['R_10_MAIN']:.3f}", f"{a['R_21_MAIN']:.3f}", f"{a['R_12_MAIN']:.3f}",
                        str(result['convergence'][2]), str(result['convergence'][0]), str(result['convergence'][1]), ""
                    ])
                else:
                    writer.writerow([
                        "FAILED", 
                        f"{p['n0']:.2e}", f"{p['r0_n_arcsec']:.1f}", f"{p['alpha_n']:.2f}", 
                        f"{p['X0']:.2e}", f"{p['alpha_X']:.2f}", f"{p['T_out']:.1f}", 
                        f"{p['T_in']:.1f}", f"{p['r0_T_arcsec']:.1f}", f"{p['distance_pc']:.1f}", 
                        f"{p['vturb']:.1f}", int(p['max_NLTE']),
                        "NaN", "NaN", "NaN", "NaN", "NaN", 
                        "NaN", "NaN", "NaN", "NaN", 
                        "NaN", "NaN", "NaN", str(result.get('error', 'Unknown Error'))
                    ])

    print("\nGrid complete. Results safely saved to CSV.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_model_grid()