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
    os.environ["OMP_NUM_THREADS"] = "5"     # Set to 4 if running 4 workers, 8 if running 2 workers
    os.environ["OPENBLAS_NUM_THREADS"] = "5"
    os.environ["MKL_NUM_THREADS"] = "5"

    # 2. Local imports to ensure threading limits are respected
    from nh3_NLTE_sphere import run_model
    from nh3_NLTE_analysis import analyse_spectra

    try:
        # -----------------------------
        # Step 2: Analyse Spectra
        # -----------------------------
        # Notice save_plots=False. Set to True only if you want to generate images!
        analysis_data = analyse_spectra(
            odir=params['odir'],
            XNH3=params['XNH3'],
            numberdensity=params['numberdensity'],
            vturb=params['vturb'],
            T_cloud=params['T_cloud'],
            radius_sphere=params['radius_req'],
            max_NLTE=params['max_NLTE'],
            save_plots=False 
        )
        print(f"Completed analysis for XNH3={params['XNH3']}, n={params['numberdensity']:.2e}, radius={params['radius_req']:.2e}")
        # -----------------------------
        # Step 3: Package Payload
        # -----------------------------
        return {
            "status": "SUCCESS",
            "params": params,
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
    odir = "/home/yasho379/magritte_rebuilt/scratch/output/output_test_1e-6_parallel_12rays/"
    results_csv = os.path.join(odir, "results", "NLTE_nh3_1e-6_parallel_12rays_analysis_only.csv")
    
    os.makedirs(os.path.join(odir, "fits"), exist_ok=True)
    os.makedirs(os.path.join(odir, "images"), exist_ok=True)
    os.makedirs(os.path.join(odir, "results"), exist_ok=True)

    max_NLTE = 300 

    T_cloud_values = [36]       
    vturb_values = [100]            
    XNH3_values = [1e-8]            

    log_n_start, log_n_end = 3.5, 8.5
    n_steps = 30
    numberdensity_values = np.logspace(log_n_start, log_n_end, n_steps)

    log_N_dv_start, log_N_dv_end = 14.0, 17.0
    y_steps = 30
    target_log_N_dv_values = np.linspace(log_N_dv_start, log_N_dv_end, y_steps)

    tasks = []
    for T_cloud in T_cloud_values:
        for vturb in vturb_values:
            for XNH3 in XNH3_values:
                for numberdensity in numberdensity_values:
                    for target_log_y in target_log_N_dv_values:
                        
                        target_val_kms = 10**target_log_y
                        radius_req = (target_val_kms * (vturb/1000)) / (numberdensity * XNH3)

                        tasks.append({
                            'wdir': wdir,
                            'odir': odir,
                            'XNH3': float(XNH3),
                            'numberdensity': float(numberdensity),
                            'vturb': float(vturb),
                            'T_cloud': float(T_cloud),
                            'radius_req': float(radius_req),
                            'max_NLTE': max_NLTE
                        })

    # -----------------------------
    # Initialize the CSV File
    # -----------------------------
    # Write the master header row before starting the grid
    master_header = [
        'Status', 'T_cloud', 'vturb', 'XNH3', 'numberdensity', 'radius_req', 
        'A_10', 'A_21', 'A_MAIN', 'A_12', 'A_01', 
        'R_01_MAIN', 'R_10_MAIN', 'R_21_MAIN', 'R_12_MAIN', 
        'N_NH3','Error_Msg'
    ]
    
    # Only create the file and write the header if it doesn't already exist
    if not os.path.exists(results_csv):
        with open(results_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(master_header)
        print(f"Created new results file: {results_csv}")
    else:
        print(f"Found existing results file. Appending new runs to {results_csv}")

    print(f"\nStarting Grid of {len(tasks)} runs.")
    print("Executing with 4 concurrent workers...\n")
    
    # -----------------------------
    # Execute with Multiprocessing
    # -----------------------------
    # maxtasksperchild=10 is a good sweet spot to prevent memory leaks while 
    # not paying the startup tax on every single model.
    with multiprocessing.Pool(processes=3, maxtasksperchild=1) as pool:
        
        for result in tqdm(pool.imap_unordered(worker_task, tasks), total=len(tasks)):
            
            p = result['params']
            
            # The Main Thread safely opens the CSV and appends the row
            with open(results_csv, mode='a', newline='') as f:
                writer = csv.writer(f)
                
                if result['status'] == "SUCCESS":
                    a = result['analysis']
                    writer.writerow([
                        "SUCCESS", p['T_cloud'], p['vturb'], p['XNH3'], p['numberdensity'], p['radius_req'], 
                        f"{a['A_10']:.3f}", f"{a['A_21']:.3f}", f"{a['A_MAIN']:.3f}", f"{a['A_12']:.3f}", f"{a['A_01']:.3f}",
                        f"{a['R_01_MAIN']:.3f}", f"{a['R_10_MAIN']:.3f}", f"{a['R_21_MAIN']:.3f}", f"{a['R_12_MAIN']:.3f}",
                        f"{a['N_NH3']:.3e}", ""
                    ])
                else:
                    # Write failed row, padding missing analysis columns with NaNs
                    writer.writerow([
                        "FAILED", p['T_cloud'], p['vturb'], p['XNH3'], p['numberdensity'], p['radius_req'], 
                        "NaN", "NaN", "NaN", "NaN", "NaN", 
                        "NaN", "NaN", "NaN", "NaN", 
                        "NaN", "NaN", "Nan", "Nan",
                    ])

    print("\nGrid complete. Results safely saved to CSV.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_model_grid()