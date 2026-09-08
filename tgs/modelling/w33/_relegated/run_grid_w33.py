"""
Stage A: extend the sphere catalogue into W33's actual regime.

Follows ModelGrid.py's exact worker/CSV pattern (see the committed `master`
version -- this script's `worker_task`, CSV header, and multiprocessing.Pool
usage are a direct copy of that, not a reinvention), but retargeted:

- Per-source T_kin (Table A.5: 13-16 K, below the existing catalogue's {18,36,54} K
  and below Stutzki's own 18-36 K grid), and (X_NH3, numberdensity, target_log_N_dv)
  swept as before.
- vturb is NOT taken from each source's observed (1,1) linewidth (Table 4:
  2.1-4.6 km/s). Session finding, confirmed against Stutzki & Winnewisser (1985):
  the hyperfine-anomaly mechanism this whole pipeline is built to test requires a
  *narrow* local/microturbulent linewidth (~0.1-0.3 km/s -- coincides with the
  paper's own thermal linewidth formula for NH3 at 13-20 K) so that adjacent
  hyperfine components don't overlap in velocity space; that's what allows
  selective trapping to differentiate between them at all. The broader *observed*
  linewidth per source is understood as statistical/gradient broadening on top of
  that narrow local width, not the local width itself. Feeding the broad observed
  linewidth in as `vturb` (the mistake in this script's first version) dilutes the
  line-center optical depth by roughly the ratio of the two linewidths -- this was
  caught by a 6-11x brightness deficit against Stutzki's own fits in Stage 0 using
  the same wrong convention. CLUMP_DV_VALUES is swept instead, per source.
- Density and column axes narrowed around W33's actual Table A.5/6 values
  (n_H2 ~ 0.5-2.6e4 cm^-3, N_para ~ 1e15 cm^-2) rather than the existing
  catalogue's much broader range -- finer resolution where it's actually needed.

Default grid size is deliberately modest given how slow high-optical-depth NLTE
runs converge (session finding: tau>5 rows average ~253 iterations even at
max_NLTE=300) -- check the printed task count before launching a larger one.

Note on grid history: W33_A was run to completion at the denser settings
(8 numberdensity x 6 target_log_N_dv points, max_NLTE=300 -- 192 of its own
960-task allocation). After reviewing W33_A's results, the grid was trimmed to
4x4 points and max_NLTE=200 (session request) for the remaining 4 sources
(W33_B, Main1, A1, B1) to speed up completion -- so W33_A's saved rows are on a
finer grid than the other sources' will be. Keep this in mind when comparing
across sources; screen.py's ranking doesn't care about grid density, but the
*coverage* (how close the nearest grid point is to the true optimum) differs.
"""

import os
import sys
import csv
import multiprocessing
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params as p

WDIR = "/home/yasho379/magritte_rebuilt/tgs/"
ODIR = "/home/yasho379/magritte_rebuilt/output_w33_grid_extension/"
RESULTS_CSV = os.path.join(ODIR, "results", "NLTE_nh3_w33_extension.csv")

DEFAULT_MAX_NLTE = 200  # trimmed from 300 after W33_A's run (session request) to speed up
                         # the remaining 4 sources; W33_A itself used 300.

# Narrow local/microturbulent linewidth sweep (FWHM, km/s) -- see module docstring.
# NOT each source's observed (1,1) linewidth.
CLUMP_DV_VALUES = [0.1, 0.3]

# X_NH3 sweep bracketing Table 6's per-source range (1.4-4.0e-8)
X_NH3_VALUES = [1e-8, 4e-8]

# numberdensity sweep narrowed around Table A.5's per-source range (0.5-2.6e4 cm^-3)
# -- log 3.0-4.7 = 1e3-5e4 cm^-3, with margin either side. Trimmed from 8 to 4
# points (session request) for the remaining 4 sources after W33_A's run.
LOG_N_START, LOG_N_END, N_STEPS = 3.0, 4.7, 4
NUMBERDENSITY_VALUES = np.logspace(LOG_N_START, LOG_N_END, N_STEPS)

# target log10(N_NH3/Delta-v) sweep narrowed around W33's implied range
# (N_para~1e15 cm^-2, Delta-v~2-4 km/s -> N/dv ~ 2.5-5e14, log10~14.4-14.7);
# kept wider than that point estimate since this is what screen.py searches over.
# Trimmed from 6 to 4 points (session request) for the remaining 4 sources.
LOG_NDV_START, LOG_NDV_END, NDV_STEPS = 13.5, 15.5, 4
TARGET_LOG_NDV_VALUES = np.linspace(LOG_NDV_START, LOG_NDV_END, NDV_STEPS)


def worker_task(task):
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    from nh3_NLTE_sphere import run_model
    from nh3_NLTE_analysis import analyse_spectra

    try:
        convergence_info = run_model(
            wdir=task['wdir'], odir=task['odir'], XNH3=task['XNH3'],
            numberdensity=task['numberdensity'], vturb=task['vturb'],
            T_cloud=task['T_cloud'], radius_sphere=task['radius_req'],
            max_NLTE=task['max_NLTE'],
        )
        analysis_data = analyse_spectra(
            odir=task['odir'], XNH3=task['XNH3'], numberdensity=task['numberdensity'],
            vturb=task['vturb'], T_cloud=task['T_cloud'], radius_sphere=task['radius_req'],
            max_NLTE=task['max_NLTE'], save_plots=False,
        )
        return {"status": "SUCCESS", "task": task, "convergence": convergence_info,
                "analysis": analysis_data}
    except Exception as e:
        return {"status": "FAILED", "task": task, "error": str(e)}


def build_tasks(source_names=None, X_values=X_NH3_VALUES,
                 numberdensity_values=NUMBERDENSITY_VALUES,
                 target_log_ndv_values=TARGET_LOG_NDV_VALUES,
                 clump_dv_values=CLUMP_DV_VALUES,
                 max_NLTE=DEFAULT_MAX_NLTE):
    source_names = source_names or list(p.EMISSION_SOURCES)
    tasks = []
    for name in source_names:
        src = p.source(name)
        T_cloud = src['T_kin']
        for clump_dv in clump_dv_values:
            vturb = p.fwhm_kms_to_vturb_ms(clump_dv)
            for XNH3 in X_values:
                for numberdensity in numberdensity_values:
                    for target_log_y in target_log_ndv_values:
                        target_val = 10 ** target_log_y
                        radius_req = (target_val * (vturb / 1000.0)) / (numberdensity * XNH3)
                        tasks.append({
                            'wdir': WDIR, 'odir': ODIR, 'source': name,
                            'clump_dv': float(clump_dv),
                            'XNH3': float(XNH3), 'numberdensity': float(numberdensity),
                            'vturb': float(vturb), 'T_cloud': float(T_cloud),
                            'radius_req': float(radius_req), 'max_NLTE': max_NLTE,
                        })
    return tasks


MASTER_HEADER = [
    'Status', 'source', 'clump_dv', 'T_cloud', 'vturb', 'XNH3', 'numberdensity', 'radius_req',
    'A_10', 'A_21', 'A_MAIN', 'A_12', 'A_01',
    'R_01_MAIN', 'R_10_MAIN', 'R_21_MAIN', 'R_12_MAIN',
    'N_NH3', 'Main Hyperfine Optical Depth', 'Halting Iteration', 'Final Convergence',
    'Error_Msg',
]


def run_grid(tasks=None, processes=4, results_csv=RESULTS_CSV):
    tasks = tasks if tasks is not None else build_tasks()

    os.makedirs(os.path.join(ODIR, "fits"), exist_ok=True)
    os.makedirs(os.path.join(ODIR, "images"), exist_ok=True)
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)

    if not os.path.exists(results_csv):
        with open(results_csv, 'w', newline='') as f:
            csv.writer(f).writerow(MASTER_HEADER)
        print(f"Created new results file: {results_csv}")
    else:
        print(f"Found existing results file. Appending new runs to {results_csv}")

    print(f"\nStarting Stage A grid of {len(tasks)} runs, {processes} concurrent workers.\n")

    with multiprocessing.Pool(processes=processes, maxtasksperchild=1) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_task, tasks), 1):
            t = result['task']
            with open(results_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if result['status'] == "SUCCESS":
                    a = result['analysis']
                    conv = result['convergence']
                    writer.writerow([
                        "SUCCESS", t['source'], t['clump_dv'], t['T_cloud'], t['vturb'], t['XNH3'],
                        t['numberdensity'], t['radius_req'],
                        f"{a['A_10']:.3f}", f"{a['A_21']:.3f}", f"{a['A_MAIN']:.3f}",
                        f"{a['A_12']:.3f}", f"{a['A_01']:.3f}",
                        f"{a['R_01_MAIN']:.3f}", f"{a['R_10_MAIN']:.3f}",
                        f"{a['R_21_MAIN']:.3f}", f"{a['R_12_MAIN']:.3f}",
                        f"{a['N_NH3']:.3e}", str(conv[2]), str(conv[0]), str(conv[1]), "",
                    ])
                else:
                    writer.writerow([
                        "FAILED", t['source'], t['clump_dv'], t['T_cloud'], t['vturb'], t['XNH3'],
                        t['numberdensity'], t['radius_req'],
                        "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN",
                        "NaN", "NaN", "NaN", "NaN", result['error'],
                    ])
            if i % 25 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] done")

    print("\nStage A grid complete.")


if __name__ == "__main__":
    tasks = build_tasks()
    print(f"Stage A task count: {len(tasks)} "
          f"({len(p.EMISSION_SOURCES)} sources x {len(X_NH3_VALUES)} X values x "
          f"{N_STEPS} densities x {NDV_STEPS} column targets)")
    multiprocessing.freeze_support()
    run_grid(tasks)
