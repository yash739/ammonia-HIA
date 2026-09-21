"""Paper I, Work item -1, step 4: the real test. NLTE (max_NLTE=250) wall
time for cube, the two-band hybrid mesh (core + surface radial shells,
cube seed only in the well-populated middle band), and the pure radial
mesh, all at the SAME physical point -- LOG_N=6.0, T=24.0, LOG_NDV=15.0,
the same reference point used throughout this mesh-economy investigation's
LTE diagnostics (chord_experiment.py's own point).

Answers the open question from the 09-16 notes (Sec 3.1): is the radial
mesh's ~13x NLTE slowdown vs cube driven by total point count, or by the
near-origin cell geometry (the origin point + 0.01R inner boundary shell)
specifically? The hybrid keeps that same near-origin structure but at
814/1313 = 62% of radial's total points -- if NLTE cost tracks point count,
hybrid should land near 62% of radial's wall time; if it tracks near-origin
geometry instead, hybrid should cost close to what radial does regardless.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import params as p
from nh3_NLTE_sphere import run_model

WDIR = "/home/yasho379/magritte_rebuilt/production/tgs/"
ODIR = "/tmp/mesh_economy_nlte/"
LOG_N, T, LOG_NDV = 6.0, 24.0, 15.0
XNH3 = 1e-8
DV_KMS = 0.3

MESHES = [
    ('cube', None),
    ('hybrid_core0.4_surf0.85', {'kind': 'hybrid', 'core_radius_frac': 0.4, 'surface_frac': 0.85}),
    ('radial', {'kind': 'radial', 'n_shell': 14, 'base': 260, 'seed': 7}),
]


def main():
    for sub in ('fits', 'images'):
        os.makedirs(os.path.join(ODIR, sub), exist_ok=True)
    n_H2 = 10.0 ** LOG_N
    N = 10.0 ** LOG_NDV * DV_KMS
    radius = N / (2.0 * n_H2 * XNH3)

    results = {}
    for name, mesh in MESHES:
        print(f"=== {name} ===", flush=True)
        t0 = time.time()
        hi, conv, tau_main, extra, npoints, nb = run_model(
            wdir=WDIR, odir=ODIR, XNH3=XNH3, numberdensity=n_H2,
            vturb=p.fwhm_kms_to_vturb_ms(DV_KMS), T_cloud=T, max_NLTE=250,
            radius_sphere=radius, nrays=12, resolution=10, nx_pix=16, ny_pix=16,
            spectrum='integrated', fov_pad_factor=1.0, mesh=mesh)
        dt = time.time() - t0
        results[name] = dict(npoints=npoints, wall_s=dt, convergence=conv, tau_main=tau_main)
        print(f"=== {name}: npoints={npoints}  wall_s={dt:.1f}  convergence={conv:.2f}%  "
              f"tau_main={tau_main:.4f} ===", flush=True)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    cube_wall = results['cube']['wall_s']
    for name, r in results.items():
        print(f"{name:28s} npoints={r['npoints']:5d}  wall_s={r['wall_s']:8.1f}  "
              f"ratio_to_cube={r['wall_s']/cube_wall:6.2f}x  conv={r['convergence']:.2f}%")


if __name__ == '__main__':
    main()
