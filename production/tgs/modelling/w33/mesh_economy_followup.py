"""Paper I, Work item -1 follow-up (09-22): three direct questions raised
after the mesh economy conclusion (hybrid not adopted) -- does raw cube
RESOLUTION help, does matching cube and radial POINT COUNT close the gap,
and does the BOUNDARY shell configuration matter. All LTE-only
(max_NLTE=0), same reference point as chord_experiment.py throughout
(T=24K, log n_H2=6.0, log(N/dv)=15.0).
"""
import os

import numpy as np

from lte_probe import run_lte, tau_from_brightness, chord_metrics

LOG_N, T, LOG_NDV = 6.0, 24.0, 15.0


def q1_resolution_sweep():
    print("=" * 78)
    print("Q1: does upping the CUBE mesh's resolution (with the ordinary density")
    print("    remesher, as used in production) fix the chord-law defect?")
    print("=" * 78)
    for res in (10, 14, 18, 24):
        run = run_lte(LOG_N, T, LOG_NDV, mesh=None, resolution=res, odir='/tmp/mesh_followup/')
        b, tau, _ = tau_from_brightness(run)
        m = chord_metrics(b, tau)
        print(f"  resolution={res:3d}  npoints={run['npoints']:5d}  RMS={m['rms']:6.1%}  "
              f"monotonic={str(m['monotonic']):5s}  limb/law={m['limb']:.3f}  A(tau_c)={m['A']:.3f}")
    print("  -> RMS plateaus ~20-24% (never approaches radial's 1.4%); monotonicity")
    print("     never fixes at ANY resolution tested. Limb recovery DOES improve")
    print("     substantially (0.003 -> 0.60-0.68) because higher resolution gives")
    print("     the remesher finer boxes right at the one place it has any density")
    print("     gradient to detect (r=r_out); the flat interior never gets that,")
    print("     regardless of seed density, so the core stays collapsed.")
    print()


def q2_matched_point_count():
    print("=" * 78)
    print("Q2: if cube and radial meshes had ~the same total point count, does the")
    print("    gap close? (raw/unremeshed cube seed, mesh={'kind':'cube_raw'})")
    print("=" * 78)
    for res in (10, 15, 16, 20):
        mesh = {'kind': 'cube_raw'}
        run = run_lte(LOG_N, T, LOG_NDV, mesh=mesh, resolution=res, odir='/tmp/mesh_followup/')
        b, tau, _ = tau_from_brightness(run)
        m = chord_metrics(b, tau)
        print(f"  cube_raw res={res:3d}  npoints={run['npoints']:5d}  RMS={m['rms']:6.1%}  "
              f"monotonic={str(m['monotonic']):5s}  limb/law={m['limb']:.3f}  A(tau_c)={m['A']:.3f}")
    print("  (radial mesh, for reference: npoints=1313  RMS=1.4%  monotonic=True  limb/law=0.786)")
    print("  -> Skipping the remesher entirely is the single biggest lever: even")
    print("     res=10 raw (376 pts) beats the REMESHED cube at res=24 (647 pts,")
    print("     23.5% RMS). At matched-or-excess point count (1128-1952 vs radial's")
    print("     1313), raw cube RMS drops to ~4.7-4.9% -- close to radial's 1.4% but")
    print("     NOT matching it, and monotonicity STILL never fixes at any point")
    print("     count tested. So point count under Cartesian topology explains most")
    print("     but not all of the gap; radial's shell structure (uniform coverage")
    print("     PER SHELL IN r, not per unit volume) has a genuine topological")
    print("     advantage no amount of Cartesian point count fully replicates.")
    print()


def q3_boundary_resolution():
    print("=" * 78)
    print("Q3: does the boundary shell configuration (healpy_order, currently")
    print("    hardcoded to 3 everywhere: 12*3^2=108 points/shell) matter?")
    print("=" * 78)
    for order in (2, 3, 5, 8):
        mesh = {'kind': 'cube', 'boundary_healpy_order': order}
        run = run_lte(LOG_N, T, LOG_NDV, mesh=mesh, resolution=10, odir='/tmp/mesh_followup/')
        b, tau, _ = tau_from_brightness(run)
        m = chord_metrics(b, tau)
        print(f"  healpy_order={order:2d}  npoints={run['npoints']:5d}  RMS={m['rms']:6.1%}  "
              f"monotonic={str(m['monotonic']):5s}  limb/law={m['limb']:.3f}  A(tau_c)={m['A']:.3f}")
    print("  -> RMS stays in the same 21-25% band regardless of boundary order;")
    print("     monotonicity never fixes. The interior remesher collapse remains")
    print("     the dominant defect, not boundary under-sampling. One real, minor")
    print("     wrinkle: at fov_pad_factor=1.0 (used throughout this diagnostic)")
    print("     the OUTER boundary shell sits almost exactly at r=r_out, so it")
    print("     directly participates in the 'limb' bin -- order=3's especially bad")
    print("     limb/law=0.003 here is a coincidence of that overlap, not a trend")
    print("     (order=2/5/8 all score better on limb despite fewer/more points).")
    print("     Production's actual fov_pad_factor=1.15 places the boundary outside")
    print("     the sphere, so this specific pad=1.0 quirk doesn't carry over.")


if __name__ == '__main__':
    q1_resolution_sweep()
    q2_matched_point_count()
    q3_boundary_resolution()
