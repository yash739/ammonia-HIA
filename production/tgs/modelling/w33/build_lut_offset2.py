"""Second offset grid: refines log_N_dv, the one axis the first offset grid
(build_lut_offset.py) never touched.

That grid refined density (crossed against the full log_N_dv axis) and
added T=27K (crossed against the full density axis), but every point it
added still sat at one of the original 5 log_N_dv values -- so log_N_dv
itself never gained a single new point. The hold-one-out check run
specifically on this axis (see the report) found median interpolation
errors of 6-9%, an order of magnitude worse than the ~1% the OTHER two
axes now have, worst at the two interior points nearest the top of the
axis (14.90, 15.35) where the ratio-vs-log_N_dv curves are most convex.

This grid adds the 4 log_N_dv midpoints (14.225, 14.675, 15.125, 15.575)
crossed against the FULL density axis (all 14 points: the original 9 plus
the first offset grid's 5 density midpoints) at each of the 4 complete
temperatures (18/24/30/36K) -- i.e. the genuine 2D refinement the first
offset grid applied to density and T but not to this axis. 14 x 4 x 4 = 224
new models.

Cost note: unlike density (cheap to refine upward -- high density is the
FAST corner), log_N_dv is expensive to refine upward: measured median wall
time climbs from 152s at log_N_dv=14.0 to 1291s at 15.80, an 8.5x
increase, and the new midpoints sit throughout that range. Budget
accordingly (~8-9 CPU-hours per temperature at the interpolated per-point
costs, so multiple hours wall-clock even at 3 workers).

Usage: python3 build_lut_offset2.py [--processes N] [--dry-run]
Writes to output_lut_offset2/ -- separate from output_lut_coarse/ and
output_lut_offset/, so no resume-contamination risk with either.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_lut as b
import numpy as np

OFFSET2_ODIR = "/home/yasho379/magritte_rebuilt/production/output_lut_offset2/"

LOG_N_MIDPOINTS = [3.75, 4.25, 4.75, 5.25, 5.75]  # from build_lut_offset.py
COMPLETE_TEMPS = [18.0, 24.0, 30.0, 36.0]
LOG_NDV_MIDPOINTS = [14.225, 14.675, 15.125, 15.575]


def offset2_points():
    log_n_full, _, _ = b.grid_axes()
    log_n_all = sorted(set(round(float(x), 6) for x in log_n_full) |
                        set(round(float(x), 6) for x in LOG_N_MIDPOINTS))
    pts = []
    for log_n in log_n_all:
        for T in COMPLETE_TEMPS:
            for ndv in LOG_NDV_MIDPOINTS:
                pts.append((float(log_n), float(T), float(ndv)))
    return pts


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--processes', type=int, default=3)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--dv', type=float, default=b.REFERENCE_DV_KMS)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    pts = offset2_points()
    log_n_all, _, _ = b.grid_axes()
    n_density = len(set(round(float(x), 6) for x in log_n_all) | set(LOG_N_MIDPOINTS))
    print(f"offset2 grid: {len(pts)} points "
          f"({n_density} density x {len(COMPLETE_TEMPS)} temps x "
          f"{len(LOG_NDV_MIDPOINTS)} new log_N_dv midpoints)")

    b.build(odir=OFFSET2_ODIR, processes=a.processes, dv_kms=a.dv,
            limit=a.limit, dry_run=a.dry_run, points=pts)
