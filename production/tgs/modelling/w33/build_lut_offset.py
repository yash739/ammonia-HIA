"""The second, denser/offset grid -- fills in the two regions the coarse
grid's own hold-one-out interpolation check (see the report's interpolation
validity section) found worst-behaved, rather than uniformly halving every
axis:

  - log_n_H2 midpoints at the LOW-density end (3.75, 4.25, 4.75, 5.25, 5.75),
    where R_10_MAIN's non-monotonic valley was hardest to interpolate,
    crossed with the coarse grid's own complete temperatures (18/24/30/36 K)
    and log_N_dv axis.
  - one new temperature, 27 K (between the coarse grid's 24 and 36 K, and
    incidentally close to Stutzki's own Fig. 5/6 reference of 26 K),
    crossed with the FULL existing density axis, since R_01_MAIN's
    non-monotonic amplitude peak was worst at HIGH density specifically,
    not low.

145 models total (100 + 45), reusing build_lut.build()'s exact resumable
machinery via its new `points=` override -- no new build logic.

Usage: python3 build_lut_offset.py [--processes N] [--dry-run]
Writes to output_lut_offset/ -- a separate directory from output_lut_coarse/,
so no resume-contamination risk between the two grids.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_lut as b

OFFSET_ODIR = "/home/yasho379/magritte_rebuilt/production/output_lut_offset/"

LOG_N_MIDPOINTS = [3.75, 4.25, 4.75, 5.25, 5.75]
COMPLETE_TEMPS = [18.0, 24.0, 30.0, 36.0]
NEW_TEMP = 27.0


def offset_points():
    log_n_full, _, log_ndv = b.grid_axes()
    pts = []
    # Density refinement at the coarse grid's complete temperatures.
    for log_n in LOG_N_MIDPOINTS:
        for T in COMPLETE_TEMPS:
            for ndv in log_ndv:
                pts.append((float(log_n), float(T), float(ndv)))
    # Temperature refinement (27K) across the FULL existing density axis.
    for log_n in log_n_full:
        for ndv in log_ndv:
            pts.append((float(log_n), NEW_TEMP, float(ndv)))
    return pts


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--processes', type=int, default=3)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--dv', type=float, default=b.REFERENCE_DV_KMS)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    pts = offset_points()
    print(f"offset grid: {len(pts)} points "
          f"({len(LOG_N_MIDPOINTS)}x{len(COMPLETE_TEMPS)}x5 density-refinement "
          f"+ {len(b.grid_axes()[0])}x1x5 temperature-refinement)")

    b.build(odir=OFFSET_ODIR, processes=a.processes, dv_kms=a.dv,
            limit=a.limit, dry_run=a.dry_run, points=pts)
