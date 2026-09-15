"""The final, non-lumpy grid: a true full cross-product across all three
axes, replacing the coarse+offset1+offset2 patchwork with one properly
sampled table. Supersedes build_lut_offset2.py (stopped after 25 rows,
merged into output_lut_offset/ so that work isn't lost).

Final axes (all three now uniform, no more per-temperature "islands"):
  - log_n_H2:  14 points, {3.5..6.0 step 0.25} + {6.5,7.0,7.5} (unchanged
    from what coarse+offset1 already established)
  - T_cloud:   14 points, 9-48K in exact 3K steps -- extends the old
    non-uniform 6-point axis (12,18,24,30,36,48, gaps up to 12K) down to
    9K and fills every gap to a uniform 3K step, so EVERY temperature now
    gets the full density x log_N_dv resolution (the "gold block"
    treatment that previously only applied to T in {18,24,30,36}).
  - log_N_dv:  9 points, 14.0-15.8 in exact 0.225 steps (the axis
    build_lut_offset2.py was midway through populating)

Full cross product: 14 x 14 x 9 = 1764 points. This script computes only
the ones NOT already present in output_lut_coarse/ or output_lut_offset/
(which now also holds offset2's 25 completed rows, merged in) -- both are
read directly rather than relying on build_lut.build()'s own single-
directory resume logic, since the already-done points are split across
two prior directories.

Writes to output_lut_gold/ (a new directory; the two prior dirs are left
untouched as historical record). Resumable within itself the normal way.

Usage: python3 build_lut_gold.py [--processes N] [--dry-run]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import build_lut as b

GOLD_ODIR = "/home/yasho379/magritte_rebuilt/production/output_lut_gold/"
PRIOR_DIRS = [
    "/home/yasho379/magritte_rebuilt/production/output_lut_coarse/",
    "/home/yasho379/magritte_rebuilt/production/output_lut_offset/",
]

LOG_N_AXIS = [3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.5, 7.0, 7.5]
T_AXIS = [float(t) for t in range(9, 49, 3)]  # 9,12,...,48 -- 14 points
LOG_NDV_AXIS = [14.0, 14.225, 14.45, 14.675, 14.9, 15.125, 15.35, 15.575, 15.8]


def _key(a, bb, c, nd=4):
    return (round(float(a), nd), round(float(bb), nd), round(float(c), nd))


def already_done_keys():
    keys = set()
    for d in PRIOR_DIRS:
        path = os.path.join(d, "results", "lut_dv0.30.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for a, bb, c in zip(df['log_n_H2'], df['T_cloud'], df['log_N_dv']):
            keys.add(_key(a, bb, c))
    return keys


def gold_points():
    done = already_done_keys()
    pts = []
    for a in LOG_N_AXIS:
        for bb in T_AXIS:
            for c in LOG_NDV_AXIS:
                if _key(a, bb, c) not in done:
                    pts.append((float(a), float(bb), float(c)))
    return pts, done


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--processes', type=int, default=3)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--dv', type=float, default=b.REFERENCE_DV_KMS)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    pts, done = gold_points()
    full = len(LOG_N_AXIS) * len(T_AXIS) * len(LOG_NDV_AXIS)
    print(f"gold grid: {full} total cross-product points "
          f"({len(LOG_N_AXIS)} density x {len(T_AXIS)} T x {len(LOG_NDV_AXIS)} log_N_dv)")
    print(f"already done (coarse+offset merged): {len(done)}")
    print(f"new points to run: {len(pts)}")

    b.build(odir=GOLD_ODIR, processes=a.processes, dv_kms=a.dv,
            limit=a.limit, dry_run=a.dry_run, points=pts)
