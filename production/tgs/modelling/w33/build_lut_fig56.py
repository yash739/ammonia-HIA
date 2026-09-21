"""Small, targeted grid at Stutzki & Winnewisser's (1985) own Fig. 5/6
parameters -- T_k in {18, 26, 36} K, n'_H2 in {10^3.5, 10^5.0, 10^7.0}
cm^-3 (grid.py:29-32, matching the printed figure caption exactly), full
log_N_dv sweep (9 points, matching gold's own axis) -- 3 x 3 x 9 = 81
models.

Built as its own small grid, separate from gold, for two reasons: (1) gold
never sampled T=26K at all (its T_AXIS is 9,12,...,48 in steps of 3, which
skips 26 -- Stutzki's own reference temperature for Figs. 5-6, previously
worked around by substituting 27K and flagging the ~4% offset); (2) gold's
tau_main column is known-wrong (see important_notes/
magritte-quirks-2026-09-18.md Sec 6, and the session that added this
script) and is being left as-is rather than patched retroactively --
this grid is built fresh, after the tau_main fix in nh3_NLTE_sphere.py,
so every row here has a genuinely correct tau_main from the start.

Usage: python3 build_lut_fig56.py [--processes N] [--dry-run]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_lut as b

FIG56_ODIR = "/home/yasho379/magritte_rebuilt/production/output_lut_fig56/"

LOG_N_AXIS = [3.5, 5.0, 7.0]
T_AXIS = [18.0, 26.0, 36.0]
LOG_NDV_AXIS = [14.0, 14.225, 14.45, 14.675, 14.9, 15.125, 15.35, 15.575, 15.8]


def fig56_points():
    return [(float(a), float(bb), float(c))
            for a in LOG_N_AXIS for bb in T_AXIS for c in LOG_NDV_AXIS]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--processes', type=int, default=3)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--dv', type=float, default=b.REFERENCE_DV_KMS)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    pts = fig56_points()
    print(f"fig56 grid: {len(pts)} total cross-product points "
          f"({len(LOG_N_AXIS)} density x {len(T_AXIS)} T x {len(LOG_NDV_AXIS)} log_N_dv)")

    b.build(odir=FIG56_ODIR, processes=a.processes, dv_kms=a.dv,
            limit=a.limit, dry_run=a.dry_run, points=pts)
