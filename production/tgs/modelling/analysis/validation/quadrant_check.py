"""Quadrant check on the gold grid: every converged model is a static,
homogeneous sphere with no imposed kinematics, so it should land in quadrant
II of the (HIA_IS, HIA_OS) plane (HIA_IS < 1, HIA_OS > 1; Zhou et al. 2020,
Wu et al. 2024) and never in quadrants III (infall) or IV (forbidden).

Prints the quadrant counts and writes the two point sets the paper's
fig:quadrant_grid plots (Overleaf data/quadrant_lut_main.csv and
data/quadrant_lut_cold.csv).

Usage: python3 -m analysis.validation.quadrant_check [--lut PATH] [--outdir DIR]
"""
import argparse
import os

import pandas as pd

from nh3hia import paths


def classify(d):
    """Return boolean masks for quadrants I-IV of the (HIA_IS, HIA_OS) plane."""
    q2 = (d.HIA_IS < 1) & (d.HIA_OS > 1)
    q1 = (d.HIA_IS >= 1) & (d.HIA_OS >= 1)
    q3 = (d.HIA_IS < 1) & (d.HIA_OS <= 1)
    q4 = (d.HIA_IS >= 1) & (d.HIA_OS < 1)
    return {'I': q1, 'II': q2, 'III': q3, 'IV': q4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lut', default=paths.LUT_GOLD_CSV)
    ap.add_argument('--outdir', default=os.path.join(paths.LUT_GOLD, 'results'))
    a = ap.parse_args()

    d = pd.read_csv(a.lut)
    d = d[(d.Status == 'SUCCESS') & (d.convergence_ok == True)]  # noqa: E712
    q = classify(d)
    print(f"{len(d)} converged models")
    for name, m in q.items():
        s = d[m]
        extra = ''
        if name != 'II' and len(s):
            extra = (f"  T values {sorted(float(t) for t in s.T_cloud.unique())}, "
                     f"max HIA_IS {s.HIA_IS.max():.4f}")
        print(f"  quadrant {name:>3}: {int(m.sum()):5d}{extra}")

    os.makedirs(a.outdir, exist_ok=True)
    d[q['II']][['HIA_IS', 'HIA_OS']].round(5).to_csv(
        os.path.join(a.outdir, 'quadrant_lut_main.csv'), index=False)
    d[~q['II']][['HIA_IS', 'HIA_OS']].round(5).to_csv(
        os.path.join(a.outdir, 'quadrant_lut_cold.csv'), index=False)
    print('wrote quadrant_lut_main.csv / quadrant_lut_cold.csv to', a.outdir)


if __name__ == '__main__':
    main()
