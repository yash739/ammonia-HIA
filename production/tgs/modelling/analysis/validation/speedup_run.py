"""Run the solver-profile validation (Step 1): each variant over its nodes into its own
resumable output dir under production/output_speedup_tests/, using the production builder.

Never touches output_lut_gold/ or production/tgs/model_files/: a private working dir
(output_speedup_tests/wdir, symlinks to the LAMDA files) holds the model-file cache, and
model files are deleted after use (keep_model_file=False).

Usage: python3 -m analysis.validation.speedup_run --set stages [--processes 3] [--dry-run]
       sets: determinism stages ablations ladder_levels ladder_prec all
"""
import argparse
import os

import pandas as pd

import nh3hia.lut.build as b
from nh3hia import paths
from nh3hia.model3d import solver_profiles as sp
from analysis.validation import speedup_nodes as N

OUT = os.path.join(paths.PRODUCTION, 'output_speedup_tests')
WDIR = os.path.join(OUT, 'wdir', '')


def make_wdir():
    os.makedirs(os.path.join(WDIR, 'model_files'), exist_ok=True)
    for name in (sp.LAMDA_FULL, sp.LAMDA_36):
        dst = os.path.join(WDIR, name)
        if not os.path.exists(dst):
            os.symlink(os.path.join(paths.TGS, name), dst)


def variant_dir(profile):
    return os.path.join(OUT, 'v_' + sp.canonical(profile).replace('+', '-'), '')


def gold_wall(points):
    g = pd.read_csv(paths.LUT_GOLD_CSV)
    out = {}
    for p in points:
        r = g[(abs(g.log_n_H2 - p[0]) < 1e-6) & (abs(g.T_cloud - p[1]) < 1e-6) & (abs(g.log_N_dv - p[2]) < 1e-6)]
        out[p] = float(r.wall_s.iloc[0]) if len(r) else float('nan')
    return out


def plan(which):
    pts, _ = N.nodes()
    sub16 = N.subset(16)
    hot = [p for p in pts if p[1] >= 42.0 and p[0] >= 7.5] + [p for p in pts if p in N.STRATA['masing']]
    prec_nodes = N.subset(8)
    jobs = []
    if which in ('determinism', 'all'):
        w = gold_wall(pts)
        cheap = sorted([p for p in pts if w[p] == w[p] and w[p] < 450], key=lambda p: w[p])[:6]
        jobs.append(('legacy', cheap, None))
    if which in ('stages', 'all'):
        jobs += [(s, pts, None) for s in N.STAGES]
    if which in ('ablations', 'all'):
        jobs += [(s, sub16, None) for s in N.ABLATIONS]
    if which in ('ladder_levels', 'all'):
        jobs += [(s, hot, None) for s in N.LEVEL_LADDER]
    if which in ('ladder_prec', 'all'):
        jobs += [(s, prec_nodes, 2000 if s.endswith('prec8') else None) for s in N.PREC_LADDER]
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='stages')
    ap.add_argument('--processes', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    make_wdir()
    for profile, pts, max_nlte in plan(a.set):
        print(f"\n##### {profile}: {len(pts)} nodes -> {variant_dir(profile)}", flush=True)
        b.build(odir=variant_dir(profile), processes=a.processes, points=pts, solver=profile,
                wdir=WDIR, keep_model_file=False, save_populations=True, max_nlte=max_nlte,
                dry_run=a.dry_run)


if __name__ == '__main__':
    main()
