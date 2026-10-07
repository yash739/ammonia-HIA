"""Rebuild of the full gold table (16 log n x 14 T x 11 log N/dv = 2464 models, Delta v = 0.3
km/s) on a RADIAL mesh with a validated fast solver profile, into its own directory.
output_lut_gold/ (cube mesh, legacy solver) is never read for writing, never touched.

The mesh, solver profile and processes come from the command line so the exact settings are
recorded; the CSV carries mesh_kind and solver_profile columns. Points are ordered coarse-to-fine
(every 2nd index per axis first), so a usable sub-lattice exists whenever the run is stopped.
Model files are deleted after use; converged fractional populations are saved per model
(populations/<key>.npz). Resumable.

Usage: python3 -m analysis.grids.build_gold_grid_radial --solver <profile> \
           --n-shell 12 --base 140 [--processes 3] [--odir production/output_lut_gold_radial] [--dry-run]
"""
import argparse
import os

import nh3hia.lut.build as b
from nh3hia import paths
from analysis.grids.build_gold_grid import gold_points
from nh3hia.lut.axes import LOG_N_AXIS, LOG_N_EXT_AXIS, LOG_NDV_AXIS, LOG_NDV_EXT_AXIS

DEFAULT_ODIR = os.path.join(paths.PRODUCTION, 'output_lut_gold_radial', '')


def full_gold_points():
    return gold_points(LOG_N_AXIS + LOG_N_EXT_AXIS, LOG_NDV_AXIS + LOG_NDV_EXT_AXIS)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--solver', required=True,
                    help="profile string, or 'regional' = close+lev36+nq12+prec4 up to log N/dv 15.8, "
                         "close+nq12+prec4 (all 48 levels) above it")
    ap.add_argument('--n-shell', type=int, default=12)
    ap.add_argument('--base', type=int, default=140)
    ap.add_argument('--processes', type=int, default=3)
    ap.add_argument('--odir', default=DEFAULT_ODIR)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    gold = os.path.realpath(paths.LUT_GOLD)
    assert not os.path.realpath(a.odir).startswith(gold), 'refusing to write inside the gold grid'
    pts = full_gold_points()
    assert len(pts) == 2464, len(pts)
    mesh = {'kind': 'radial', 'n_shell': a.n_shell, 'base': a.base, 'seed': 7}
    print(f"radial gold rebuild: {len(pts)} points, mesh {mesh}, solver {a.solver}, odir {a.odir}", flush=True)
    wdir = os.path.join(a.odir, 'wdir', '')
    os.makedirs(os.path.join(wdir, 'model_files'), exist_ok=True)
    for name in os.listdir(paths.TGS):
        if name.startswith('p-nh3@loreau') and not os.path.exists(os.path.join(wdir, name)):
            os.symlink(os.path.join(paths.TGS, name), os.path.join(wdir, name))
    from nh3hia.model3d import solver_profiles
    solver = (solver_profiles.regional('close+lev36+nq12+prec4', 'close+nq12+prec4')
              if a.solver == 'regional' else a.solver)
    b.build(odir=a.odir, processes=a.processes, points=pts, mesh=mesh, solver=solver, wdir=wdir,
            keep_model_file=False, save_populations=True, order='coarse_to_fine', limit=a.limit,
            dry_run=a.dry_run)
