"""Stutzki & Winnewisser Figs. 5/6 grid on the radial 12x140 mesh with the regional fast solver profile, into
production/output_lut_fig56_radial/ (the cube-mesh output_lut_fig56/ is never touched). Same settings as
build_gold_grid_radial, so Figs. 5/6 are drawn from models whose central optical depth is right (the cube mesh
halves it). Default --temps 26: the 18 K and 36 K slices are nodes of the radial gold table itself
(output_lut_gold_radial/, T axis 9..48 in 3 K steps), so only Stutzki's 26 K -- which the gold axis skips -- needs
building (8 densities x 9 columns = 72 models). Resumable.

Usage: python3 -m analysis.grids.build_fig56_grid_radial [--processes N] [--dry-run]
"""
import argparse
import os

import nh3hia.lut.build as b
from nh3hia import paths
from nh3hia.model3d import solver_profiles
from analysis.grids.build_fig56_grid import fig56_points

ODIR = os.path.join(paths.PRODUCTION, 'output_lut_fig56_radial', '')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--processes', type=int, default=1)
    ap.add_argument('--n-shell', type=int, default=12)
    ap.add_argument('--base', type=int, default=140)
    ap.add_argument('--temps', default='26', help='comma-separated T list (default 26; use 18,26,36 for all 216)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    temps = [float(t) for t in a.temps.split(',')]
    pts = [p for p in fig56_points() if p[1] in temps]
    assert len(pts) == 72 * len(temps), len(pts)
    mesh = {'kind': 'radial', 'n_shell': a.n_shell, 'base': a.base, 'seed': 7}
    wdir = os.path.join(ODIR, 'wdir', '')
    os.makedirs(os.path.join(wdir, 'model_files'), exist_ok=True)
    for name in os.listdir(paths.TGS):
        if name.startswith('p-nh3@loreau') and not os.path.exists(os.path.join(wdir, name)):
            os.symlink(os.path.join(paths.TGS, name), os.path.join(wdir, name))
    solver = solver_profiles.regional('close+lev36+nq12+prec4', 'close+nq12+prec4')
    print(f"fig56 radial rebuild: {len(pts)} points, mesh {mesh}", flush=True)
    b.build(odir=ODIR, processes=a.processes, points=pts, mesh=mesh, solver=solver, wdir=wdir,
            keep_model_file=False, save_populations=True, order='coarse_to_fine', dry_run=a.dry_run)
