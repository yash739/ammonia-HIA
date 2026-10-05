"""Non-LTE cube vs radial vs hybrid mesh comparison at the points that carry
Paper I's numbers.

Every run goes through the production grid builder (nh3hia.lut.build) with
production settings (pad 1.15, 16x16 images, nrays 12, max_NLTE 250, the
same spectral extraction), changing only the mesh -- so the ratios, (2,2)/(1,1)
and the held-out (2,1)/(1,1) are directly comparable with the gold grid.
Model files written since 2026-09-21 keep their converged populations, so
tau(b) images and populations by radius can be re-derived afterwards from
production/tgs/model_files/ without re-solving.

Points (exact grid nodes nearest the 3D Loreau best fits, so the cube runs
are directly comparable with the gold rows):
  OMC S3 (7.5, 36, 15.35)   OMC S4 (7.5, 39, 15.35)   S106 (0,0) (7.0, 27, 15.575)
  W33 A  (8.0, 36, 15.8)    W33 B  (8.0, 33, 16.025)
plus one deep-masing point (5.0, 36, 15.8) and the n = 1e7 column of the 18 K
Figs. 5-6 slice at five columns. An nrays = 48 radial run at OMC S3 and W33 B
tests the old (cube, LTE-only) nrays-insensitivity result.

Usage: python3 -m analysis.mesh.mesh_nlte_comparison [--processes 3] [--dry-run]
"""
import argparse
import os

from nh3hia import paths
import nh3hia.lut.build as b

OUT = os.path.join(paths.PRODUCTION, 'output_mesh_nlte')

FIT_POINTS = {
    'OMC S3': (7.5, 36.0, 15.35),
    'OMC S4': (7.5, 39.0, 15.35),
    'S106 (0,0)': (7.0, 27.0, 15.575),
    'W33 A': (8.0, 36.0, 15.8),
    'W33 B': (8.0, 33.0, 16.025),
}
MASING_POINT = {'masing 36 K': (5.0, 36.0, 15.8)}
SLICE_18K = {f'18 K slice {c}': (7.0, 18.0, c) for c in (14.0, 14.45, 14.9, 15.35, 15.8)}
ALL_POINTS = {**FIT_POINTS, **MASING_POINT, **SLICE_18K}

MESHES = {
    'cube': None,
    'hybrid': {'kind': 'hybrid', 'core_radius_frac': 0.4, 'surface_frac': 0.85, 'seed': 7},
    'radial12x140': {'kind': 'radial', 'n_shell': 12, 'base': 140, 'seed': 7},
    'radial14x260': {'kind': 'radial', 'n_shell': 14, 'base': 260, 'seed': 7},
}
# (set name, mesh name, nrays, points) in run order: cheap first.
SETS = [(m, m, None, ALL_POINTS) for m in MESHES] + [
    ('radial14x260_nr48', 'radial14x260', 48, {k: FIT_POINTS[k] for k in ('OMC S3', 'W33 B')}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--processes', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', default=None, help='comma-separated set names')
    a = ap.parse_args()
    for name, mesh, nrays, pts in SETS:
        if a.only and name not in a.only.split(','):
            continue
        print(f"\n===== {name} (nrays={nrays or b.NRAYS}) =====", flush=True)
        b.build(odir=os.path.join(OUT, name, ''), processes=a.processes, points=list(pts.values()),
                mesh=MESHES[mesh], nrays=nrays, dry_run=a.dry_run)


if __name__ == '__main__':
    main()
