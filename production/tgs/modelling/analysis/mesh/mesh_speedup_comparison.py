"""Step 2: cube vs radial vs hybrid non-LTE meshes with an accepted solver profile, over the
validation nodes (analysis.validation.speedup_nodes), writing one resumable dir per mesh under
production/output_speedup_tests/ (never output_lut_gold/). Reference mesh for the comparison
tables: radial 14x260 (the geometrically validated one).

Usage: python3 -m analysis.mesh.mesh_speedup_comparison --solver <profile> [--processes 3]
           [--n-nodes 24] [--only radial12x140,cube] [--dry-run]
"""
import argparse
import os

import nh3hia.lut.build as b
from nh3hia.model3d.sphere import mesh_tag
from analysis.validation import speedup_nodes as N
from analysis.validation.speedup_run import OUT, WDIR, make_wdir

MESHES = {
    'cube': None,
    'radial12x140': {'kind': 'radial', 'n_shell': 12, 'base': 140, 'seed': 7},
    'radial14x260': {'kind': 'radial', 'n_shell': 14, 'base': 260, 'seed': 7},
    'hybrid': {'kind': 'hybrid', 'core_radius_frac': 0.4, 'surface_frac': 0.85, 'seed': 7},
}


MESH_PER_STRATUM = {'corner': 4, 'slice': 2, 'masing': 3, 'cold_lowdens': 4, 'hot_dense': 3,
                    'retrieval': 5, 'thin': 2, 'fill': 1}


def mesh_nodes(n=24):
    """24 nodes spread over every validation stratum (corners every other one, etc.)."""
    pts, label = N.nodes()
    out = []
    for name, k in MESH_PER_STRATUM.items():
        group = [p for p in pts if label[p] == name]
        if name == 'corner':      # a mix of low/high column and cold/hot
            group = [group[i] for i in (1, 2, 5, 7)]
        step = max(1, len(group) // k)
        out += group[::step][:k]
    return out[:n]


def mesh_dir(name, nrays=None):
    return os.path.join(OUT, f"mesh_{name}{'' if nrays in (None, 12) else f'_nr{nrays}'}", '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solver', required=True)
    ap.add_argument('--processes', type=int, default=3)
    ap.add_argument('--n-nodes', type=int, default=24)
    ap.add_argument('--only', default=None)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    make_wdir()
    pts = mesh_nodes(a.n_nodes)
    runs = [(name, 12, pts) for name in MESHES]
    runs += [('radial12x140', 24, pts[:: max(1, len(pts) // 4)][:4])]       # angular-quadrature sensitivity
    for name, nrays, nodes in runs:
        if a.only and name not in a.only.split(','):
            continue
        print(f"\n##### mesh {name} nrays={nrays}: {len(nodes)} nodes", flush=True)
        b.build(odir=mesh_dir(name, nrays), processes=a.processes, points=nodes, solver=a.solver,
                mesh=MESHES[name], nrays=nrays, wdir=WDIR, keep_model_file=False,
                save_populations=True, dry_run=a.dry_run)


if __name__ == '__main__':
    main()
