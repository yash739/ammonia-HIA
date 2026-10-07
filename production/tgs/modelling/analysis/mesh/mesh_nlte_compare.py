"""Compare the non-LTE mesh runs of analysis.mesh.mesh_nlte_comparison.

For every point and mesh: the four satellite/main ratios, HIA_IS, HIA_OS,
(2,2)/(1,1), the held-out (2,1)/(1,1), the disc-integrated main-line T_B,
the (1,1) central optical depth and the convergence; each expressed relative
to the radial 14x260 mesh (the geometrically validated reference). From the
saved model files (converged populations): the (1,1) F1=0->1 population
ratio (n_u/g_u)/(n_l/g_l) by radius (> 1 = inverted).

Writes production/output_mesh_nlte/comparison.csv and prints a table.
Usage: python3 -m analysis.mesh.mesh_nlte_compare [--no-populations]
"""
import argparse
import os

import numpy as np
import pandas as pd

from nh3hia import paths
from nh3hia.model3d.sphere import mesh_tag
from analysis.mesh.mesh_nlte_comparison import OUT, SETS, MESHES, ALL_POINTS

QUANTITIES = ['R_01_MAIN', 'R_10_MAIN', 'R_12_MAIN', 'R_21_MAIN', 'HIA_IS', 'HIA_OS',
              'R_22_MAIN', 'R_21_11', 'A_MAIN', 'tau_main']
REFERENCE = 'radial14x260'


def load_sets():
    rows = []
    label_of = {tuple(np.round(v, 3)): k for k, v in ALL_POINTS.items()}
    for name, mesh, nrays, _ in SETS:
        f = os.path.join(OUT, name, 'results', 'lut_dv0.30.csv')
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        d['set'] = name
        d['mesh_name'] = mesh
        d['point'] = [label_of.get((round(a, 3), round(b, 3), round(c, 3)), '?')
                      for a, b, c in zip(d.log_n_H2, d.T_cloud, d.log_N_dv)]
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def inversion_by_radius(row, mesh):
    """(n_u/g_u)/(n_l/g_l) for (1,1) F1=0->1, median in radial bins."""
    import magritte.core as magritte
    from nh3hia.escape1d import model as m
    M = m.NH3Model()
    iu = [i for i, q in enumerate(M.qn) if q == (1, 1, -1, 0)][0]
    il = [i for i, q in enumerate(M.qn) if q == (1, 1, 1, 1)][0]
    f = os.path.join(paths.TGS, 'model_files',
                     f"NLTE_analytic_sphere_nh3_{row.XNH3}_{row.numberdensity:.2e}_{row.radius_sphere:.2e}"
                     f"_{row.vturb}_{float(row.T_cloud)}_pad{row.fov_pad_factor}_res{int(row.resolution)}"
                     f"_nr{int(row.nrays)}{mesh_tag(mesh)}.hdf5")
    if not os.path.exists(f):
        return {}
    mod = magritte.Model(f)
    nl = mod.lines.lineProducingSpecies[0].linedata.nlev
    pop = np.array(mod.lines.lineProducingSpecies[0].population).reshape(-1, nl)
    if pop.sum() == 0:
        return {}
    rr = np.linalg.norm(np.array(mod.geometry.points.position), axis=1)
    rr = rr / (rr.max() / row.fov_pad_factor)
    ratio = (pop[:, iu] / M.g[iu]) / (pop[:, il] / M.g[il])
    ok = pop.sum(axis=1) > 0
    out = {}
    for lo, hi in ((0.0, 0.3), (0.3, 0.7), (0.7, 1.0)):
        s = ok & (rr >= lo) & (rr < hi)
        out[f'inv01_r{lo:.1f}-{hi:.1f}'] = float(np.median(ratio[s])) if s.any() else np.nan
    inside = ok & (rr < 1.0)
    out['inv01_frac_inverted'] = float(np.mean(ratio[inside] > 1)) if inside.any() else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-populations', action='store_true')
    a = ap.parse_args()
    d = load_sets()
    if d.empty:
        print('no results yet')
        return
    if not a.no_populations:
        extra = [inversion_by_radius(r, MESHES[r.mesh_name]) for r in d.itertuples()]
        d = pd.concat([d, pd.DataFrame(extra)], axis=1)
    ref = d[d.set == REFERENCE].set_index('point')
    for q in QUANTITIES:
        d[f'{q}_rel_to_radial'] = [
            (r[q] / ref.loc[r.point, q] - 1.0) if r.point in ref.index else np.nan
            for _, r in d.iterrows()]
    out = os.path.join(OUT, 'comparison.csv')
    d.to_csv(out, index=False)
    pd.set_option('display.width', 250)
    cols = ['point', 'set', 'final_convergence', 'npoints', 'wall_s'] + \
           [f'{q}_rel_to_radial' for q in ('R_01_MAIN', 'R_10_MAIN', 'R_12_MAIN', 'R_21_MAIN',
                                           'R_22_MAIN', 'R_21_11', 'A_MAIN', 'tau_main')]
    cols += [c for c in d.columns if c.startswith('inv01')]
    print(d.sort_values(['point', 'set'])[cols].round(3).to_string(index=False))
    print('saved', out)


if __name__ == '__main__':
    main()
