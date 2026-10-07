"""Shareable figures comparing the cube (production) and radial seed meshes of the 3D non-LTE sphere.

fig_mesh_geometry.png : the three point clouds (cube 377 pts, radial 12x140 935 pts, radial 14x260 1702 pts)
                        as a central slice, and how many points sit at each radius.
fig_mesh_ratio_differences.png : how the non-LTE line ratios differ between meshes at the same physical
                        parameters (24 nodes spanning the grid; reference radial 14x260) and what each mesh costs.

Usage: python3 -m analysis.mesh.mesh_visuals
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from nh3hia.model3d.sphere import build_point_cloud
from analysis.mesh.mesh_speedup_comparison import MESHES
from analysis.validation.speedup_run import OUT
from analysis.mesh.mesh_speedup_report import load, rel, KEY

FIG = os.path.join(OUT, 'figures')
NAMES = {'cube': 'Cube seed (production)', 'radial12x140': 'Radial 12 shells x 140', 'radial14x260': 'Radial 14 shells x 260'}
COL = {'cube': '#c0392b', 'radial12x140': '#2471a3', 'radial14x260': '#1e8449'}


def cloud(name):
    pos, nb = build_point_cloud(1.0, 1.0, 1.15, resolution=10, fov_pad_factor=1.15, mesh=MESHES[name])
    return np.asarray(pos), nb


def geometry():
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 8.6), gridspec_kw=dict(height_ratios=[3, 1.8]))
    for j, name in enumerate(NAMES):
        pos, nb = cloud(name)
        r = np.linalg.norm(pos, axis=1)
        inside = r <= 1.0 + 1e-9
        core = r < 0.05                   # the 0.01 R inner boundary shell (108 points), identical in every mesh
        mid = inside & (r >= 0.05) & (r < 0.95)
        a = ax[0, j]
        a.add_patch(Circle((0, 0), 1.0, fill=False, color='k', lw=1.2))
        a.scatter(pos[~inside, 0], pos[~inside, 1], s=5, color='0.7', zorder=2)
        a.scatter(pos[inside & ~core, 0], pos[inside & ~core, 1], s=9, color=COL[name], alpha=0.8, zorder=3)
        a.scatter([0], [0], s=40, color='k', marker='+', zorder=4)
        a.set_xlim(-1.2, 1.2); a.set_ylim(-1.2, 1.2); a.set_aspect('equal')
        a.set_title(f"{NAMES[name]}: {len(pos)} points\n{mid.sum()} between 0.05 R and 0.95 R, {(r >= 0.95)[inside].sum()} on the limb shell", fontsize=10)
        a.set_xlabel('x / R'); a.set_ylabel('y / R') if j == 0 else None
        a.text(0, -1.17, 'all points projected on the sky plane; + = 108 centre points, grey = outside', ha='center', fontsize=7, color='0.35')
        b = ax[1, j]
        bins = np.linspace(0.05, 1, 20)
        h, _ = np.histogram(r[inside], bins=bins)
        b.bar(0.5 * (bins[1:] + bins[:-1]), h, width=0.047, color=COL[name])
        b.set_xlabel('radius r / R'); b.set_ylabel('points per 0.05 R') if j == 0 else None
        b.set_xlim(0, 1)
        b.text(0.03, 0.92, f"{(r[inside] < 0.4).sum() - core.sum()} points between 0.05 R and 0.4 R", transform=b.transAxes, fontsize=9, va='top')
    fig.suptitle('Same sphere, three seed meshes: the cube leaves almost the whole interior empty', fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(os.path.join(FIG, 'fig_mesh_geometry.png'), dpi=150)
    plt.close(fig)


def ratio_differences():
    cube, r12, r14 = load('mesh_cube'), load('mesh_radial12x140'), load('mesh_radial14x260')
    cols = ['R_01_MAIN', 'R_10_MAIN', 'R_12_MAIN', 'R_21_MAIN', 'R_22_MAIN']
    lab = [r'$F_1\!:0\to1$ / main', r'$F_1\!:1\to0$ / main', r'$F_1\!:1\to2$ / main', r'$F_1\!:2\to1$ / main', '(2,2) / (1,1)']
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw=dict(width_ratios=[2.3, 1]))
    a = ax[0]
    for k, (nm, d) in enumerate((('cube', cube), ('radial12x140', r12))):
        x = rel(d, r14, cols)
        for i, c in enumerate(cols):
            v = x[c].dropna().values
            a.scatter(np.full(len(v), i + (-0.15 if k == 0 else 0.15)) + np.random.default_rng(1).uniform(-0.05, 0.05, len(v)),
                      v, s=22, color=COL[nm], alpha=0.75, label=NAMES[nm] if i == 0 else None)
            a.hlines(np.median(v), i + (-0.15 if k == 0 else 0.15) - 0.1, i + (-0.15 if k == 0 else 0.15) + 0.1, color='k', lw=2)
    a.axhline(0, color='0.4', lw=0.8)
    a.set_xticks(range(len(cols))); a.set_xticklabels(lab)
    a.set_ylabel('difference from radial 14x260 at the same parameters [%]')
    a.set_title('Non-LTE hyperfine ratios (16 nodes spanning the grid; black bar = median)')
    a.legend(); a.grid(alpha=0.25)
    a.set_ylim(-45, 45)
    b = ax[1]
    names = ['cube', 'radial12x140', 'radial14x260']
    pts = [377, 935, 1702]
    wall = []
    for nm, d in zip(names, (cube, r12, r14)):
        m = d.merge(r14[KEY], on=KEY)               # common nodes only
        wall.append(m.wall_s.median())
    b.bar(range(3), wall, color=[COL[n] for n in names])
    for i, (w, p) in enumerate(zip(wall, pts)):
        b.text(i, w, f"{w:.0f} s\n{p} pts", ha='center', va='bottom', fontsize=9)
    b.set_xticks(range(3)); b.set_xticklabels(['cube', 'radial\n12x140', 'radial\n14x260'])
    b.set_ylabel('median wall time per model [s]'); b.set_title('Cost (fast solver profile, same 16 nodes)')
    b.set_ylim(0, max(wall) * 1.25)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'fig_mesh_ratio_differences.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    geometry()
    ratio_differences()
    print('saved to', FIG)
