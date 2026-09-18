import sys, numpy as np
sys.path.insert(0,'.')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nh3_NLTE_sphere as S

cases = [('Cube, pad 1.0 (legacy-style)', dict(fov_pad_factor=1.0, mesh=None), 1.0),
         ('Cube, pad 1.15 (gold)', dict(fov_pad_factor=1.15, mesh=None), 1.15),
         ('Radial, pad 1.0 (platinum)', dict(fov_pad_factor=1.0, mesh={'kind':'radial'}), 1.0)]
fig = plt.figure(figsize=(18, 16))
for i, (title, kw, pad) in enumerate(cases):
    p, nb = S.build_point_cloud(1.0, 1.0, pad, resolution=10, **kw)
    r = np.linalg.norm(p, axis=1)
    inner = np.abs(r - 0.01) < 1e-9
    outer = np.abs(r - pad) < 1e-9
    body = ~inner & ~outer
    c = np.where(inner, 'tab:purple', np.where(outer, 'tab:orange', np.where(r <= 1.0, 'tab:blue', 'tab:gray')))

    ax = fig.add_subplot(3, 3, 3*i + 1, projection='3d')
    ax.scatter(*p[body & (r <= 1)].T, s=5, c='tab:blue', alpha=0.6, label='inside sphere')
    ax.scatter(*p[body & (r > 1)].T, s=5, c='tab:gray', alpha=0.6, label='outside sphere')
    ax.scatter(*p[outer].T, s=3, c='tab:orange', alpha=0.35, label='outer boundary shell')
    ax.scatter(*p[inner].T, s=6, c='tab:purple', label='inner boundary shell (0.01R)')
    ax.set_box_aspect((1, 1, 1)); ax.set_title(f'{title}\n{len(p)} points', fontsize=12)
    if i == 0: ax.legend(fontsize=8, loc='upper left')

    ax = fig.add_subplot(3, 3, 3*i + 2)
    sl = np.abs(p[:, 2]) < 0.12
    ax.scatter(p[sl, 0], p[sl, 1], s=12, c=c[sl])
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), 'k-', lw=1.5, label='sphere surface r = R')
    if pad > 1: ax.plot(pad*np.cos(th), pad*np.sin(th), 'k:', lw=1, label=f'boundary r = {pad}R')
    ax.set_aspect('equal'); ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
    ax.set_title(f'Equatorial slice |z| < 0.12R ({sl.sum()} points)', fontsize=11)
    ax.set_xlabel('x / R'); ax.set_ylabel('y / R'); ax.legend(fontsize=8, loc='lower right')

    ax = fig.add_subplot(3, 3, 3*i + 3)
    sel = ~inner & ~outer & (r <= 1.0)
    rs = np.sort(r[sel])
    ax.step(rs, np.arange(1, len(rs)+1)/len(rs), where='post', color='tab:blue', lw=2,
            label='cumulative fraction of interior points')
    rr = np.linspace(0, 1, 200)
    ax.plot(rr, rr**3, 'k--', lw=1.5, label='uniform sampling: (r/R)$^3$')
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel('r / R'); ax.set_ylabel('fraction of non-boundary points with radius < r')
    ax.set_title(f'{len(rs)} non-boundary points inside R', fontsize=11)
    ax.legend(fontsize=9, loc='upper left'); ax.grid(alpha=0.3)
fig.suptitle('Magritte point clouds for a uniform sphere (dimensionless, R = 1, resolution = 10)', fontsize=15)
plt.tight_layout()
out = '/home/yasho379/magritte_rebuilt/scratch/plots/mesh_point_clouds_cube_vs_radial.png'
plt.savefig(out, dpi=110); print('saved', out)
