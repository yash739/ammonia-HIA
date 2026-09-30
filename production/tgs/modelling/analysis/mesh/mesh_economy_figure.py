"""Paper I, Work item -1 (mesh economy): publication-style figures matching
fig:chord_experiment's own layout/conventions in paper1_draft.tex --
tau(b) recovered two independent ways (brightness inversion, Magritte's own
optical-depth image), against the analytic chord law, now with the
two-band hybrid mesh added as a third curve. Plus a companion NLTE-cost
bar chart (the number that actually decided Work item -1's outcome).

Reuses the same validated nh3hia/model3d/lte_probe.py machinery as analysis/mesh/chord_experiment.py --
no new measurement code, purely presentation.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nh3hia.model3d.lte_probe import run_lte, tau_from_brightness, tau_from_image, chord_metrics
from nh3hia import paths

OUT_DIR = paths.PRODUCTION + "/output_lut_gold/results/"
LOG_N, T, LOG_NDV = 6.0, 24.0, 15.0  # same reference point as analysis/mesh/chord_experiment.py

MESHES = [
    ('cube', None, 'tab:red', '-'),
    ('radial', {'kind': 'radial', 'n_shell': 14, 'base': 260, 'seed': 7}, 'tab:green', '-'),
    ('hybrid', {'kind': 'hybrid', 'core_radius_frac': 0.4, 'surface_frac': 0.85}, 'tab:blue', '-'),
]

# measured this session (mesh_economy_nlte_cost.py), same physical point,
# max_NLTE=250 -- hardcoded here rather than re-measured (that run already
# took ~35 min; this script is presentation only)
NLTE_COST = {'cube': (269, 91.5), 'hybrid': (814, 675.6), 'radial': (1313, 1175.8)}


def chord_law_figure():
    results = {}
    for name, mesh, color, ls in MESHES:
        run = run_lte(LOG_N, T, LOG_NDV, mesh=mesh, odir='/tmp/mesh_economy_fig/')
        b_bri, tau_bri, max_frac = tau_from_brightness(run)
        b_img, tau_img = tau_from_image(run)
        m_bri = chord_metrics(b_bri, tau_bri)
        m_img = chord_metrics(b_img, tau_img)
        results[name] = dict(npoints=run['npoints'], color=color, ls=ls,
                              b_bri=b_bri, tau_bri=tau_bri, m_bri=m_bri,
                              b_img=b_img, tau_img=tau_img, m_img=m_img)
        print(f"{name:8s} npoints={run['npoints']:5d}  "
              f"RMS(bri)={m_bri['rms']:6.1%}  monotonic={m_bri['monotonic']}  limb/law={m_bri['limb']:.3f}  "
              f"RMS(img)={m_img['rms']:6.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, key, title in [(axes[0], 'bri', r'$\tau(b)$ from brightness inversion'),
                            (axes[1], 'img', r"$\tau(b)$ from Magritte's own image")]:
        for name, mesh, color, ls in MESHES:
            r = results[name]
            b, tau = r[f'b_{key}'], r[f'tau_{key}']
            m = r[f'm_{key}']
            ax.scatter(b, tau, s=6, alpha=0.3, color=color,
                       label=f"{name} (n={r['npoints']}, RMS={m['rms']:.1%})")
            bb = np.linspace(0, 1, 200)
            ax.plot(bb, m['A'] * np.sqrt(np.clip(1 - bb ** 2, 0, None)), '--', color=color, lw=1.5)
        ax.set_xlabel(r'impact parameter $b/R$')
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel(r'$\tau$')
    plt.suptitle(rf'Optical depth vs.\ chord length, LTE ($T_k$={T:.0f}K, $\log n_{{\rm H_2}}$={LOG_N}, '
                 rf'$\log(N/\Delta v)$={LOG_NDV}): dashed = fitted $\tau_c\sqrt{{1-(b/R)^2}}$ chord law',
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out_png = os.path.join(OUT_DIR, 'mesh_economy_chord_experiment_hybrid_20260922.png')
    plt.savefig(out_png, dpi=200)
    print(f"\nsaved {out_png}")
    plt.close(fig)
    return results


def cost_figure():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    names = list(NLTE_COST.keys())
    colors = {'cube': 'tab:red', 'hybrid': 'tab:blue', 'radial': 'tab:green'}

    ax = axes[0]
    walls = [NLTE_COST[n][1] for n in names]
    bars = ax.bar(names, walls, color=[colors[n] for n in names], alpha=0.85)
    for b, n in zip(bars, names):
        pts, wall = NLTE_COST[n]
        ax.text(b.get_x() + b.get_width() / 2, wall, f'{wall:.0f}s\n({wall/NLTE_COST["cube"][1]:.2f}$\\times$)',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('wall time [s] to converge ($\\mathrm{max\\_NLTE}=250$)')
    ax.set_title('NLTE cost')
    ax.grid(alpha=0.25, axis='y')

    ax = axes[1]
    pts = [NLTE_COST[n][0] for n in names]
    ax.bar(names, pts, color=[colors[n] for n in names], alpha=0.85)
    for i, n in enumerate(names):
        ax.text(i, NLTE_COST[n][0], f"{NLTE_COST[n][0]}", ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('mesh points')
    ax.set_title('Point count')
    ax.grid(alpha=0.25, axis='y')

    plt.suptitle(rf'Mesh economy: NLTE cost at $T_k$={T:.0f}K, $\log n_{{\rm H_2}}$={LOG_N}, '
                 rf'$\log(N/\Delta v)$={LOG_NDV}', fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out_png = os.path.join(OUT_DIR, 'mesh_economy_nlte_cost_20260922.png')
    plt.savefig(out_png, dpi=200)
    print(f"saved {out_png}")
    plt.close(fig)


if __name__ == '__main__':
    chord_law_figure()
    cost_figure()
