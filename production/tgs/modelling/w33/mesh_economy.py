"""Paper I, Work item -1: mesh economy investigation.

Step 1: region-split RMS on the existing cube mesh, with a bin edge added
at b/R=0.4 -- quantifies whether the tau(b) chord-law failure really
concentrates in the core (b/R<0.4, the region build_point_cloud's own
docstring says the cube mesh structurally cannot populate) as opposed to
being spread evenly across the disc.

Step 2/3: LTE chord-law validation of the new hybrid mesh
(mesh={'kind':'hybrid', 'core_radius_frac':X}, implemented in
nh3_NLTE_sphere.py this session) against the cube and radial baselines,
at 1-2 core_radius_frac choices.

Both steps are LTE-only (max_NLTE=0), costing seconds per model.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lte_probe import run_lte, tau_from_brightness, tau_from_image, chord_metrics

OUT_DIR = "/home/yasho379/magritte_rebuilt/production/output_lut_gold/results/"
LOG_N, T, LOG_NDV = 6.0, 24.0, 15.0  # same reference point as chord_experiment.py

CUBE_RADIAL_BINS = np.array([0.0, 0.15, 0.30, 0.40, 0.45, 0.60, 0.75, 0.90, 1.0])


def region_split_rms(b, tau, A, split=0.4):
    """RMS fractional deviation from the fitted chord law, computed
    separately for b<split and b>=split (b<0.9 only, matching chord_metrics'
    own core definition)."""
    shape = np.sqrt(np.clip(1.0 - b ** 2, 0.0, None))
    law = A * shape
    inside = (b < 0.9) & np.isfinite(tau) & (law > 0)
    rel = tau[inside] / law[inside]
    core = b[inside] < split
    outer = ~core
    rms_core = float(np.sqrt(np.mean((rel[core] - 1.0) ** 2))) if core.sum() > 0 else np.nan
    rms_outer = float(np.sqrt(np.mean((rel[outer] - 1.0) ** 2))) if outer.sum() > 0 else np.nan
    return rms_core, rms_outer, int(core.sum()), int(outer.sum())


def step1_region_split():
    print("=" * 70)
    print("STEP 1: region-split RMS (cube mesh), bin edge at b/R=0.4")
    print("=" * 70)
    run = run_lte(LOG_N, T, LOG_NDV, mesh=None, odir='/tmp/mesh_economy/')
    b, tau, max_frac = tau_from_brightness(run)
    m = chord_metrics(b, tau, bins=CUBE_RADIAL_BINS)
    rms_core, rms_outer, n_core, n_outer = region_split_rms(b, tau, m['A'], split=0.4)
    print(f"cube mesh: npoints={run['npoints']}  overall RMS={m['rms']:.1%}  monotonic={m['monotonic']}")
    print(f"  b<0.4 (core):  RMS={rms_core:.1%}  n={n_core}")
    print(f"  b>=0.4 (outer): RMS={rms_outer:.1%}  n={n_outer}")
    if np.isfinite(rms_core) and np.isfinite(rms_outer) and rms_outer > 0:
        print(f"  core/outer RMS ratio: {rms_core/rms_outer:.2f}x")
    return dict(rms_core=rms_core, rms_outer=rms_outer, n_core=n_core, n_outer=n_outer,
                overall_rms=m['rms'], npoints=run['npoints'])


MESH_VARIANTS = [
    ('cube', None),
    ('radial', {'kind': 'radial', 'n_shell': 14, 'base': 260, 'seed': 7}),
    ('hybrid_core0.4_only', {'kind': 'hybrid', 'core_radius_frac': 0.4}),
    ('hybrid_core0.4_surf0.85', {'kind': 'hybrid', 'core_radius_frac': 0.4, 'surface_frac': 0.85}),
    ('hybrid_core0.4_surf0.80', {'kind': 'hybrid', 'core_radius_frac': 0.4, 'surface_frac': 0.80}),
]


def step2_hybrid_validation():
    """First tried a core-only hybrid (fills the empty 0.01-0.40R zone build_
    point_cloud's own docstring identifies), which fixed monotonicity but only
    brought RMS to ~17% -- step 1's region-split result explains why: the
    outer/limb region (b>=0.4) has WORSE RMS (42%) than the core (30%), so a
    core-only fix addresses the smaller half of the problem. Adding a second
    radial-shell band near the surface (surface_frac) closes the rest of the
    gap -- see the two hybrid_core..._surf... entries below."""
    print()
    print("=" * 70)
    print("STEP 2/3: hybrid mesh LTE chord-law validation")
    print("=" * 70)
    results = {}
    colors_cycle = iter(['tab:red', 'tab:green', 'tab:purple', 'tab:blue', 'tab:orange'])
    colors = {}
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, mesh in MESH_VARIANTS:
        run = run_lte(LOG_N, T, LOG_NDV, mesh=mesh, odir='/tmp/mesh_economy/')
        b, tau, _ = tau_from_brightness(run)
        m = chord_metrics(b, tau)
        results[name] = dict(npoints=run['npoints'], rms=m['rms'], monotonic=m['monotonic'], limb=m['limb'])
        print(f"{name:26s} npoints={run['npoints']:5d}  RMS={m['rms']:6.1%}  "
              f"monotonic={str(m['monotonic']):5s}  limb/law={m['limb']:.3f}")
        c = next(colors_cycle)
        colors[name] = c
        ax.scatter(b, tau, s=6, alpha=0.3, color=c,
                   label=f"{name} (n={run['npoints']}, RMS={m['rms']:.1%})")
        bb = np.linspace(0, 1, 200)
        ax.plot(bb, m['A'] * np.sqrt(np.clip(1 - bb ** 2, 0, None)), '--', color=c, lw=1.3)

    ax.set_xlabel(r'impact parameter $b/R$')
    ax.set_ylabel(r'$\tau$')
    ax.set_title(rf'Mesh economy: chord-law validation ($T_k$={T:.0f}K, $\log n_{{\rm H_2}}$={LOG_N}, '
                 rf'$\log(N/\Delta v)$={LOG_NDV})', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'mesh_economy_hybrid_validation.png')
    plt.savefig(out_png, dpi=160)
    print(f"\nsaved {out_png}")
    return results


if __name__ == '__main__':
    region_split = step1_region_split()
    hybrid_results = step2_hybrid_validation()
