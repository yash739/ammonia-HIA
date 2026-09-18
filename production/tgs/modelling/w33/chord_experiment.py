"""Optical-depth-vs-chord-length experiment: cube vs radial seed mesh,
reusing the validated lte_probe.py machinery (tau(b) from brightness AND
from Magritte's own optical-depth image, both against the exact analytic
chord law tau(b)=tau_centre*sqrt(1-(b/R)^2), which is exact at LTE for a
uniform sphere -- any departure is mesh/imaging, not physics).

Reproduces the same comparison behind the 09-16 notes' Sec 2.1/2.3
findings (non-monotonic cube tau(b), ~half central value, 1.6-2.4x too
high mid-disc; radial mesh monotonic and RMS-converging with refinement),
but as a saved, reproducible figure with the underlying numbers on record
-- no such figure existed before.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lte_probe import run_lte, tau_from_brightness, tau_from_image, chord_metrics

OUT_PNG = "/home/yasho379/magritte_rebuilt/production/output_lut_gold/results/chord_experiment_cube_vs_radial.png"
RADIAL_MESH = {'kind': 'radial', 'n_shell': 14, 'base': 260, 'seed': 7}

# moderate optical depth so tau is recoverable from brightness everywhere
# on the disc (not saturated) but the chord shape is clearly resolved.
LOG_N, T, LOG_NDV = 6.0, 24.0, 15.0


def main():
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    results = {}
    for name, mesh in [('cube', None), ('radial', RADIAL_MESH)]:
        run = run_lte(LOG_N, T, LOG_NDV, mesh=mesh, odir='/tmp/chord_experiment/')
        b_bri, tau_bri, max_frac = tau_from_brightness(run)
        b_img, tau_img = tau_from_image(run)
        m_bri = chord_metrics(b_bri, tau_bri)
        m_img = chord_metrics(b_img, tau_img)
        results[name] = dict(npoints=run['npoints'], tau_main_reported=run['tau_main_reported'],
                              max_frac=max_frac, b_bri=b_bri, tau_bri=tau_bri,
                              b_img=b_img, tau_img=tau_img, m_bri=m_bri, m_img=m_img)
        print(f"=== {name} mesh: npoints={run['npoints']} tau_main_reported={run['tau_main_reported']:.4f} ===")
        print(f"  from brightness: tau_centre(fit)={m_bri['A']:.4f}  RMS={m_bri['rms']:.1%}  "
              f"monotonic={m_bri['monotonic']}  limb/law={m_bri['limb']:.3f}  max T/T0={max_frac:.3f}")
        print(f"  from image:      tau_centre(fit)={m_img['A']:.4f}  RMS={m_img['rms']:.1%}  "
              f"monotonic={m_img['monotonic']}  limb/law={m_img['limb']:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, key, title in [(axes[0], 'bri', r'$\tau(b)$ from brightness inversion'),
                            (axes[1], 'img', r"$\tau(b)$ from Magritte's own image")]:
        for name, color in [('cube', 'tab:red'), ('radial', 'tab:green')]:
            r = results[name]
            b, tau = r[f'b_{key}'], r[f'tau_{key}']
            m = r[f'm_{key}']
            ax.scatter(b, tau, s=6, alpha=0.35, color=color, label=f'{name} (RMS={m["rms"]:.1%})')
            bb = np.linspace(0, 1, 200)
            ax.plot(bb, m['A'] * np.sqrt(np.clip(1 - bb ** 2, 0, None)), '--', color=color, lw=1.5)
        ax.set_xlabel(r'impact parameter $b/R$')
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel(r'$\tau$')
    plt.suptitle(rf'Optical depth vs. chord length, LTE ($T_k$={T:.0f}K, $\log n_{{\rm H_2}}$={LOG_N}, '
                 rf'$\log(N/\Delta v)$={LOG_NDV}): dashed = fitted $\tau_c\sqrt{{1-(b/R)^2}}$ chord law',
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(OUT_PNG, dpi=160)
    print(f"\nsaved {OUT_PNG}")

    c, r = results['cube'], results['radial']
    print(f"\nsummary: cube tau_centre={c['m_bri']['A']:.3f}  radial tau_centre={r['m_bri']['A']:.3f}  "
          f"ratio={c['m_bri']['A']/r['m_bri']['A']:.2f}")
    print(f"cube monotonic={c['m_bri']['monotonic']}  radial monotonic={r['m_bri']['monotonic']}")
    print(f"cube limb/law={c['m_bri']['limb']:.3f}  radial limb/law={r['m_bri']['limb']:.3f}")


if __name__ == '__main__':
    main()
