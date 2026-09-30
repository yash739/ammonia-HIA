"""Overlay our T=18 K slice onto Stutzki & Winnewisser (1985) Figs. 5 and 6,
drawn directly on the scanned page (production/references/stutzki_1985.pdf,
page 8 = journal p. 20) so nothing is lost to hand-digitisation.

Axis calibration was measured from the frame lines and tick marks of a 200 dpi
render of that page (pixel coordinates below). The paper's x-axis
tau(Delta F1=0) is the CENTRAL-CHORD optical depth: his own Eq. (11) curve on
the figure matches stutzki_physics.eq11_thermal_ratio (radial tau) only when
the plotted tau is halved (best-fit scale 2.1, rms 0.037). Our tau11_chord is
therefore plotted directly on his axis.
"""
import argparse
import os
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from nh3hia import paths

PDF = paths.PRODUCTION + '/references/stutzki_1985.pdf'
DPI = 200

# page-pixel calibration at 200 dpi: x = x_tau1 + ppd*log10(tau); y = y0 - ppu*value
FIG5_UPPER = dict(x_tau1=388.5, ppd=187.0, y0=523.5 + 66, ppu=317.5,
                  box=(150, 120 + 66, 790, 545 + 66))
FIG5_LOWER = dict(x_tau1=388.5, ppd=187.0, y0=960.5 + 66, ppu=381.5,
                  box=(150, 560 + 66, 790, 1000 + 66))
FIG6_T18 = dict(x_tau1=800 + 309.25, ppd=179.75, y0=565.7, ppu=361.5,
                box=(800 + 60, 150, 800 + 700, 600))
COLORS = {3.5: 'tab:red', 5.0: 'tab:blue', 7.0: 'tab:green'}


def page_image():
    out = '/tmp/stutzki1985_p8'
    if not os.path.exists(out + '-08.png'):
        subprocess.run(['pdftoppm', '-f', '8', '-l', '8', '-r', str(DPI), '-png', PDF, out], check=True)
    return np.array(Image.open(out + '-08.png').convert('RGB'))


def to_px(cal, tau, val):
    return cal['x_tau1'] + cal['ppd'] * np.log10(tau), cal['y0'] - cal['ppu'] * val


def panel(ax, page, cal, df, ycol_fn, title, est):
    x0, y0, x1, y1 = cal['box']
    ax.imshow(page[y0:y1, x0:x1], extent=(x0, x1, y1, y0))
    for (mesh, mode, log_n), g in df.groupby(['mesh', 'mode', 'log_n_H2']):
        g = g.sort_values('tau11_chord')
        g = g[g.tau11_chord > 0]
        if g.empty:
            continue
        px, py = to_px(cal, g.tau11_chord.values, ycol_fn(g, est))
        style = dict(color=COLORS.get(log_n, 'k'), lw=2.2, alpha=0.85)
        if mesh == 'cube':
            style.update(ls='--', lw=1.6)
        elif mesh == 'cube_taucorr':
            style.update(ls=':', lw=2.4)
        marker = 'o' if mode == 'nlte' else 'x'
        conv = g.final_convergence.fillna(100).values >= 90
        ax.plot(px, py, **style, label=f"{mesh} {mode} n=10^{log_n:g}")
        ax.scatter(px[conv], py[conv], marker=marker, s=22, color=style['color'], zorder=5)
        ax.scatter(px[~conv], py[~conv], marker=marker, s=40, facecolors='none',
                   edgecolors='k', zorder=6)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', nargs='+',
                    default=[paths.SCRATCH + '/output/fig56_T18/slice_nlte.csv'])
    ap.add_argument('--meshes', default='radial,cube')
    ap.add_argument('--est', choices=('fit', 'pk'), default='fit')
    ap.add_argument('--out', default=paths.SCRATCH + '/plots/fig56_T18_overlay.png')
    a = ap.parse_args()

    df = pd.concat([pd.read_csv(c) for c in a.csv if os.path.exists(c)])
    df = df[(df.Status == 'SUCCESS') & df.mesh.isin(a.meshes.split(','))]
    page = page_image()
    r01 = lambda g, e: g[f'disc_{e}_R01'].values
    r10 = lambda g, e: g[f'disc_{e}_R10'].values
    rin = lambda g, e: 0.5 * (g[f'disc_{e}_R12'].values + g[f'disc_{e}_R21'].values)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7.5))
    panel(axes[0], page, FIG5_UPPER, df, r01, 'Fig. 5 upper: F1=0->1 / main', a.est)
    panel(axes[1], page, FIG5_LOWER, df, r10, 'Fig. 5 lower: F1=1->0 / main', a.est)
    panel(axes[2], page, FIG6_T18, df, rin, 'Fig. 6, T=18 K: inner average / main', a.est)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=6, fontsize=9)
    fig.suptitle("Stutzki & Winnewisser (1985) scanned (black) vs this work at T_k=18 K (colour: red 10^3.5, blue 10^5, green 10^7 cm^-3; "
                 "solid=radial mesh, dashed=cube; o=NLTE, x=LTE; hollow=convergence<90%)\n"
                 f"Ratios from the disc-averaged spectrum ({a.est} estimator), x = central-chord tau(1,1). "
                 "His T=18 curves -- Fig. 5: thick solid 10^3.5, dotted 10^5, dashed 10^7; Fig. 6: dashed 10^3.5, dotted 10^5, solid 10^7",
                 fontsize=10)
    plt.tight_layout(rect=(0, 0.07, 1, 0.95))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    plt.savefig(a.out, dpi=130)
    print('saved', a.out)


if __name__ == '__main__':
    main()
