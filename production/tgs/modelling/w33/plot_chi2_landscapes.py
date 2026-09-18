"""Chi^2 loss landscapes for the legacy-grid retrieval of Stutzki (1984)
observed ratio vectors -- to check the retrieved minimum is a real basin in a
smooth landscape, not a noise spike picked out of a rough one.

For each source, computes weighted_chi2 at EVERY row of the corrected legacy
catalogue (not a profiled/marginalized subset), plots chi2 vs log10(n_H2),
colored by T_cloud, with Stutzki's own Table 1a (adopted, high-density) and
Table 1b (rejected, low-density) solutions marked as vertical lines where
available. This is the plain scatter of every grid point actually evaluated,
so the shape of the minimum -- or its absence -- is visible directly.
"""
import os
import sys
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stutzki_tables_full as sf
from legacy_catalogue import load_legacy_catalogue
from scoring import weighted_chi2

ODIR = "/home/yasho379/magritte_rebuilt/scratch/output/output_chi2_landscapes/"
os.makedirs(ODIR, exist_ok=True)

KEYS = ('R_01_MAIN', 'R_10_MAIN', 'R_21_MAIN', 'R_12_MAIN')
ROWS = load_legacy_catalogue(min_convergence=90.0, tau_range=(0.02, 60))

T_COLORS = {18.0: 'tab:blue', 36.0: 'tab:orange', 54.0: 'tab:green'}


def norm_pos_multi(p):
    m = re.match(r'\((-?\d+),\s*(-?\d+)\)\s*(.*)', p)
    if m:
        base = f"{m.group(1)},{m.group(2)}"
        suffix = m.group(3).strip()
        return f"{base} {suffix}" if suffix else base
    return p.strip()


A1_NORM = {(region, norm_pos_multi(pos)): e for (region, pos), e in sf.TABLE_1A_FULL.items()}
B1_NORM = {(region, norm_pos_multi(pos)): e for (region, pos), e in sf.TABLE_1B_FULL.items()}


def to_key(name, offset):
    name0 = re.sub(r'\s*\([a-z]\)\s*$', '', name).strip()
    om = {'OMC 2': 'OMC2', 'OMC S1': 'S1', 'OMC S2': 'S2', 'OMC S3': 'S3', 'OMC S4': 'S4'}
    if name0 in om:
        return ('OMC', om[name0])
    if name0.startswith('S106'):
        return ('S106', norm_pos_multi(offset))
    if name0.startswith('S87'):
        tag = '21.0 km/s' if '21.0' in name0 else ('23.5 km/s' if '23.5' in name0 else None)
        return ('S87', f'0,0 {tag}') if tag else None
    if name0.startswith('W48'):
        base = norm_pos_multi(offset)
        return ('W48', f'{base} 45 km/s' if '45.0' in name0 else base)
    return None


def plot_source(csv_key, out_name, title):
    e = sf.TABLE_2_1984_FULL[csv_key]
    if None in (e['R_10_MAIN'], e['R_12_MAIN'], e['R_21_MAIN'], e['R_01_MAIN']):
        print(f"skip {csv_key}: non-detection in one ratio")
        return None
    ov = np.array([e['R_01_MAIN'], e['R_10_MAIN'], e['R_21_MAIN'], e['R_12_MAIN']])
    ev = np.array([e.get(f'{k}_err') or 0.05 * ov[i] for i, k in enumerate(KEYS)])

    log_n = np.array([np.log10(r['numberdensity']) for r in ROWS])
    T = np.array([r['T_cloud'] for r in ROWS])
    chi2_all = np.array([weighted_chi2(np.array([r[k] for k in KEYS]), ov, ev) for r in ROWS])

    fig, ax = plt.subplots(figsize=(7.5, 5))

    # Raw scatter (every vturb/radius combination at each grid density) shown
    # faint in the background, so the profile below is visibly an envelope of
    # real evaluated points, not a smoothed or cherry-picked curve.
    for Tval in sorted(set(T)):
        m = T == Tval
        ax.scatter(log_n[m], chi2_all[m], s=8, color=T_COLORS.get(Tval, 'gray'),
                   alpha=0.15, zorder=1)

    # Profile likelihood: at each DISTINCT grid density, the minimum chi2 over
    # every other nuisance parameter (vturb, radius/column) actually evaluated
    # at that density -- this is what shows the basin shape cleanly, since the
    # raw scatter above mixes many columns/linewidths at the same density.
    log_n_round = np.round(log_n, 6)
    for Tval in sorted(set(T)):
        m = T == Tval
        uniq = sorted(set(log_n_round[m]))
        prof = [chi2_all[m & (log_n_round == u)].min() for u in uniq]
        ax.plot(uniq, prof, 'o-', ms=5, lw=1.6, color=T_COLORS.get(Tval, 'gray'),
                label=f'T={Tval:.0f} K (profile min)', zorder=3)

    imin = np.argmin(chi2_all)
    ax.scatter([log_n[imin]], [chi2_all[imin]], marker='*', s=280, color='crimson',
               edgecolor='k', zorder=5,
               label=f'retrieved min (T={T[imin]:.0f} K, $\\chi^2$={chi2_all[imin]:.2f})')
    chi2 = chi2_all

    key1a = to_key(*csv_key)
    fit_a = A1_NORM.get(key1a)
    fit_b = B1_NORM.get(key1a)
    if fit_a is not None:
        ax.axvline(fit_a['log_nH2'], color='k', ls='--', lw=1.4,
                   label=f"Table 1a (adopted): log n={fit_a['log_nH2']:.2f}, T={fit_a['T_k']:.0f} K")
    if fit_b is not None:
        ax.axvline(fit_b['log_nH2'], color='k', ls=':', lw=1.4,
                   label=f"Table 1b (rejected): log n={fit_b['log_nH2']:.2f}, T={fit_b['T_k']:.0f} K")

    ax.set_xlabel(r'$\log_{10}(n_{H_2}$ / cm$^{-3})$')
    ax.set_ylabel(r'weighted $\chi^2$ (4-ratio, legacy grid)')
    ax.set_title(title)
    ax.set_yscale('log')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(ODIR, out_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}  (n_rows={len(ROWS)}, min chi2={chi2.min():.3f} at log n={log_n[imin]:.2f}, T={T[imin]:.0f})")
    return path


if __name__ == '__main__':
    targets = [
        (('OMC S3 (d)', '(0, 0)'), 'chi2_landscape_OMC_S3.png', 'OMC S3 -- good fit, closely tracks Table 1b'),
        (('OMC S4 (d)', '(0, 0)'), 'chi2_landscape_OMC_S4.png', 'OMC S4 -- good fit, closely tracks Table 1b'),
        (('S106', '(0, 0)'), 'chi2_landscape_S106_00.png', 'S106 (0,0) -- good fit, closely tracks Table 1b'),
        (('OMC S1 (d)', '(0, 0)'), 'chi2_landscape_OMC_S1.png', 'OMC S1 -- POOR fit (chi2=37.7): no clean minimum'),
        (('S106', '(-40, 0)'), 'chi2_landscape_S106_-40_0.png', 'S106 (-40,0) -- POOR fit (chi2=65.7): no clean minimum'),
        (('W33', '(0, 0)'), 'chi2_landscape_W33_1984.png', 'W33 (1984 Stutzki survey position) -- POOR fit (chi2=99.0)'),
    ]
    for key, name, title in targets:
        plot_source(key, name, title)
