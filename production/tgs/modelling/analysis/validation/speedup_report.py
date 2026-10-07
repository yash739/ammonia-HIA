"""Report for the solver-profile validation (analysis.validation.speedup_run).

For every variant directory under production/output_speedup_tests/: per-node relative
differences against (a) the existing gold rows (cube mesh, legacy solver) and (b) the
variant 'close' (exact, same code path), the pre-registered acceptance verdict, iterations,
convergence parity and speed-up. Writes report.md and figures next to the data.

Acceptance (pre-registered): |dR/R| <= 0.5% for R_01, R_10, R_12, R_21, R_22; A_MAIN <= 1%;
HIA_IS/HIA_OS <= 0.5%; R_21_11 <= 1% where the (2,1) line is detected in both rows, else
absolute difference <= 3e-3 (below that the line is the numerical ripple floor and is only
ever used as an upper limit); no node that gold converged may fail to converge (>= 90%).
Nodes gold did not converge are listed but excluded from pass/fail.

Usage: python3 -m analysis.validation.speedup_report
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nh3hia import paths
from analysis.validation import speedup_nodes as N
from analysis.validation.speedup_run import OUT

KEY = ['log_n_H2', 'T_cloud', 'log_N_dv']
RATIOS = ['R_01_MAIN', 'R_10_MAIN', 'R_12_MAIN', 'R_21_MAIN', 'R_22_MAIN', 'HIA_IS', 'HIA_OS']
TOL = {'ratio': 0.5, 'A_MAIN': 1.0, 'R_21_11': 1.0}      # per cent
R21_FLOOR_ABS = 3e-3


def load_variants():
    out = {}
    for d in sorted(glob.glob(os.path.join(OUT, 'v_*'))):
        f = os.path.join(d, 'results', 'lut_dv0.30.csv')
        if os.path.exists(f) and os.path.getsize(f) > 0:
            df = pd.read_csv(f)
            df = df[df.Status == 'SUCCESS'].copy()
            for k in KEY:
                df[k] = df[k].round(4)
            if len(df):
                out[os.path.basename(d)[2:]] = df
    return out


def load_gold():
    g = pd.read_csv(paths.LUT_GOLD_CSV)
    for k in KEY:
        g[k] = g[k].round(4)
    return g


def compare(var, ref, ref_unconv_excluded=True):
    """Row-wise relative differences var vs ref on the shared nodes."""
    m = var.merge(ref, on=KEY, suffixes=('', '_ref'))
    rows = []
    for _, r in m.iterrows():
        d = {k: r[k] for k in KEY}
        for k in RATIOS + ['A_MAIN']:
            d['d_' + k] = (r[k] / r[k + '_ref'] - 1.0) * 100.0 if r[k + '_ref'] else np.nan
        det = bool(r['detected_21']) and bool(r['detected_21_ref'])
        d['d_R_21_11'] = (r['R_21_11'] / r['R_21_11_ref'] - 1.0) * 100.0 if det else np.nan
        d['R21_abs'] = abs(r['R_21_11'] - r['R_21_11_ref'])
        d['R21_detected'] = det
        d['it'], d['it_ref'] = r['halting_iter'], r['halting_iter_ref']
        d['conv'], d['conv_ref'] = r['final_convergence'], r['final_convergence_ref']
        d['ok'], d['ok_ref'] = bool(r['convergence_ok']), bool(r['convergence_ok_ref'])
        d['wall'], d['wall_ref'] = r['wall_s'], r['wall_s_ref']
        rows.append(d)
    return pd.DataFrame(rows)


def verdict(c):
    """Per-node pass/fail list for nodes the reference converged."""
    c = c.copy()
    bad = []
    for _, r in c.iterrows():
        why = []
        if not r.ok_ref:
            bad.append(('excluded (reference unconverged)', r)); continue
        for k in RATIOS:
            if abs(r['d_' + k]) > TOL['ratio']:
                why.append(f"{k} {r['d_' + k]:+.2f}%")
        if abs(r['d_A_MAIN']) > TOL['A_MAIN']:
            why.append(f"A_MAIN {r['d_A_MAIN']:+.2f}%")
        if r.R21_detected:
            if abs(r['d_R_21_11']) > TOL['R_21_11']:
                why.append(f"R_21_11 {r['d_R_21_11']:+.2f}%")
        elif r.R21_abs > R21_FLOOR_ABS:
            why.append(f"R_21_11 abs {r.R21_abs:.1e}")
        if not r.ok:
            why.append(f"unconverged ({r.conv:.1f}%)")
        bad.append((' ; '.join(why), r))
    return bad


def summarize(name, c):
    v = verdict(c)
    tested = [r for w, r in v if not w.startswith('excluded')]
    fails = [(w, r) for w, r in v if w and not w.startswith('excluded')]
    cols = {}
    for k in RATIOS + ['A_MAIN', 'R_21_11']:
        x = c.loc[c.ok_ref, 'd_' + k].abs().dropna()
        cols[k] = (x.median() if len(x) else np.nan, x.max() if len(x) else np.nan)
    spd = (c.wall_ref / c.wall).replace([np.inf], np.nan)
    return dict(name=name, n=len(c), n_tested=len(tested), n_fail=len(fails), cols=cols,
                speed_med=spd.median(), speed_min=spd.min(), speed_max=spd.max(),
                wall_sum=c.wall.sum(), wall_ref_sum=c.wall_ref.sum(),
                it_med=c.it.median(), it_ref_med=c.it_ref.median(), fails=fails,
                n_excl=sum(1 for w, _ in v if w.startswith('excluded')))


def md_table(rows):
    hdr = '| variant | nodes | fail | speed-up med (min–max) | wall sum | median / max \\|Δ\\| % R_01 | R_10 | R_12 | R_21 | R_22 | A_MAIN | R_21_11 |'
    out = [hdr, '|' + '---|' * 12]
    for s in rows:
        f = lambda k: f"{s['cols'][k][0]:.3f} / {s['cols'][k][1]:.2f}"
        out.append(f"| {s['name']} | {s['n_tested']} (+{s['n_excl']} excl.) | {s['n_fail']} | "
                   f"{s['speed_med']:.1f}× ({s['speed_min']:.1f}–{s['speed_max']:.1f}) | {s['wall_sum']/3600:.2f} h | "
                   f"{f('R_01_MAIN')} | {f('R_10_MAIN')} | {f('R_12_MAIN')} | {f('R_21_MAIN')} | {f('R_22_MAIN')} | {f('A_MAIN')} | {f('R_21_11')} |")
    return '\n'.join(out)


def main():
    V = load_variants()
    G = load_gold()
    _, label = N.nodes()
    sections = []
    # (a) vs gold
    rows, comps = [], {}
    names = ['legacy'] + [s.replace('+', '-') for s in N.STAGES] + [s.replace('+', '-') for s in N.ABLATIONS
                                                                    if s.replace('+', '-') not in [x.replace('+', '-') for x in N.STAGES]]
    names += [s.replace('+', '-') for s in N.LEVEL_LADDER + N.PREC_LADDER]
    seen = set()
    for nm in names:
        if nm in V and nm not in seen:
            seen.add(nm)
            c = compare(V[nm], G); comps[nm] = c
            rows.append(summarize(nm, c))
    sections.append('## Variants vs gold (cube mesh, legacy solver)\n\n' + md_table(rows))
    # failures detail
    det = []
    for s in rows:
        for w, r in s['fails']:
            det.append(f"- `{s['name']}` at (log n {r.log_n_H2}, T {r.T_cloud}, log N/Δv {r.log_N_dv}) "
                       f"[{label.get((r.log_n_H2, r.T_cloud, r.log_N_dv), '?')}]: {w}")
    sections.append('## Acceptance failures (nodes gold converged)\n\n' + ('\n'.join(det) if det else 'none'))
    # (b) vs 'close' for attribution
    if 'close' in V:
        rows2 = []
        for nm, df in V.items():
            if nm in ('close', 'legacy'):
                continue
            c = compare(df, V['close']); c['ok_ref'] = True
            rows2.append(summarize(nm, c))
        sections.append("## Variants vs 'close' (exact baseline; isolates the effect of each added lever)\n\n" + md_table(rows2))
    # convergence parity
    par = []
    for nm, c in comps.items():
        lost = c[(c.ok_ref) & (~c.ok)]
        gained = c[(~c.ok_ref) & (c.ok)]
        par.append(f"| {nm} | {len(lost)} | {len(gained)} | {c.it.median():.0f} vs {c.it_ref.median():.0f} |")
    sections.append('## Convergence parity vs gold\n\n| variant | converged in gold, not here | unconverged in gold, converged here | median iterations (here vs gold) |\n|---|---|---|---|\n' + '\n'.join(par))
    open(os.path.join(OUT, 'report.md'), 'w').write('# Solver-profile validation\n\n' + '\n\n'.join(sections) + '\n')
    # figure: max error by stage per ratio, and speed-up vs gold wall
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for nm, c in comps.items():
        if nm == 'legacy':
            continue
        ax[1].scatter(c.wall_ref, c.wall_ref / c.wall, s=14, label=nm, alpha=0.7)
    ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel('gold wall time [s]'); ax[1].set_ylabel('speed-up vs gold')
    ax[1].legend(fontsize=6)
    xs = [s['name'] for s in rows]
    for k, mk in zip(RATIOS[:5], 'osd^v'):
        ax[0].plot(range(len(rows)), [s['cols'][k][1] for s in rows], marker=mk, label=k, ls='')
    ax[0].axhline(TOL['ratio'], color='k', ls='--', lw=0.8)
    ax[0].set_xticks(range(len(xs))); ax[0].set_xticklabels(xs, rotation=70, ha='right', fontsize=6)
    ax[0].set_ylabel('max |Δ| vs gold [%]'); ax[0].set_yscale('symlog', linthresh=0.01); ax[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'speedup_summary.png'), dpi=130)
    print(open(os.path.join(OUT, 'report.md')).read())


if __name__ == '__main__':
    main()
