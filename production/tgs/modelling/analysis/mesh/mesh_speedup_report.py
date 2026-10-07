"""Step 2 report: cube vs radial 12x140 vs radial 14x260 (reference) with the fast solver profile.

Per-ratio relative differences on the common nodes (each mesh vs the radial 14x260 reference and
vs the gold row), by validation stratum, plus measured wall costs and the radial cost model used
to project the full-grid rebuild. Writes production/output_speedup_tests/mesh_report.md.

Usage: python3 -m analysis.mesh.mesh_speedup_report
"""
import os

import numpy as np
import pandas as pd

from nh3hia import paths
from analysis.validation import speedup_nodes as N
from analysis.validation.speedup_run import OUT

KEY = ['log_n_H2', 'T_cloud', 'log_N_dv']
RATIOS = ['R_01_MAIN', 'R_10_MAIN', 'R_12_MAIN', 'R_21_MAIN', 'R_22_MAIN', 'R_21_11', 'HIA_IS', 'HIA_OS']


def load(name):
    f = os.path.join(OUT, name, 'results', 'lut_dv0.30.csv')
    if not (os.path.exists(f) and os.path.getsize(f)):
        return None
    d = pd.read_csv(f)
    d = d[d.Status == 'SUCCESS'].copy()
    for k in KEY:
        d[k] = d[k].round(4)
    return d


def rel(a, b, cols=RATIOS):
    m = a.merge(b, on=KEY, suffixes=('', '_b'))
    out = m[KEY].copy()
    for c in cols:
        x = (m[c] / m[c + '_b'] - 1.0) * 100.0
        # relative differences are meaningless where a ratio is at the numerical floor (cold, thin nodes):
        # require the reference value to be measurable (satellite/(2,2) ratios > 0.02; (2,1) detected in both)
        if c == 'R_21_11':
            x = x.where(m['detected_21'].astype(bool) & m['detected_21_b'].astype(bool))
        elif c.startswith('R_'):
            x = x.where(m[c + '_b'].abs() > 0.02)
        out[c] = x
    out['conv_a'], out['conv_b'] = m.final_convergence, m.final_convergence_b
    return out


def table(name, df, cols=RATIOS):
    lines = [f'| {name} | nodes | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 2)]
    lines.append(f"| median / max \\|Δ\\| % | {len(df)} | " + ' | '.join(
        f"{df[c].abs().median():.2f} / {df[c].abs().max():.1f} (n={df[c].notna().sum()})" for c in cols) + ' |')
    return '\n'.join(lines)


def main():
    gold = pd.read_csv(paths.LUT_GOLD_CSV)
    for k in KEY:
        gold[k] = gold[k].round(4)
    cube, r12, r14 = load('mesh_cube'), load('mesh_radial12x140'), load('mesh_radial14x260')
    _, label = N.nodes()
    sec = ['# Mesh comparison with the fast solver profile (regional: lev36 for log N/dv <= 15.8)\n']
    if r14 is not None:
        sec.append(f'Reference: radial 14x260 ({len(r14)} nodes, cheaper nodes only). Differences in per cent.\n')
        for nm, d in (('cube', cube), ('radial 12x140', r12)):
            sec.append(table(f'{nm} vs radial 14x260', rel(d, r14)))
        sec.append('\n' + table('gold (cube, legacy) vs radial 14x260', rel(gold, r14)))
    sec.append('\n## Cube vs radial 12x140 over all 24 mesh nodes\n')
    sec.append(table('radial 12x140 vs cube', rel(r12, cube)))
    d = rel(r12, cube)
    d['stratum'] = [label.get((a, b, c), '?') for a, b, c in zip(d.log_n_H2, d.T_cloud, d.log_N_dv)]
    sec.append('\nBy stratum (median |Δ| % of R_01 / R_10 / R_21_11):\n')
    for s, g in d.groupby('stratum'):
        sec.append(f"- {s}: {len(g)} nodes, {g.R_01_MAIN.abs().median():.1f} / {g.R_10_MAIN.abs().median():.1f} / {g.R_21_11.abs().median():.1f}")
    # costs
    sec.append('\n## Measured cost (fast profile, 3 processes x 4 threads)\n')
    rows = []
    for nm, df in (('cube', cube), ('radial 12x140', r12), ('radial 14x260', r14)):
        if df is None:
            continue
        m = df.merge(gold[KEY + ['wall_s']], on=KEY, suffixes=('', '_gold'))
        rows.append(f"| {nm} | {len(df)} | {df.npoints.median():.0f} | {df.halting_iter.median():.0f} | {df.wall_s.sum():.0f} | {m.wall_s_gold.sum():.0f} | {m.wall_s_gold.sum() / df.wall_s.sum():.1f} |")
    sec.append('| mesh | nodes | points | median iterations | wall sum [s] | gold wall sum [s] | speed-up vs gold |\n|---|---|---|---|---|---|---|\n' + '\n'.join(rows))
    m = r12.merge(gold[KEY + ['wall_s']], on=KEY, suffixes=('', '_gold'))
    a, b = np.polyfit(np.log(m.wall_s_gold), np.log(m.wall_s), 1)
    pred = np.exp(a * np.log(gold.wall_s) + b)
    ratio = pred.sum()
    sec.append(f"\nRadial 12x140 cost model (log-log fit to gold wall time, slope {a:.2f}): full 2464-node table = "
               f"{ratio / 3600:.0f} core-hours of jobs -> {ratio / 3 / 3600:.1f} h on 3 processes, "
               f"{ratio / 4 / 3600:.1f} h on 4; ratio-of-sums estimate {gold.wall_s.sum() * m.wall_s.sum() / m.wall_s_gold.sum() / 3 / 3600:.1f} h on 3.")
    open(os.path.join(OUT, 'mesh_report.md'), 'w').write('\n'.join(sec) + '\n')
    print('\n'.join(sec))


if __name__ == '__main__':
    main()
