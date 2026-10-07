"""Validation nodes and variant lists for the solver-profile speed-up tests.

All nodes are on the gold grid (nh3hia.lut.axes), so the existing gold rows
(cube mesh, legacy solver) are the reference and nothing is re-run for it except a
determinism check. Strata, chosen where each speed-up lever is most exposed:
corners; the 18/24/30/36 K slices; masing; cold low-density (iteration-bound; includes
nodes gold did not converge); hot dense (level truncation weakest: T >= 42 K, log n >= 7.5);
the five retrieval nodes; thin/easy; then a seeded fill to 48 spread over the grid.
"""
import random

from nh3hia.lut.axes import LOG_N_AXIS, LOG_N_EXT_AXIS, T_AXIS, LOG_NDV_AXIS, LOG_NDV_EXT_AXIS

LOG_N = LOG_N_AXIS + LOG_N_EXT_AXIS
LOG_NDV = LOG_NDV_AXIS + LOG_NDV_EXT_AXIS

STRATA = {
    'corner': [(3.5, 9.0, 14.0), (3.5, 9.0, 16.25), (3.5, 48.0, 14.0), (3.5, 48.0, 16.25),
               (8.5, 9.0, 14.0), (8.5, 9.0, 16.25), (8.5, 48.0, 14.0), (8.5, 48.0, 16.25)],
    'slice': [(5.5, 18.0, 15.35), (5.5, 24.0, 15.35), (5.5, 30.0, 15.35), (5.5, 36.0, 15.35)],
    'masing': [(5.0, 36.0, 15.8), (5.25, 30.0, 15.8), (4.75, 33.0, 15.575), (5.5, 39.0, 16.025)],
    'cold_lowdens': [(3.5, 12.0, 15.35), (4.0, 15.0, 15.8), (4.25, 12.0, 16.025), (3.75, 18.0, 15.575),
                     (4.5, 18.0, 15.8)],
    'hot_dense': [(7.5, 42.0, 15.8), (8.0, 45.0, 16.025), (8.5, 48.0, 16.25), (8.0, 48.0, 14.45),
                  (7.5, 45.0, 15.35), (8.5, 42.0, 14.9)],
    'retrieval': [(7.5, 36.0, 15.35), (7.5, 39.0, 15.35), (7.0, 27.0, 15.575), (8.0, 36.0, 15.8),
                  (8.0, 33.0, 16.025)],
    'thin': [(7.0, 24.0, 14.0), (6.0, 21.0, 14.225), (5.0, 27.0, 14.0), (7.5, 30.0, 14.45)],
}


def nodes(n_total=48, seed=11):
    out, label = [], {}
    for name, pts in STRATA.items():
        for p in pts:
            if p not in label:
                out.append(p)
                label[p] = name
    rng = random.Random(seed)
    grid = [(a, b, c) for a in LOG_N for b in T_AXIS for c in LOG_NDV]
    pool = [p for p in grid if p not in label]
    rng.shuffle(pool)
    # greedy spread: take the pool point farthest (in index space) from everything chosen so far
    ix = lambda p: (LOG_N.index(p[0]) / (len(LOG_N) - 1), T_AXIS.index(p[1]) / (len(T_AXIS) - 1),
                    LOG_NDV.index(p[2]) / (len(LOG_NDV) - 1))
    while len(out) < n_total:
        best = max(pool[:400], key=lambda p: min(sum((u - v) ** 2 for u, v in zip(ix(p), ix(q))) for q in out))
        out.append(best); label[best] = 'fill'; pool.remove(best)
    return out, label


# Stage = cumulative profile; every stage is compared with the gold row at each node.
STAGES = ['close',
          'close+lev36',
          'close+lev36+nq12',
          'close+lev36+nq12+prec4',
          'close+lev36+nq12+prec4+ngfixed+ngmem15']

# One-lever-at-a-time ablations relative to 'close' (which is exact), on a 16-node subset.
ABLATIONS = ['close+lev36', 'close+nq10', 'close+nq12', 'close+nq14', 'close+nq16',
             'close+prec4', 'close+prec5', 'close+ngfixed+ngmem15',
             'close+dist5', 'close+dist20', 'close+minrel8', 'close+interp2.0']
# Level-truncation ladder on the hot dense nodes (where it is weakest) and masing nodes.
LEVEL_LADDER = ['close+lev32', 'close+lev36', 'close+lev40', 'close']
# Convergence ladder against a 1e-8 "truth" (max_NLTE raised).
PREC_LADDER = ['close+prec4', 'close+prec5', 'close', 'close+prec8']


def subset(n, seed=5):
    pts, label = nodes()
    # keep stratum diversity: one-two from each stratum then fill
    out = []
    for name in list(STRATA) + ['fill']:
        out += [p for p in pts if label[p] == name][:2]
    out = out[:n]
    return out


if __name__ == '__main__':
    pts, lab = nodes()
    for p in pts:
        print(p, lab[p])
    print(len(pts), 'nodes;', len(subset(16)), 'in the ablation subset')
