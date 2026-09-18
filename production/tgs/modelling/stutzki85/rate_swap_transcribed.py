"""Full, generalized version of rate_swap_full.py's hybrid collision matrix,
built from the complete user-supplied transcriptions of Green (1980) Table
III and Stutzki & Winnewisser (1985a) Table 1 -- production/references/
Green_Table3_1980.csv and Stutzki_1985a_table1.csv -- rather than the
earlier partial, hand-transcribed (1,1)-source-only subset in
rate_swap_full.py.

The model tracks exactly 18 (J,K,F) states x 2 parities = 36 levels:
(1,1):F=0,1,2  (2,1):F=1,2,3  (2,2):F=1,2,3  (3,1):F=2,3,4  (3,2):F=2,3,4
(4,4):F=3,4,5
-- which is exactly the scope both transcribed tables cover. Checked
directly (not assumed): both tables tabulate all 15 possible unordered
inter-manifold pairs among these six manifolds, and BOTH tables use
IDENTICAL source/target direction for every one of the 15 pairs (verified
by comparing the set of (source_manifold, target_manifold) pairs found in
each CSV -- they match exactly), which is what Green's own text implies
("Rate constants... for excitation... de-excitation rates can be obtained
from detailed balance") -- the tabulated direction is always the
excitation (energy-increasing) one. This removes any ambiguity about
which direction to apply the detailed-balance formula in.

Green gives all four (source parity, target parity) combinations directly
for each pair. Stutzki's Table 1 gives the IOS relative branching factor
g(source F, source parity -> target F', 'U') for every source sublevel
except 443U and 444U -- there is nothing higher than (4,4) in this scheme
for those two to excite into, so their absence is expected, not a gap.
The 'L'-target factor is obtained, per the paper's own Eq. (22b)/(25),
from the OPPOSITE source parity's 'U'-target entry for the same (F, F')
pair -- exactly the symmetry the original (1,1)-only code already used.

Table 2's 18 intra-multiplet quasi-elastic rates (rate_swap_test.py) are
reused unchanged -- they were already complete for all six manifolds.

Any (source, target) sublevel pair genuinely absent from these two tables
(443U/444U as inter-multiplet sources) falls back element-wise to
Loreau's own collision_matrix(T) entry -- the matrix is seeded from that
baseline and only overwritten where transcribed data exists.
`coverage_report()` counts exactly how many of the model's off-diagonal
entries (among the six tracked manifolds) ended up Stutzki-sourced vs
Loreau-fallback, so the completeness claim is measured, not assumed.
"""
import os

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import rate_swap_test as rst  # TABLE2_NH3_HE, ALPHA_HE_TO_H2, _level_index

_REF_DIR = "/home/yasho379/magritte_rebuilt/production/references"
GREEN_CSV = os.path.join(_REF_DIR, "Green_Table3_1980.csv")
STUTZKI_T1_CSV = os.path.join(_REF_DIR, "Stutzki_1985a_table1.csv")

GREEN_T_GRID = np.array([15.0, 30.0, 100.0, 300.0])
STUTZKI_T_GRID = np.array([15.0, 25.0, 50.0, 100.0])

MANIFOLDS = [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (4, 4)]
F_OF = {(1, 1): (0, 1, 2), (2, 1): (1, 2, 3), (2, 2): (1, 2, 3),
        (3, 1): (2, 3, 4), (3, 2): (2, 3, 4), (4, 4): (3, 4, 5)}

# Canonical (source manifold -> target manifold) direction for each of the
# 15 pairs, exactly as both tables tabulate it (verified identical in both,
# not assumed) -- always the excitation (energy-increasing) direction.
PAIR_DIRECTIONS = [
    ((1, 1), (2, 1)), ((1, 1), (2, 2)), ((1, 1), (3, 1)), ((1, 1), (3, 2)), ((1, 1), (4, 4)),
    ((2, 1), (3, 1)), ((2, 1), (3, 2)), ((2, 1), (4, 4)),
    ((2, 2), (2, 1)), ((2, 2), (3, 1)), ((2, 2), (3, 2)), ((2, 2), (4, 4)),
    ((3, 1), (4, 4)),
    ((3, 2), (3, 1)), ((3, 2), (4, 4)),
]


def _parse_green_label(s):
    s = s.replace(' ', '')
    J, rest = s.split(',')
    return int(J), int(rest[:-1]), rest[-1]  # (J, K, 'l'/'u')


def _parse_stutzki_label(s):
    s = s.strip()
    return int(s[0]), int(s[1]), int(s[2]), s[3]  # (J, K, F, 'L'/'U')


def load_green_table3(path=GREEN_CSV):
    """dict[(Jf,Kf,pf)][(Jt,Kt,pt)] -> np.array of 4 rates (cm^3/s), NH3-He,
    at T = [15,30,100,300] K. Only entries where BOTH manifolds are in
    MANIFOLDS are kept (Green's CSV also has some (4,2)/(4,1) entries this
    model doesn't track)."""
    df = pd.read_csv(path)
    out = {}
    mset = set(MANIFOLDS)
    for _, row in df.iterrows():
        Jf, Kf, pf = _parse_green_label(row['Initial'])
        Jt, Kt, pt = _parse_green_label(row['Final'])
        if (Jf, Kf) not in mset or (Jt, Kt) not in mset:
            continue
        rates = np.array([row['15K'], row['30K'], row['100K'], row['300K']], dtype=float)
        out.setdefault((Jf, Kf, pf), {})[(Jt, Kt, pt)] = rates
    return out


def _parse_ios_val(v):
    if isinstance(v, str) and v.strip().startswith('*'):
        return 0.0
    return float(v)


def load_stutzki_table1(path=STUTZKI_T1_CSV):
    """dict[(Jf,Kf,Ff,pf)][(Jt,Kt,Ft,pt)] -> np.array of 4 IOS relative
    factors at T = [15,25,50,100] K. pt is always 'U' as tabulated."""
    df = pd.read_csv(path)
    df = df[df['Initial'] != 'Initial']  # drop the duplicated header row
    out = {}
    mset = set(MANIFOLDS)
    for _, row in df.iterrows():
        Jf, Kf, Ff, pf = _parse_stutzki_label(row['Initial'])
        Jt, Kt, Ft, pt = _parse_stutzki_label(row['Final'])
        if (Jf, Kf) not in mset or (Jt, Kt) not in mset:
            continue
        vals = np.array([_parse_ios_val(row['15K']), _parse_ios_val(row['25K']),
                          _parse_ios_val(row['50K']), _parse_ios_val(row['100K'])])
        out.setdefault((Jf, Kf, Ff, pf), {})[(Jt, Kt, Ft, pt)] = vals
    return out


class TranscribedRates:
    def __init__(self, green_path=GREEN_CSV, stutzki_path=STUTZKI_T1_CSV):
        self.green = load_green_table3(green_path)
        self.stutzki = load_stutzki_table1(stutzki_path)
        self._green_cache = {}
        self._stutzki_cache = {}

    def green_rate(self, src, tgt, T_k, alpha):
        key = (src, tgt)
        if key not in self._green_cache:
            rates = self.green.get(src, {}).get(tgt)
            self._green_cache[key] = None if rates is None else CubicSpline(
                np.log(GREEN_T_GRID), np.log10(rates / alpha))
        spline = self._green_cache[key]
        return None if spline is None else 10.0 ** spline(np.log(T_k))

    def ios_factor(self, src, tgt, T_k):
        key = (src, tgt)
        if key not in self._stutzki_cache:
            vals = self.stutzki.get(src, {}).get(tgt)
            if vals is None:
                self._stutzki_cache[key] = None
            elif np.all(vals <= 0):
                self._stutzki_cache[key] = 0.0
            else:
                self._stutzki_cache[key] = CubicSpline(
                    np.log(STUTZKI_T_GRID), np.log10(np.clip(vals, 1e-30, None)))
        spline = self._stutzki_cache[key]
        if spline is None:
            return None
        return spline if isinstance(spline, float) else 10.0 ** spline(np.log(T_k))


_RATES_SINGLETON = None


def get_rates():
    global _RATES_SINGLETON
    if _RATES_SINGLETON is None:
        _RATES_SINGLETON = TranscribedRates()
    return _RATES_SINGLETON


def build_complete_original_collision_matrix(model, T_k, alpha=rst.ALPHA_HE_TO_H2,
                                              include_table2=True, coverage=None):
    """Loreau baseline everywhere, overwritten wherever the transcribed
    Green Table III x Stutzki Table 1 data covers a pathway. `coverage`,
    if passed a dict, is updated in place with {'stutzki': n, 'loreau': n}
    off-diagonal entry counts among the levels this function actually
    touches, so completeness can be reported rather than assumed.
    """
    rates = get_rates()
    Cmat = model.collision_matrix(T_k).copy()
    par_letter = {+1: 'L', -1: 'U'}

    if include_table2:
        log_T2 = np.log(rst.TABLE2_T_GRID)
        for (J, K, Ff, Ft), rates_He in rst.TABLE2_NH3_HE.items():
            rates_H2 = np.array(rates_He) / alpha
            spline = CubicSpline(log_T2, np.log10(rates_H2))
            rate_ft = 10.0 ** spline(np.log(T_k))
            for sym in (+1, -1):
                i_from = rst._level_index(model, J, K, sym, Ff)
                i_to = rst._level_index(model, J, K, sym, Ft)
                g_from, g_to = model.g[i_from], model.g[i_to]
                Cmat[i_from, i_to] = rate_ft
                Cmat[i_to, i_from] = rate_ft * g_from / g_to
            if coverage is not None:
                coverage['stutzki'] = coverage.get('stutzki', 0) + 2

    for (Jf, Kf), (Jt, Kt) in PAIR_DIRECTIONS:
        for pf_sym in (+1, -1):
            pf_l = par_letter[pf_sym]
            for pt_sym in (+1, -1):
                pt_l = par_letter[pt_sym]
                k_rot = rates.green_rate((Jf, Kf, pf_l.lower()), (Jt, Kt, pt_l.lower()), T_k, alpha)
                if k_rot is None:
                    if coverage is not None:
                        coverage['loreau'] = coverage.get('loreau', 0) + 2 * len(F_OF[(Jf, Kf)]) * len(F_OF[(Jt, Kt)])
                    continue
                lookup_pf_l = pf_l if pt_l == 'U' else ('U' if pf_l == 'L' else 'L')
                for Ff in F_OF[(Jf, Kf)]:
                    for Ft in F_OF[(Jt, Kt)]:
                        g = rates.ios_factor((Jf, Kf, Ff, lookup_pf_l), (Jt, Kt, Ft, 'U'), T_k)
                        if g is None:
                            if coverage is not None:
                                coverage['loreau'] = coverage.get('loreau', 0) + 2
                            continue
                        if coverage is not None:
                            coverage['stutzki'] = coverage.get('stutzki', 0) + 2
                        i_from = rst._level_index(model, Jf, Kf, pf_sym, Ff)
                        i_to = rst._level_index(model, Jt, Kt, pt_sym, Ft)
                        k_up = g * k_rot
                        g_from, g_to = model.g[i_from], model.g[i_to]
                        E_from, E_to = model.E_K[i_from], model.E_K[i_to]
                        Cmat[i_from, i_to] = k_up
                        Cmat[i_to, i_from] = k_up * (g_from / g_to) * np.exp((E_to - E_from) / T_k)
    return Cmat


def coverage_report(model, T_k=24.0, alpha=rst.ALPHA_HE_TO_H2):
    cov = {}
    build_complete_original_collision_matrix(model, T_k, alpha=alpha, coverage=cov)
    total = cov.get('stutzki', 0) + cov.get('loreau', 0)
    return cov, total


if __name__ == '__main__':
    import nh3_escape_model as m
    model = m.NH3Model()
    m.assert_expected_grouping(model)
    cov, total = coverage_report(model, T_k=24.0)
    print(f"coverage at T=24K: stutzki-sourced={cov.get('stutzki',0)}  "
          f"loreau-fallback={cov.get('loreau',0)}  total={total}  "
          f"({100*cov.get('stutzki',0)/total:.1f}% Stutzki-sourced)")
