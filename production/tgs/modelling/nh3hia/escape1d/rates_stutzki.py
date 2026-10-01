"""Stutzki & Winnewisser's (1985) own collision rates for the 1D model, rebuilt
from Green (1980) Table III (NH3-He state-to-state rotational rates) and
Stutzki & Winnewisser (1985a; "S85a") Tables 1-2 (IOS hyperfine factors and
quasi-elastic rates), transcribed in production/references/.

How the rates are assembled (S85a Sect. 4):
* inter-manifold and within-manifold inversion-changing rates:
  k(J K F p -> J' K' F' p') = g(F,p -> F',U) x k_Green(J K p -> J' K' p'),
  with S85a's rule that a lower-inversion-level (L) target takes the factor
  of the opposite source parity's U-target entry; de-excitation by detailed
  balance;
* same-parity hyperfine changes within a manifold: S85a Table 2 directly.

DENSITY AXIS. The rates are used as NH3-He rates, exactly as S85 did, so the
density this rate set is paired with is S85's n' (their Sect. 4: C_He n' =
C_He n_He + C_H2 n_H2, with NH3-H2 rates 1.5x the NH3-He rates). A grid built
with these rates therefore has n' on its density axis, directly comparable
with S85's Table 1a and Figs. 4-6; n_H2 = n' / 1.75 for X(He/H2) = 0.25.

History (audit 2026-09-30, important_notes sec. 22): the first version of
this module divided the He rates by 1.5 (inverting S85's statement), omitted
the within-manifold inversion-changing rates (left at Loreau values: 53-77%
of each (1,1) sublevel's collisional outflow), applied S85a's "IOS rates
identically 0" mark to one row of each 3x3 block instead of the whole block,
and used a Green Table III transcription with the key (1,1)<->(2,1) rates
35% low at 30 K. All four are corrected here and in the reference CSVs.
Every off-diagonal entry among the 36 levels is now S85-sourced.

Assumptions to state when using these rates: S85a Table 2's last column is
printed "340 K" and taken as 100 K (as Tables 1, 3, 4); its (2,2) F 1->2
entry printed 3.18E-10 is taken as 3.18E-11 (the column's trend); S85 cite
Green (1981, NASA TM 83869) for rates computed with the same potential as
Green (1980) Table III, which is used here.
"""
import os

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import nh3hia.escape1d.rates_table2 as rst  # TABLE2_NH3_HE, ALPHA_HE_TO_H2, _level_index
from nh3hia import paths

_REF_DIR = paths.PRODUCTION + "/references"
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

    def green_rate(self, src, tgt, T_k):
        key = (src, tgt)
        if key not in self._green_cache:
            rates = self.green.get(src, {}).get(tgt)
            self._green_cache[key] = None if rates is None else CubicSpline(
                np.log(GREEN_T_GRID), np.log10(rates))
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


def build_s85_collision_matrix(model, T_k, include_table2=True, coverage=None):
    """Collision matrix (cm^3 s^-1, NH3-He; pair with S85's density n') at T_k.

    Starts from Loreau's matrix only so that any entry the S85 tables do not
    define would be visible; `coverage` (a dict, updated in place) counts the
    off-diagonal entries that are S85-sourced vs still Loreau, by number and by
    rate-weighted share of each level's collisional outflow."""
    rates = get_rates()
    Cmat = model.collision_matrix(T_k).copy()
    touched = np.zeros_like(Cmat, dtype=bool)
    par_letter = {+1: 'L', -1: 'U'}
    idx = lambda J, K, sym, F: rst._level_index(model, J, K, sym, F)

    def put(i_from, i_to, k_up):
        Cmat[i_from, i_to] = k_up
        Cmat[i_to, i_from] = (k_up * (model.g[i_from] / model.g[i_to])
                              * np.exp((model.E_K[i_to] - model.E_K[i_from]) / T_k))
        touched[i_from, i_to] = touched[i_to, i_from] = True

    if include_table2:   # same-parity hyperfine changes within a manifold
        log_T2 = np.log(rst.TABLE2_T_GRID)
        for (J, K, Ff, Ft), rates_He in rst.TABLE2_NH3_HE.items():
            rate_ft = 10.0 ** CubicSpline(log_T2, np.log10(rates_He))(np.log(T_k))
            for sym in (+1, -1):
                put(idx(J, K, sym, Ff), idx(J, K, sym, Ft), rate_ft)   # dE ~ 0

    # inter-manifold, plus within-manifold inversion-changing (l -> u)
    pairs = [((Jf, Kf), (Jt, Kt), pf, pt) for (Jf, Kf), (Jt, Kt) in PAIR_DIRECTIONS
             for pf in (+1, -1) for pt in (+1, -1)]
    pairs += [(jk, jk, +1, -1) for jk in MANIFOLDS]
    for (Jf, Kf), (Jt, Kt), pf_sym, pt_sym in pairs:
        pf_l, pt_l = par_letter[pf_sym], par_letter[pt_sym]
        k_rot = rates.green_rate((Jf, Kf, pf_l.lower()), (Jt, Kt, pt_l.lower()), T_k)
        if k_rot is None:
            continue
        lookup_pf_l = pf_l if pt_l == 'U' else ('U' if pf_l == 'L' else 'L')
        for Ff in F_OF[(Jf, Kf)]:
            for Ft in F_OF[(Jt, Kt)]:
                g = rates.ios_factor((Jf, Kf, Ff, lookup_pf_l), (Jt, Kt, Ft, 'U'), T_k)
                if g is None:
                    continue
                put(idx(Jf, Kf, pf_sym, Ff), idx(Jt, Kt, pt_sym, Ft), g * k_rot)

    if coverage is not None:
        off = ~np.eye(len(Cmat), dtype=bool)
        nz = off & (Cmat > 0)
        coverage['s85_entries'] = int((nz & touched).sum())
        coverage['loreau_entries'] = int((nz & ~touched).sum())
        out = np.where(off, Cmat, 0.0).sum(axis=1)
        lor = np.where(off & ~touched, Cmat, 0.0).sum(axis=1)
        coverage['max_loreau_outflow_share'] = float(np.max(lor / out))
    return Cmat


def coverage_report(model, T_k=24.0):
    cov = {}
    build_s85_collision_matrix(model, T_k, coverage=cov)
    return cov


if __name__ == '__main__':
    import nh3hia.escape1d.model as m
    model = m.NH3Model()
    for T in (15.0, 24.0, 36.0, 48.0):
        print(f"T={T:.0f} K:", coverage_report(model, T_k=T))
