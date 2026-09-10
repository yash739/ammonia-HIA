"""Bounded experiment: does swapping in Stutzki & Winnewisser's OWN
intra-multiplet quasi-elastic hyperfine collision rates (their 1985a paper,
Table 2 -- NH3-He, IOS approximation) for the corresponding entries of the
Loreau et al. (2023) collision matrix pull the T=36K R(F1=1->0) minimum back
out of population inversion and toward the ~0.25-0.3 floor visible in
Stutzki & Winnewisser's OWN Fig. 1 (same paper, p.11) -- rather than the full
masing this pipeline finds with Loreau's rates alone?

This is deliberately NOT a full reproduction of Stutzki's original
calculation: Table 2 only covers the quasi-elastic (intra-(J,K)-multiplet,
Delta F1 within one rotation-inversion level) collisional pathways -- the
rotational-level-changing rates (needed for the rest of the 36-level system)
would additionally require Table 1's IOS *relative factors* combined with
Green's (1980) absolute CS rotational rates, an unpublished NASA technical
memo not available here. So this swaps ONLY the specific collisional
pathway plausibly most relevant to the outer-satellite anomaly (the one
redistributing population among F1 sublevels within a manifold, directly
competing against the radiative selective-trapping mechanism for those same
sublevels) and leaves every other entry (including all rotational-changing
rates) as Loreau's.

NH3-He -> NH3-H2: divided by alpha=1.5, per stutzki_1985.pdf's own Sect. 4
("NH3-He rates are estimated to be a factor alpha=1.5 higher than the
NH3-H2 rates (Green, 1981)").

Table 2's temperature header for its last column is transcribed here as
100 K, matching Table 1's explicitly stated 15/25/50/100 K grid and the
smooth row-to-row trend -- the OCR text handed to this session for that one
column literally read "340 K", which is almost certainly a scan artifact
(no other table in the paper uses 340 K), but this is flagged rather than
silently resolved.
"""
import numpy as np
from scipy.interpolate import CubicSpline

import nh3_escape_model as m

ALPHA_HE_TO_H2 = 1.5

# (J, K, F_from, F_to): rates in cm^3/s at T = [15, 25, 50, 100] K, NH3-He, IOS.
# Stutzki & Winnewisser (1985a), Table 2.
TABLE2_NH3_HE = {
    (1, 1, 0, 2): [3.57e-11, 2.49e-11, 2.04e-11, 1.05e-11],
    (1, 1, 0, 1): [9.32e-11, 8.66e-11, 7.07e-11, 3.37e-11],
    (1, 1, 2, 1): [3.94e-11, 3.28e-11, 2.69e-11, 1.31e-11],
    (2, 2, 1, 3): [2.79e-11, 1.92e-11, 1.56e-11, 9.18e-12],
    (2, 2, 1, 2): [7.59e-11, 7.02e-11, 5.84e-11, 3.18e-11],
    (2, 2, 3, 2): [4.48e-11, 3.88e-11, 3.21e-11, 1.77e-11],
    (2, 1, 2, 3): [6.09e-11, 5.26e-11, 4.32e-11, 2.08e-11],
    (2, 1, 2, 1): [4.46e-11, 4.14e-11, 3.45e-11, 1.72e-11],
    (2, 1, 3, 1): [1.13e-11, 7.41e-12, 5.74e-12, 2.32e-12],
    (3, 2, 3, 4): [5.70e-11, 4.96e-11, 4.14e-11, 2.07e-11],
    (3, 2, 3, 2): [4.66e-11, 4.28e-11, 3.58e-11, 1.83e-11],
    (3, 2, 4, 2): [1.26e-11, 8.06e-12, 6.49e-12, 2.82e-12],
    (3, 1, 3, 4): [5.76e-11, 5.09e-11, 4.21e-11, 2.09e-11],
    (3, 1, 3, 2): [4.71e-11, 4.38e-11, 3.66e-11, 1.86e-11],
    (3, 1, 4, 2): [1.27e-11, 8.30e-12, 6.35e-12, 2.74e-12],
    (4, 4, 3, 5): [2.07e-11, 1.37e-11, 1.06e-11, 7.45e-12],
    (4, 4, 3, 4): [6.17e-11, 5.70e-11, 4.82e-11, 3.03e-11],
    (4, 4, 5, 4): [4.56e-11, 4.05e-11, 3.40e-11, 2.16e-11],
}
TABLE2_T_GRID = np.array([15.0, 25.0, 50.0, 100.0])


def _level_index(model, J, K, sym, F1):
    for i, (Jl, Kl, syml, F1l) in enumerate(model.qn):
        if (Jl, Kl, syml, F1l) == (J, K, sym, F1):
            return i
    raise KeyError((J, K, sym, F1))


def build_hybrid_collision_matrix(model, T_k, alpha=ALPHA_HE_TO_H2, include_21=True):
    """Loreau baseline everywhere, EXCEPT the 18 intra-multiplet quasi-elastic
    pathways of Table 2, applied identically to both inversion parities
    (sym=+1 and sym=-1) per the paper's own statement that the rates are the
    same for both inversion levels."""
    Cmat = model.collision_matrix(T_k).copy()
    log_T = np.log(TABLE2_T_GRID)

    for (J, K, Ff, Ft), rates_He in TABLE2_NH3_HE.items():
        rates_H2 = np.array(rates_He) / alpha
        spline = CubicSpline(log_T, np.log10(rates_H2))
        rate_ft = 10.0 ** spline(np.log(T_k))  # rate(Ff -> Ft)
        for sym in (+1, -1):
            i_from = _level_index(model, J, K, sym, Ff)
            i_to = _level_index(model, J, K, sym, Ft)
            g_from, g_to = model.g[i_from], model.g[i_to]
            Cmat[i_from, i_to] = rate_ft
            Cmat[i_to, i_from] = rate_ft * g_from / g_to  # detailed balance, dE~0

    if include_21:
        _apply_11_21_rotational_swap(model, Cmat, T_k, alpha)

    return Cmat


# ---------------------------------------------------------------------------
# The (1,1)<->(2,1) ROTATIONAL collision rate -- the pathway that actually
# competes against the (2,1)->(1,1) FIR radiative trapping mechanism, and
# per the earlier Table-2-only experiment, the more likely lever for the
# masing discrepancy (that experiment changed the quasi-elastic rates by up
# to 36x and only shifted R(1->0) by ~0.03-0.05 -- nowhere near enough).
#
# Absolute rotational rates: Green (1980), Table III (NH3-He, HF+long-range
# potential, CS/B10 -- this is the "Green (1980) CS-calculation" Stutzki &
# Winnewisser (1985a) explicitly cite for combining with their IOS relative
# factors). Read directly off the scanned table image (p.2745) and verified
# internally: values increase smoothly and monotonically with T for all four
# (parity) combinations, as physically expected.
#
# Relative factors g_FF': Stutzki & Winnewisser (1985a)'s OWN IOS values, but
# NOT read from the noisy 1985 scan -- taken instead from Loreau et al.
# (2023) Table 3, which quotes them in parentheses directly. Self-consistent
# (each row sums to 1.000, as it must: g_FF' = rate_hyperfine/rate_rotational
# so Sum_F' g_FF' = 1) and matches this session's independent transcription
# of the scanned Table 1 image. Held constant in T (paper's Sect. 4: "the IOS
# and recoupling relative factors are almost independent of temperature ...
# up to temperatures of at least 100 K"), so only the rotational rate needs a
# T-spline.
#
# Parity bookkeeping: Loreau's Table 3 only tabulates source=(1,1)_lower
# (1_1+ -> 2_1+ and 1_1+ -> 2_1-). Reusing it for source=(1,1)_upper is an
# ASSUMPTION -- justified only by the IOS approximation neglecting the
# inversion splitting the same way it neglects hyperfine splitting (both
# small compared to collision energy, Stutzki 1985a Sect. 1), giving IOS no
# dynamical handle to distinguish which parity is source vs target. This is
# the one genuinely unverified piece of this combination, flagged rather
# than smoothed over.
# ---------------------------------------------------------------------------
GREEN_TABLE3_T_GRID = np.array([15.0, 30.0, 100.0, 300.0])

# Green (1980) Table III, NH3-He, cm^3/s, at T=[15,30,100,300]K.
GREEN_11_21_ROT_RATES_HE = {
    ('L', 'L'): [1.2e-13, 9.6e-13, 5.0e-12, 1.1e-11],
    ('L', 'U'): [7.0e-13, 5.4e-12, 2.9e-11, 6.2e-11],
    ('U', 'L'): [8.0e-13, 5.8e-12, 3.0e-11, 6.3e-11],
    ('U', 'U'): [1.3e-13, 9.5e-13, 4.9e-12, 1.1e-11],
}

# Stutzki & Winnewisser (1985a) IOS relative factors g_FF', at 25K, quoted
# verbatim (parenthetical values) in Loreau et al. (2023) Table 3.
STUTZKI_11_21_GFF_SAME = {  # source parity == target parity
    0: {1: 0.08, 2: 0.65, 3: 0.27},
    1: {1: 0.31, 2: 0.20, 3: 0.48},
    2: {1: 0.16, 2: 0.35, 3: 0.50},
}
STUTZKI_11_21_GFF_OPP = {  # source parity != target parity
    0: {1: 0.57, 2: 0.28, 3: 0.15},
    1: {1: 0.27, 2: 0.51, 3: 0.23},
    2: {1: 0.09, 2: 0.67, 3: 0.24},
}


def _apply_11_21_rotational_swap(model, Cmat, T_k, alpha):
    log_T = np.log(GREEN_TABLE3_T_GRID)
    rot_rate = {}
    for (sym_from, sym_to), rates_He in GREEN_11_21_ROT_RATES_HE.items():
        rates_H2 = np.array(rates_He) / alpha
        spline = CubicSpline(log_T, np.log10(rates_H2))
        rot_rate[(sym_from, sym_to)] = 10.0 ** spline(np.log(T_k))

    sym_code = {'L': +1, 'U': -1}
    for sym_from_code in ('L', 'U'):
        for sym_to_code in ('L', 'U'):
            gff = (STUTZKI_11_21_GFF_SAME if sym_from_code == sym_to_code
                   else STUTZKI_11_21_GFF_OPP)
            k_rot = rot_rate[(sym_from_code, sym_to_code)]
            sym_from, sym_to = sym_code[sym_from_code], sym_code[sym_to_code]
            for Ff, row in gff.items():
                i_from = _level_index(model, 1, 1, sym_from, Ff)
                for Ft, g in row.items():
                    i_to = _level_index(model, 2, 1, sym_to, Ft)
                    k_up = g * k_rot  # excitation (1,1) -> (2,1): E increases
                    E_from, E_to = model.E_K[i_from], model.E_K[i_to]
                    g_from, g_to = model.g[i_from], model.g[i_to]
                    k_down = k_up * (g_from / g_to) * np.exp((E_to - E_from) / T_k)
                    Cmat[i_from, i_to] = k_up
                    Cmat[i_to, i_from] = k_down


def compare_at(model, T_k, n_H2, log_N_dv_vals):
    Cmat_loreau = model.collision_matrix(T_k)
    Cmat_hybrid = build_hybrid_collision_matrix(model, T_k)

    print(f"\nT_k={T_k}K, n_H2=10^{np.log10(n_H2):.2f}")
    print(f"{'log_N_dv':>9} {'tau_main(L)':>12} {'R10(L)':>9} {'tau_main(H)':>12} {'R10(H)':>9}")
    x0_L = x0_H = None
    for log_N in log_N_dv_vals:
        outL = m.run_one(model, Cmat_loreau, T_k=T_k, n_H2=n_H2, log_N_dv=log_N, x0=x0_L)
        outH = m.run_one(model, Cmat_hybrid, T_k=T_k, n_H2=n_H2, log_N_dv=log_N, x0=x0_H)
        x0_L, x0_H = outL['x'], outH['x']
        r10L = outL['R_10'] if np.isfinite(outL['R_10']) else float('nan')
        r10H = outH['R_10'] if np.isfinite(outH['R_10']) else float('nan')
        print(f"{log_N:9.2f} {outL['tau_main']:12.4f} {r10L:9.4f} "
              f"{outH['tau_main']:12.4f} {r10H:9.4f}"
              f"{'  MASER(L)' if not np.isfinite(outL['R_10']) or outL['any_maser'] else ''}"
              f"{'  MASER(H)' if not np.isfinite(outH['R_10']) or outH['any_maser'] else ''}")


if __name__ == '__main__':
    model = m.NH3Model()
    m.assert_expected_grouping(model)
    # Stutzki's own Fig.1 closed "0.3" loop sits around n_H2~10^5, N~14.6-15.0
    # at T=36K -- scan a small neighbourhood of that.
    for log_n in (4.5, 5.0, 5.5):
        compare_at(model, 36.0, 10 ** log_n, np.arange(14.2, 15.61, 0.1))
