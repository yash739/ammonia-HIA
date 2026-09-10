"""Maximal-coverage Stutzki-rate collision matrix: everything extractable
from Green (1980) Table III (NH3-He absolute rotational rates) and Stutzki &
Winnewisser (1985a) Table 1 (IOS relative factors), for SOURCE=(1,1) (both
inversion parities) to every target Table 1 covers -- its own inversion
transition, (2,2), (2,1), (3,2), (3,1), (4,4) -- plus the Table 2 quasi
-elastic intra-multiplet rates already in rate_swap_test.py. This is the
complete outgoing rotational network for the (1,1) level, which is what
every reported ratio (R_01, R_10, R_12, R_21, R_2211, T_B_main) directly
depends on.

NOT covered (Table 1 doesn't tabulate these; still Loreau's modern rates):
source=(2,2) or (2,1) or higher to targets other than what's reachable via
detailed balance from the (1,1) entries above -- i.e. the (2,2)<->(2,1),
(2,1)<->(3,x) etc. pathways that don't touch (1,1) directly.

Both PDFs (Stutzki1985a.pdf, green1980.pdf) were re-read directly (not from
memory) for every number here, with every Table 1 row checked against the
Eq.(12a) identity (relative factors to one target's hyperfine sublevels must
sum to 1.000) as an automatic, self-consistency check. That check caught
zero errors in the Table 1 data pulled for this file; a similar careful
re-read of Green's Table III (which has no analogous self-check) DID catch
and fix three transcription errors from an earlier pass (see
table3_raw.txt's edit history / the conversation) -- flagged here since
Table III's numbers have no equivalent internal verification and should be
trusted less than Table 1's.
"""
import numpy as np
from scipy.interpolate import CubicSpline

import rate_swap_test as rst  # reuses TABLE2_NH3_HE quasi-elastic swap, ALPHA_HE_TO_H2

GREEN_T_GRID = np.array([15.0, 30.0, 100.0, 300.0])

# Green (1980) Table III, NH3-He absolute rotational rates, cm^3/s.
# Re-verified directly against the scanned table image; the three entries
# marked below were wrong in an earlier transcription pass (memory-based)
# and are fixed here.
GREEN_TABLE3 = {
    ('1,1L', '1,1U'): [6.7e-11, 6.8e-11, 7.8e-11, 8.7e-11],
    ('1,1L', '2,2L'): [9.1e-13, 4.1e-12, 1.5e-11, 2.7e-11],
    ('1,1L', '2,2U'): [6.8e-14, 2.2e-13, 2.5e-13, 4.1e-13],
    ('1,1L', '2,1L'): [1.2e-13, 9.6e-13, 5.0e-12, 1.1e-11],
    ('1,1L', '2,1U'): [7.0e-13, 5.4e-12, 2.9e-11, 6.2e-11],
    ('1,1L', '3,2L'): [6.6e-16, 5.8e-14, 1.8e-12, 5.8e-12],
    ('1,1L', '3,2U'): [1.0e-15, 9.6e-14, 3.9e-12, 1.7e-11],
    ('1,1L', '3,1L'): [1.4e-16, 1.6e-14, 5.1e-13, 2.4e-12],
    ('1,1L', '3,1U'): [5.1e-16, 6.7e-14, 2.7e-12, 1.0e-11],  # FIXED (was a mistranscribed duplicate)
    ('1,1L', '4,4L'): [8.7e-18, 1.0e-14, 1.8e-12, 1.1e-11],
    ('1,1L', '4,4U'): [8.1e-19, 1.1e-15, 3.2e-13, 2.6e-12],
    ('1,1U', '2,2L'): [8.9e-14, 2.5e-13, 3.8e-13, 4.6e-13],
    ('1,1U', '2,2U'): [9.0e-13, 4.0e-12, 1.5e-11, 2.7e-11],
    ('1,1U', '2,1L'): [8.0e-13, 5.8e-12, 3.0e-11, 6.3e-11],
    ('1,1U', '2,1U'): [1.3e-13, 9.5e-13, 4.9e-12, 1.1e-11],
    ('1,1U', '3,2L'): [1.2e-15, 1.1e-13, 4.1e-12, 1.8e-11],
    ('1,1U', '3,2U'): [6.7e-16, 5.7e-14, 1.8e-12, 6.0e-12],
    ('1,1U', '3,1L'): [1.4e-16, 1.6e-14, 5.3e-13, 2.7e-12],  # FIXED (was swapped with 3,1U row)
    ('1,1U', '3,1U'): [5.6e-16, 7.3e-14, 2.8e-12, 1.1e-11],  # FIXED (was swapped with 3,1L row)
    ('1,1U', '4,4L'): [8.8e-19, 1.1e-15, 3.2e-13, 2.7e-12],
    ('1,1U', '4,4U'): [9.1e-18, 1.0e-14, 1.8e-12, 1.1e-11],
}

# Stutzki & Winnewisser (1985a) Table 1, T=15K (weakly T-dependent per the
# paper's own text), source=(1,1) both parities, to every target covered.
# {(source_parity, source_F, target_JK): {target_F: g}}; target parity is
# always 'U' as tabulated -- the 'L' target is obtained via the Eq.(22b)
# symmetry g(source=v, target=L) = g(source=-v, target=U), applied in code.
TABLE1_FROM_11 = {
    ('L', 0, '1,1'): {2: 0.27, 1: 0.73},          # (1,1) inversion itself
    ('L', 2, '1,1'): {0: 0.054, 2: 0.64, 1: 0.30},
    ('L', 1, '1,1'): {0: 0.24, 2: 0.51, 1: 0.25},

    ('L', 0, '2,2'): {},   # IOS rate identically 0 (footnote *)
    ('L', 2, '2,2'): {},
    ('L', 1, '2,2'): {},
    ('U', 0, '2,2'): {1: 6.27e-02, 3: 0.84, 2: 0.10},
    ('U', 2, '2,2'): {1: 0.31, 3: 0.31, 2: 0.38},
    ('U', 1, '2,2'): {1: 6.09e-02, 3: 0.60, 2: 0.33},

    ('L', 0, '2,1'): {2: 0.28, 3: 0.15, 1: 0.57},
    ('L', 2, '2,1'): {2: 0.24, 3: 0.67, 1: 0.09},
    ('L', 1, '2,1'): {2: 0.51, 3: 0.23, 1: 0.27},
    ('U', 0, '2,1'): {2: 0.68, 3: 0.25, 1: 7.71e-02},
    ('U', 2, '2,1'): {2: 0.24, 3: 0.68, 1: 8.56e-02},
    ('U', 1, '2,1'): {2: 0.50, 3: 0.22, 1: 0.27},

    ('L', 0, '3,2'): {3: 0.13, 4: 0.78, 2: 8.84e-02},
    ('L', 2, '3,2'): {3: 0.36, 4: 0.28, 2: 0.36},
    ('L', 1, '3,2'): {3: 0.36, 4: 0.55, 2: 9.21e-02},
    ('U', 0, '3,2'): {3: 0.66, 4: 0.23, 2: 0.11},
    ('U', 2, '3,2'): {3: 0.35, 4: 0.45, 2: 0.20},
    ('U', 1, '3,2'): {3: 0.19, 4: 0.46, 2: 0.35},

    ('L', 0, '3,1'): {3: 0.67, 4: 0.22, 2: 0.11},
    ('L', 2, '3,1'): {3: 0.36, 4: 0.45, 2: 0.19},
    ('L', 1, '3,1'): {3: 0.18, 4: 0.46, 2: 0.35},
    ('U', 0, '3,1'): {3: 0.22, 4: 0.11, 2: 0.66},
    ('U', 2, '3,1'): {3: 0.26, 4: 0.64, 2: 9.93e-02},
    ('U', 1, '3,1'): {3: 0.49, 4: 0.18, 2: 0.33},

    ('L', 0, '4,4'): {3: 0.15, 5: 0.24, 4: 0.61},
    ('L', 2, '4,4'): {3: 0.23, 5: 0.42, 4: 0.36},
    ('L', 1, '4,4'): {3: 0.35, 5: 0.44, 4: 0.20},
    ('U', 0, '4,4'): {3: 0.68, 5: 0.11, 4: 0.21},
    ('U', 2, '4,4'): {3: 0.12, 5: 0.61, 4: 0.27},
    ('U', 1, '4,4'): {3: 0.36, 5: 0.17, 4: 0.47},
}

TARGET_JK = {'2,2': (2, 2), '2,1': (2, 1), '3,2': (3, 2), '3,1': (3, 1), '4,4': (4, 4)}


def _green_rate_spline(model, init_label, final_label, T_k, alpha):
    rates_He = np.array(GREEN_TABLE3[(init_label, final_label)]) / alpha
    spline = CubicSpline(np.log(GREEN_T_GRID), np.log10(rates_He))
    return 10.0 ** spline(np.log(T_k))


def build_full_hybrid_collision_matrix(model, T_k, alpha=rst.ALPHA_HE_TO_H2, include_table2=True):
    """Loreau baseline everywhere except: (a) Table 2's 18 quasi-elastic
    intra-multiplet rates (unchanged from rate_swap_test.py), (b) the FULL
    (1,1)<->{itself, 2,2; 2,1; 3,2; 3,1; 4,4} rotational network from
    Green Table III x Stutzki Table 1."""
    Cmat = model.collision_matrix(T_k).copy()

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

    sym_code = {'L': +1, 'U': -1}

    # (1,1) inversion transition: only source=L->target=U is tabulated;
    # source=U->target=U is the (physically distinct) same-parity case, not
    # used here. Detailed balance gives the L<-U reverse hyperfine rate.
    k_rot_inv = _green_rate_spline(model, '1,1L', '1,1U', T_k, alpha)
    for Ff in (0, 1, 2):
        row = TABLE1_FROM_11[('L', Ff, '1,1')]
        i_from = rst._level_index(model, 1, 1, +1, Ff)
        for Ft, g in row.items():
            i_to = rst._level_index(model, 1, 1, -1, Ft)
            k_up = g * k_rot_inv
            E_from, E_to = model.E_K[i_from], model.E_K[i_to]
            g_from, g_to = model.g[i_from], model.g[i_to]
            k_down = k_up * (g_from / g_to) * np.exp((E_to - E_from) / T_k)
            Cmat[i_from, i_to] = k_up
            Cmat[i_to, i_from] = k_down

    # (1,1) -> {2,2; 2,1; 3,2; 3,1; 4,4}: both source parities tabulated
    # (against target=U); target=L obtained via the Eq.(22b) symmetry
    # g(source=v, target=L) = g(source=-v, target=U).
    for target_key, (J2, K2) in TARGET_JK.items():
        k_rot = {}
        for sym_from_code in ('L', 'U'):
            for sym_to_code in ('L', 'U'):
                init_label = f'1,1{sym_from_code}'
                final_label = f'{J2},{K2}{sym_to_code}'
                if (init_label, final_label) in GREEN_TABLE3:
                    k_rot[(sym_from_code, sym_to_code)] = _green_rate_spline(
                        model, init_label, final_label, T_k, alpha)

        for sym_from_code in ('L', 'U'):
            for sym_to_code in ('L', 'U'):
                if (sym_from_code, sym_to_code) not in k_rot:
                    continue
                # symmetry: target=L row reuses the OPPOSITE source parity's target=U data
                data_source_parity = sym_from_code if sym_to_code == 'U' else \
                    ('L' if sym_from_code == 'U' else 'U')
                for Ff in (0, 1, 2):
                    row = TABLE1_FROM_11.get((data_source_parity, Ff, target_key), {})
                    if not row:
                        continue
                    i_from = rst._level_index(model, 1, 1, sym_code[sym_from_code], Ff)
                    for Ft, g in row.items():
                        i_to = rst._level_index(model, J2, K2, sym_code[sym_to_code], Ft)
                        k_up = g * k_rot[(sym_from_code, sym_to_code)]
                        E_from, E_to = model.E_K[i_from], model.E_K[i_to]
                        g_from, g_to = model.g[i_from], model.g[i_to]
                        k_down = k_up * (g_from / g_to) * np.exp((E_to - E_from) / T_k)
                        Cmat[i_from, i_to] = k_up
                        Cmat[i_to, i_from] = k_down

    return Cmat
