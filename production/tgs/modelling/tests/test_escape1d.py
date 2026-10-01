"""Tests for the 1D escape-probability model (nh3hia.escape1d) and its
reconstruction of Stutzki & Winnewisser's (1985, "S85") collision rates.

The last test is the strongest: run at S85's own Table 1a best-fit parameters
with S85's own rates, the model must reproduce S85's own model outputs
(T_B,theor of the (1,1) main line, and the (2,1)/(1,1) predictions of their
Table 3). No Magritte compute; runs in well under a minute.
"""
import numpy as np
import pandas as pd
import pytest

from nh3hia import paths
from nh3hia.escape1d import model as m, grid as G, rates_stutzki as rs, rates_table2 as rt
from nh3hia.stutzki import params as sp


@pytest.fixture(scope='module')
def M():
    return m.NH3Model()


def _group_of(M, Ju, Ku, Jl, Kl, F1u, F1l, sym_u):
    for t, L in enumerate(M.labels):
        if (L['Ju'], L['Ku'], L['Jl'], L['Kl'], L['F1u'], L['F1l']) == (Ju, Ku, Jl, Kl, F1u, F1l) \
                and M.qn[M.u_idx[t]][2] == sym_u:
            return M.group_id[t]
    raise KeyError


def test_grouping_matches_s85(M):
    """S85 Sect. 3: 12 overlapping groups + 28 single lines; both (2,1)->(1,1)
    outer satellites (F1=1->0 and F1=2->1) single in each parity band."""
    m.assert_expected_grouping(M)
    sizes = np.bincount(M.group_id)
    for sym_u in (+1, -1):
        for F1u, F1l in ((1, 0), (2, 1)):
            assert sizes[_group_of(M, 2, 1, 1, 1, F1u, F1l, sym_u)] == 1
        assert sizes[_group_of(M, 2, 1, 1, 1, 3, 2, sym_u)] == 4   # main + inner satellites


def test_linewidth_is_fwhm(M):
    assert M.dv_equiv_kms == pytest.approx(0.3 * 1.0644670, rel=1e-6)


def test_beta_eq4_continued_to_negative_tau():
    b = m.get_beta_spline()
    assert b(np.array([-1e-6]))[0] == pytest.approx(1.0, abs=1e-5)
    assert b(np.array([1e-6]))[0] == pytest.approx(1.0, abs=1e-5)
    for tau in (-0.5, -2.0, -10.0):   # Eq. (4) evaluated directly
        z = np.linspace(-9, 9, 20001)
        direct = np.trapezoid(np.exp(-z * z - tau * np.exp(-z * z)), z) / np.sqrt(np.pi)
        assert b(np.array([tau]))[0] == pytest.approx(direct, rel=1e-4)
        assert b(np.array([tau]))[0] > 1.0


def test_s85_rates_complete_and_unscaled(M):
    """Every off-diagonal entry S85-sourced; He rates used as-is (S85's n')."""
    for T in (15.0, 30.0, 48.0):
        cov = rs.coverage_report(M, T_k=T)
        assert cov['loreau_entries'] == 0 and cov['max_loreau_outflow_share'] == 0.0
    C = rs.build_s85_collision_matrix(M, 15.0)
    i, j = rt._level_index(M, 1, 1, +1, 0), rt._level_index(M, 1, 1, +1, 1)
    assert C[i, j] == pytest.approx(9.32e-11, rel=1e-6)   # S85a Table 2, 15 K, no alpha


def test_ios_zero_blocks(M):
    """S85a Table 1 '*)': (1,1)L -> (2,2)U and (2,2)L -> (4,4)U are identically
    zero for the whole 3x3 block (and, by S85a's parity rule, (1,1)U -> (2,2)L)."""
    C = rs.build_s85_collision_matrix(M, 24.0)
    for F in (0, 1, 2):
        for Ft in (1, 2, 3):
            assert C[rt._level_index(M, 1, 1, +1, F), rt._level_index(M, 2, 2, -1, Ft)] == 0.0
            assert C[rt._level_index(M, 1, 1, -1, F), rt._level_index(M, 2, 2, +1, Ft)] == 0.0


def test_within_manifold_inversion_rates(M):
    """(1,1) L F -> U F' = Table 1 factor x Green's 1,1l->1,1u rate."""
    C = rs.build_s85_collision_matrix(M, 15.0)
    i = rt._level_index(M, 1, 1, +1, 0)
    assert C[i, rt._level_index(M, 1, 1, -1, 2)] == pytest.approx(0.27 * 6.7e-11, rel=1e-3)
    assert C[i, rt._level_index(M, 1, 1, -1, 1)] == pytest.approx(0.73 * 6.7e-11, rel=1e-3)


def test_green_transcription_fixes():
    g = pd.read_csv(f'{paths.REFERENCES}/Green_Table3_1980.csv')
    row = g[(g.Initial == '1, 1l') & (g.Final == '2, 1u')].iloc[0]
    assert (row['30K'], row['300K']) == (5.4e-12, 6.2e-11)
    row = g[(g.Initial == '1, 1u') & (g.Final == '2, 1l')].iloc[0]
    assert row['30K'] == 5.8e-12


def test_no_linewidth_dependence_at_fixed_N_over_dv():
    """With S85's fixed overlap scheme, results depend on N/dv only."""
    out = []
    for dv in (0.2, 0.8):
        Mdv = m.NH3Model(dv_clump_kms=dv)
        C = G.RATES_FN['s85'](Mdv, 30.0)
        o = m.run_one(Mdv, C, 30.0, 10 ** 5.5, 15.0, dv_clump_kms=dv, guard_masers=False)
        out.append((o['tau_main'], o['R_01'], o['R_21_11']))
    assert np.allclose(out[0], out[1], rtol=1e-6)


def test_reproduces_s85_own_model_at_table1a(M):
    """At S85's 23 Table 1a fits (n' axis, S85 rates): T_B(1,1) within 5% of
    their T_B,theor everywhere, and (2,1)/(1,1) within 10% of their Table 3
    theoretical values."""
    theory_21 = {('S106', '0,0'): 0.018, ('OMC', 'S3'): 0.052, ('OMC', 'S4'): 0.068}
    for reg, d in sp.TABLE_1A.items():
        for pos, p in d.items():
            T = p['T_k']
            C = G.RATES_FN['s85'](M, T)
            o = m.run_one(M, C, T, 10 ** p['log_nH2'], p['log_N_NH3'] - np.log10(0.3),
                          guard_masers=False)
            assert o['converged']
            assert o['T_B_main'] == pytest.approx(p['T_B_theor'], rel=0.05), (reg, pos)
            if (reg, pos) in theory_21:
                assert o['R_21_11'] == pytest.approx(theory_21[(reg, pos)], rel=0.10), (reg, pos)
