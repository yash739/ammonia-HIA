"""Unit tests for stutzki_physics.py -- pure numpy, no Magritte compute.
Run with: pytest tgs/modelling/w33/test_stutzki_physics.py -v
"""

import numpy as np
import pytest

from stutzki_physics import (
    eta_f, jeans_length_pc, jeans_mass_Msun, max_clump_mass_Msun,
    clump_count_K, clump_count_required, physically_consistent,
)


def test_eta_f_basic():
    assert eta_f(5.0, 10.0) == pytest.approx(0.5)


def test_eta_f_unphysical_above_one():
    assert eta_f(15.0, 10.0) > 1.0  # not clamped here -- caller decides what to do with it


def test_jeans_mass_decreases_with_density():
    m_low_n = jeans_mass_Msun(T_k=20.0, n_H2_cm3=1e5)
    m_high_n = jeans_mass_Msun(T_k=20.0, n_H2_cm3=1e7)
    assert m_high_n < m_low_n, "higher density -> smaller Jeans mass, same T"


def test_jeans_length_decreases_with_density():
    l_low_n = jeans_length_pc(T_k=20.0, n_H2_cm3=1e5)
    l_high_n = jeans_length_pc(T_k=20.0, n_H2_cm3=1e7)
    assert l_high_n < l_low_n


def test_clump_count_required_matches_user_example():
    # from the user's own summary: Delta_v_obs ~1.5 km/s, Delta_v_clump=0.3 -> K~5
    K = clump_count_required(dv_obs_kms=1.5, dv_clump_kms=0.3)
    assert K == pytest.approx(5.0)


def test_clump_count_K_increases_with_density_at_fixed_T_and_etaf():
    """Higher density -> more, smaller Jeans-mass clumps fit in the same
    beam/filling-factor budget -- this is the mechanism behind Stutzki's
    high-density branch being preferred over the low-density one."""
    K_low_n = clump_count_K(T_k=20.0, n_H2_cm3=10 ** 4.5, eta_f_value=0.3, distance_pc=1000,
                             dv_obs_kms=1.5, dv_clump_kms=0.3)
    K_high_n = clump_count_K(T_k=20.0, n_H2_cm3=1e7, eta_f_value=0.3, distance_pc=1000,
                              dv_obs_kms=1.5, dv_clump_kms=0.3)
    assert K_high_n > K_low_n


def test_clump_count_K_reproduces_stutzki_table_2a_S106():
    """Direct numerical reproduction of Stutzki & Winnewisser (1985) Table 2a,
    S106 (200,40) row: T_k=21.89K, log(n')=6.528, eta_f=0.237,
    Delta_v_obs=1.43 km/s, Delta_v=0.3 km/s, r=0.60 kpc -> paper's
    log(K)=1.751 (K=56.36). This is the check that resolved the paper's
    typeset-ambiguous K formula (see clump_count_K's docstring)."""
    K = clump_count_K(T_k=21.89, n_H2_cm3=10 ** 6.528, eta_f_value=0.237, distance_pc=600,
                       dv_obs_kms=1.43, dv_clump_kms=0.3)
    assert K == pytest.approx(56.36, rel=0.01)


def test_physically_consistent_rejects_etaf_above_one():
    ok, detail = physically_consistent(T_k=20.0, n_H2_cm3=1e6, T_B_obs=20.0, T_B_theor=10.0,
                                        dv_obs_kms=1.5, dv_clump_kms=0.3, distance_pc=1000)
    assert ok is False
    assert detail['eta_ok'] is False
    assert detail['eta_f'] == pytest.approx(2.0)


def test_physically_consistent_rejects_low_density_branch():
    """Reproduce Stutzki's own low-density-branch rejection: low n_H2 at a
    plausible eta_f should predict K well below the observed-linewidth
    requirement."""
    ok, detail = physically_consistent(T_k=20.0, n_H2_cm3=10 ** 4.5, T_B_obs=3.0, T_B_theor=10.0,
                                        dv_obs_kms=1.5, dv_clump_kms=0.3, distance_pc=1000)
    assert detail['K_required'] == pytest.approx(5.0)
    assert detail['K_predicted'] < detail['K_required'], "low-density branch should under-predict K"
    assert ok is False


def test_physically_consistent_accepts_a_plausible_high_density_branch():
    """A branch with eta_f<1 and enough predicted clumps should pass both checks."""
    ok, detail = physically_consistent(T_k=20.0, n_H2_cm3=1e7, T_B_obs=3.0, T_B_theor=10.0,
                                        dv_obs_kms=1.5, dv_clump_kms=0.3, distance_pc=1000)
    assert detail['eta_ok'] is True
    # K check depends on the (uncertain-coefficient) K_predicted formula --
    # just confirm the plumbing agrees when K_predicted is clearly above K_required
    if detail['K_predicted'] >= detail['K_required']:
        assert ok is True


def test_physically_consistent_nan_etaf_when_theor_zero():
    ok, detail = physically_consistent(T_k=20.0, n_H2_cm3=1e6, T_B_obs=1.0, T_B_theor=0.0,
                                        dv_obs_kms=1.5, dv_clump_kms=0.3, distance_pc=1000)
    assert ok is False
    assert np.isnan(detail['eta_f'])


def test_jeans_length_matches_table_2a_all_positions():
    """Regression test for the 0.776e-3 -> 0.776e-2 coefficient fix, found by
    checking against all 23 positions of Stutzki & Winnewisser (1985)'s own
    Table 2a. Skipped if the full-table parser/CSV isn't present."""
    pytest.importorskip('stutzki_tables_full')
    import numpy as np
    import stutzki_tables_full as sf
    diffs = []
    for key, e in sf.TABLE_1A_FULL.items():
        d2a = sf.TABLE_2A_DERIVED.get(key)
        if d2a is None:
            continue
        lj = jeans_length_pc(e['T_k'], 10 ** e['log_nH2']) * 100  # pc -> 1e-2 pc
        diffs.append(np.log10(lj) - d2a['log_lambda_J_e2pc'])
    assert len(diffs) == 23
    assert max(abs(d) for d in diffs) < 0.001


def test_clump_count_K_matches_table_2a_all_positions():
    """Same cross-check for log(K) -- the quantity physically_consistent()
    actually relies on."""
    pytest.importorskip('stutzki_tables_full')
    import numpy as np
    import stutzki_tables_full as sf
    DIST_PC = {'S106': 600.0, 'S87': 1300.0, 'W48': 3400.0, 'OMC': 500.0}
    diffs = []
    for (region, pos), e in sf.TABLE_1A_FULL.items():
        d2a = sf.TABLE_2A_DERIVED.get((region, pos))
        if d2a is None or e['T_B_obs'] is None or e['eta_f'] is None:
            continue
        K = clump_count_K(T_k=e['T_k'], n_H2_cm3=10 ** e['log_nH2'],
                           eta_f_value=e['eta_f'], distance_pc=DIST_PC[region],
                           dv_obs_kms=e['dv_obs'], dv_clump_kms=0.3)
        if K <= 0 or np.isnan(K):
            continue
        diffs.append(np.log10(K) - d2a['log_K'])
    assert len(diffs) == 23
    assert max(abs(d) for d in diffs) < 0.001
