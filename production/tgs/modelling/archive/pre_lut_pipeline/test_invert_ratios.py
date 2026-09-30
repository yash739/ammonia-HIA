"""Unit tests for invert_ratios.py's grid/bounds/dedup logic -- all mockable,
zero Magritte compute. Run with: pytest tgs/modelling/w33/test_invert_ratios.py -v
"""

import numpy as np
import pytest

from invert_ratios import (
    Bounds, build_grid, task_key, radius_from_N_dv, is_boundary_hugging,
    expand_bounds, shrink_bounds_around, merge_top_k, DEFAULT_CLUMP_DV_KMS,
)


def _row(n_H2, T, N_dv, X=1e-8, r=1e18, score=0.1, dv=DEFAULT_CLUMP_DV_KMS):
    """N_dv: N_NH3/Delta_v [cm^-2/(km/s)] -- converted to N_NH3_target (what
    the real worker rows store) the same way radius_from_N_dv does."""
    return dict(numberdensity=n_H2, T_cloud=T, clump_dv=dv, XNH3=X, radius_sphere=r,
                N_NH3_target=N_dv * dv, score=score)


def test_build_grid_shape():
    b = Bounds(log_n_lo=3.0, log_n_hi=5.0, T_lo=10.0, T_hi=20.0, log_Ndv_lo=14.0, log_Ndv_hi=15.0)
    grid = build_grid(b, (3, 2, 2))
    assert len(grid) == 3 * 2 * 2
    n_vals = sorted({round(g[0], 6) for g in grid})
    assert len(n_vals) == 3
    assert n_vals[0] == pytest.approx(1000.0, rel=1e-3)
    assert n_vals[-1] == pytest.approx(100000.0, rel=1e-3)


def test_build_grid_respects_T_kin_fixed():
    b = Bounds(log_n_lo=3.0, log_n_hi=4.0, T_lo=10.0, T_hi=20.0, log_Ndv_lo=14.0, log_Ndv_hi=15.0)
    grid = build_grid(b, (2, 5, 2), T_kin_fixed=13.0)
    Ts = {g[1] for g in grid}
    assert Ts == {13.0}
    assert len(grid) == 2 * 1 * 2


def test_radius_from_N_dv_identity():
    """N0 = 2*r*n*X should reproduce N_NH3_per_dv*dv when inverted."""
    N_dv, dv, n, X = 1e15, 0.3, 1e4, 1e-8
    radius, N_target = radius_from_N_dv(N_dv, dv, n, X)
    assert N_target == pytest.approx(N_dv * dv)
    recovered_N = 2.0 * radius * n * X
    assert recovered_N == pytest.approx(N_target, rel=1e-9)


def test_task_key_dedup_collision():
    k1 = task_key(9363.292088239417, 16.0, 0.3, 1e-08, 1.5452966455826295e19)
    k2 = task_key(9363.292088239417, 16.0, 0.3, 1e-08, 1.5452966455826295e19)
    assert k1 == k2
    k3 = task_key(9363.292088239417, 16.0, 0.1, 1e-08, 1.5452966455826295e19)  # real dv difference
    assert k1 != k3


def test_boundary_hugging_detects_density_edge():
    b = Bounds(log_n_lo=3.0, log_n_hi=5.0, T_lo=10.0, T_hi=20.0, log_Ndv_lo=14.0, log_Ndv_hi=15.0)
    grid_n = (5, 5, 5)
    survivors = [_row(1000.0, 15.0, 10 ** 14.5, score=0.05)]  # n_H2 at the floor
    assert is_boundary_hugging(survivors, b, grid_n) is True


def test_boundary_hugging_detects_Ndv_edge():
    b = Bounds(log_n_lo=3.0, log_n_hi=5.0, T_lo=10.0, T_hi=20.0, log_Ndv_lo=14.0, log_Ndv_hi=15.0)
    grid_n = (5, 5, 5)
    survivors = [_row(10 ** 4.0, 15.0, 10 ** 15.0, score=0.05)]  # N/dv at the ceiling
    assert is_boundary_hugging(survivors, b, grid_n) is True


def test_boundary_hugging_false_for_interior_survivors():
    b = Bounds(log_n_lo=3.0, log_n_hi=5.0, T_lo=10.0, T_hi=20.0, log_Ndv_lo=14.0, log_Ndv_hi=15.0)
    grid_n = (5, 5, 5)
    survivors = [_row(10 ** 4.0, 15.0, 10 ** 14.5, score=0.05)]
    assert is_boundary_hugging(survivors, b, grid_n) is False


def test_boundary_hugging_ignores_T_axis_when_T_kin_fixed():
    b = Bounds(log_n_lo=3.0, log_n_hi=5.0, T_lo=13.0, T_hi=13.0, log_Ndv_lo=14.0, log_Ndv_hi=15.0)
    grid_n = (5, 1, 5)
    survivors = [_row(10 ** 4.0, 13.0, 10 ** 14.5, score=0.05)]
    assert is_boundary_hugging(survivors, b, grid_n, T_kin_fixed=13.0) is False


def test_expand_bounds_grows_the_hugging_side_only():
    b = Bounds(log_n_lo=3.0, log_n_hi=5.0, T_lo=10.0, T_hi=20.0, log_Ndv_lo=14.0, log_Ndv_hi=15.0)
    grid_n = (5, 5, 5)
    survivors = [_row(1000.0, 15.0, 10 ** 14.5, score=0.05)]  # hugging the LOW n_H2 edge only
    b2 = expand_bounds(b, survivors, grid_n, factor=2.0)
    assert b2.log_n_lo < b.log_n_lo
    assert b2.log_n_hi == pytest.approx(b.log_n_hi)
    assert b2.log_Ndv_lo == pytest.approx(b.log_Ndv_lo)
    assert b2.log_Ndv_hi == pytest.approx(b.log_Ndv_hi)


def test_shrink_bounds_around_covers_all_survivors():
    b = Bounds(log_n_lo=3.0, log_n_hi=6.0, T_lo=10.0, T_hi=20.0, log_Ndv_lo=13.5, log_Ndv_hi=15.5)
    survivors = [_row(10 ** 3.5, 12.0, 10 ** 14.0, score=0.05), _row(10 ** 4.5, 16.0, 10 ** 15.0, score=0.06)]
    b2 = shrink_bounds_around(survivors, b, shrink_factor=0.4)
    assert b2.log_n_lo <= 3.5 <= b2.log_n_hi
    assert b2.log_n_lo <= 4.5 <= b2.log_n_hi
    assert (b2.log_n_hi - b2.log_n_lo) < (b.log_n_hi - b.log_n_lo)


def test_merge_top_k_dedupes_and_sorts():
    a = [_row(1000.0, 15.0, 10 ** 14.5, score=0.2), _row(2000.0, 15.0, 10 ** 14.5, score=0.05)]
    b = [_row(1000.0, 15.0, 10 ** 14.5, score=0.2), _row(3000.0, 15.0, 10 ** 14.5, score=0.01)]
    merged = merge_top_k(a, b, K=10)
    assert len(merged) == 3
    assert merged[0]['score'] == pytest.approx(0.01)
    assert merged[-1]['score'] == pytest.approx(0.2)


def test_merge_top_k_respects_K():
    rows = [_row(1000.0 * i, 15.0, 10 ** 14.5, score=float(i)) for i in range(1, 11)]
    merged = merge_top_k([], rows, K=3)
    assert len(merged) == 3
    assert [r['score'] for r in merged] == [1.0, 2.0, 3.0]
