"""Solver profiles: parameter-only speed-ups of the Magritte non-LTE solve."""
import os
import numpy as np
import pytest

from nh3hia import paths
from nh3hia.model3d import solver_profiles as sp
from nh3hia.lut.build import coarse_to_fine


def _read_lamda(path):
    L = open(path).read().split('\n')
    k = {s: next(i for i, l in enumerate(L) if l.startswith(s))
         for s in ('! LEVEL |', '! NUMBER OF RADIATIVE', '! TRANS | UP  | LOW | A_ul',
                   '! NUMBER OF COLLISION PARTNERS', '! TRANS | UP  | LOW | RATE')}
    levels = [l.split() for l in L[k['! LEVEL |'] + 1:k['! NUMBER OF RADIATIVE']] if l.strip()]
    rad = [l.split() for l in L[k['! TRANS | UP  | LOW | A_ul'] + 1:k['! NUMBER OF COLLISION PARTNERS']] if l.strip()]
    col = [l.split() for l in L[k['! TRANS | UP  | LOW | RATE'] + 1:] if l.strip()]
    return levels, rad, col


def test_legacy_is_the_original_configuration():
    s = sp.parse('legacy')
    assert s['sum_all_lines'] is True and s['nquads'] == 20 and s['pop_prec'] == 1e-6
    assert s['lamda'] == sp.LAMDA_FULL
    assert sp.file_tag('legacy') == '' and sp.file_tag(None) == ''


def test_tokens_and_canonical_order():
    s = sp.parse('close+lev36+nq12+prec4+ngfixed+ngmem15')
    assert s['sum_all_lines'] is False and s['lamda'] == sp.LAMDA_36
    assert s['nquads'] == 12 and s['pop_prec'] == pytest.approx(1e-4)
    assert s['ng_adaptive'] is False and s['ng_mem'] == 15
    assert sp.canonical('nq12+close') == sp.canonical('close+nq12')
    assert sp.file_tag('close') == '_spclose'


@pytest.mark.parametrize('bad', ['fast', 'close+nq', 'close5', 'lev36x', 'prec'])
def test_bad_tokens_rejected(bad):
    with pytest.raises(ValueError):
        sp.parse(bad)


def test_profile_sets_only_known_magritte_parameters():
    import magritte.core as mc
    m = mc.Model()
    sp.apply_after_read(m, 'close+prec4+ngfixed+ngmem15+minrel8+interp2.0+dist12')
    P = m.parameters
    assert P.sum_opacity_emissivity_over_all_lines is False
    assert P.pop_prec == pytest.approx(1e-4) and P.use_adaptive_Ng_acceleration is False
    assert P.Ng_acceleration_mem_limit == 15 and P.max_distance_opacity_contribution == 12
    sp.apply_after_read(m, 'legacy')
    assert P.sum_opacity_emissivity_over_all_lines is True and P.pop_prec == pytest.approx(1e-6)


def test_truncated_lamda_is_a_strict_subset_of_the_full_file():
    full = _read_lamda(os.path.join(paths.TGS, sp.LAMDA_FULL))
    t36 = _read_lamda(os.path.join(paths.TGS, sp.LAMDA_36))
    assert len(t36[0]) == 36 and len(t36[1]) == 77
    assert [l[3] for l in t36[0]] == [l[3] for l in full[0][:36]]     # same levels, same order
    kept = [l for l in full[1] if int(l[1]) <= 36]                    # radiative: both levels kept
    assert [(l[1], l[2], l[3], l[4]) for l in t36[1]] == [(l[1], l[2], l[3], l[4]) for l in kept]
    assert float(t36[1][6][4]) == pytest.approx(float(full[1][6][4]))  # (2,2) stays at line index 6
    kc = {(l[1], l[2]): l[3:] for l in full[2] if int(l[1]) <= 36}
    assert len(t36[2]) == len(kc)
    assert all(kc[(l[1], l[2])] == l[3:] for l in t36[2])             # collision rates untouched


def test_coarse_to_fine_order_is_a_permutation_with_coarse_lattice_first():
    pts = [(a, t, c) for a in range(5) for t in range(4) for c in range(3)]
    o = coarse_to_fine(pts)
    assert sorted(o) == sorted(pts) and len(set(o)) == len(pts)
    coarse = [p for p in pts if all(x % 2 == 0 or x == m for x, m in zip(p, (4, 3, 2)))]
    assert set(o[:len(coarse)]) == set(coarse)
    assert coarse_to_fine(pts) == o                                    # deterministic
