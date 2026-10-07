"""Named, composable solver profiles for the Magritte non-LTE solve.

A profile is a '+'-joined list of tokens naming *parameter* choices only; no
Magritte source is touched. `legacy` (the default everywhere) reproduces the
settings used to build the gold table, byte for byte.

Tokens (each justified and validated in production/output_speedup_tests/report.md):
  close        sum opacity/emissivity only over lines within
               `max_distance_opacity_contribution` (10) line widths
               (`sum_opacity_emissivity_over_all_lines = False`). This is Magritte's
               upstream default; the vendored copy carries a local edit that flips
               it to True, which sums all 115 lines at every ray step.
  lev<N>       LAMDA file truncated to the lowest N levels (lev36 = 77 of 115 lines),
               `p-nh3@loreau_<N>lev.dat.txt` (lev36 is tracked; others are generated on demand)
  nq<N>        N Gauss-Hermite quadrature points per line (legacy 20), e.g. nq12
  prec<E>      population-change stop threshold 1e-<E> (legacy 1e-6), e.g. prec4
  ngfixed      fixed-order Ng acceleration instead of Magritte's adaptive criterion
  ngmem<N>     Ng memory limit (legacy 9), e.g. ngmem15
  minrel<E>    `min_rel_pop_for_convergence` = 1e-<E> (Magritte default 1e-10)
  interp<X>    `max_interpolation_diff` = X (Magritte default 1.4), e.g. interp2.0
  dist<X>      `max_distance_opacity_contribution` = X line widths (default 10)

Examples: 'legacy', 'close', 'close+lev36+nq12+prec4', 'close+lev36+nq12+prec4+ngfixed+ngmem15'.
"""
import os
import re

LEGACY = 'legacy'
LAMDA_FULL = 'p-nh3@loreau.dat.txt'
LAMDA_36 = 'p-nh3@loreau_36lev.dat.txt'


def lamda_name(n_levels):
    return LAMDA_FULL if n_levels is None else f'p-nh3@loreau_{n_levels}lev.dat.txt'

_DEFAULTS = dict(sum_all_lines=True, lamda=LAMDA_FULL, nquads=20, pop_prec=1.0e-6,
                 ng_adaptive=None, ng_mem=None, min_rel_pop=None, max_interp_diff=None,
                 max_dist=None)

_TOKEN = re.compile(r'^(close|lev|ngfixed|nq|prec|ngmem|minrel|interp|dist)([0-9.]*)$')


def parse(profile):
    """Return the settings dict for a profile string (validated)."""
    s = dict(_DEFAULTS)
    if profile in (None, '', LEGACY):
        return s
    for tok in profile.split('+'):
        m = _TOKEN.match(tok)
        if not m:
            raise ValueError(f"unknown solver-profile token {tok!r} in {profile!r}")
        kind, val = m.groups()
        if kind in ('close', 'ngfixed'):
            if val:
                raise ValueError(f"token {tok!r} takes no value")
        elif not val:
            raise ValueError(f"token {tok!r} needs a value")
        if kind == 'close':
            s['sum_all_lines'] = False
        elif kind == 'lev':
            s['lamda'] = lamda_name(int(val))
        elif kind == 'nq':
            s['nquads'] = int(val)
        elif kind == 'prec':
            s['pop_prec'] = 10.0 ** (-float(val))
        elif kind == 'ngfixed':
            s['ng_adaptive'] = False
        elif kind == 'ngmem':
            s['ng_mem'] = int(val)
        elif kind == 'minrel':
            s['min_rel_pop'] = 10.0 ** (-float(val))
        elif kind == 'interp':
            s['max_interp_diff'] = float(val)
        elif kind == 'dist':
            s['max_dist'] = float(val)
    return s


def canonical(profile):
    """Order-independent name (used in file names and CSV provenance)."""
    if profile in (None, '', LEGACY):
        return LEGACY
    parse(profile)
    return '+'.join(sorted(profile.split('+')))


def file_tag(profile):
    """Model-file suffix: empty for legacy so existing cache names are unchanged."""
    c = canonical(profile)
    return '' if c == LEGACY else '_sp' + c.replace('+', '-')


def truncate_lamda(src, n_levels, out):
    """Write the lowest `n_levels` levels of a LAMDA file with the radiative and collisional
    transitions among them (transition numbers renumbered; level and line order preserved,
    so the (2,2) line stays at index 6)."""
    L = open(src).read().split('\n')
    idx = lambda t: next(k for k, l in enumerate(L) if l.startswith(t))
    a, lv, nr, tr, cp, ct, ctr = (idx(t) for t in (
        '! NUMBER OF ENERGY LEVELS', '! LEVEL |', '! NUMBER OF RADIATIVE', '! TRANS | UP',
        '! NUMBER OF COLLISION PARTNERS', '! NUMBER OF COLLISIONAL', '! TRANS | UP  | LOW | RATE'))
    levels = [l for l in L[lv + 1:nr] if l.strip()]
    rad = [l for l in L[tr + 1:cp] if l.strip()]
    col = [l for l in L[ctr + 1:] if l.strip()]
    keep = lambda l: int(l.split()[1]) <= n_levels
    ren = lambda ls: [re.sub(r'^\s*\d+', f'{k + 1:7d}', l) for k, l in enumerate(ls)]
    rad2, col2 = ren([l for l in rad if keep(l)]), ren([l for l in col if keep(l)])
    open(out, 'w').write('\n'.join(
        L[:a + 1] + [str(n_levels), L[lv]] + levels[:n_levels] + [L[nr], str(len(rad2)), L[tr]] + rad2
        + L[cp:ct] + [L[ct], str(len(col2))] + L[ct + 2:ctr + 1] + col2 + ['']))


def lamda_path(wdir, profile):
    """LAMDA file for the profile inside `wdir`; truncated variants are generated from the
    full file on first use."""
    path = os.path.join(wdir, parse(profile)['lamda'])
    if not os.path.exists(path):
        m = re.match(r'p-nh3@loreau_(\d+)lev\.dat\.txt$', os.path.basename(path))
        if not m:
            raise FileNotFoundError(path)
        truncate_lamda(os.path.join(wdir, LAMDA_FULL), int(m.group(1)), path)
    return path


def apply_before_write(model, profile):
    """Settings that are stored in the model file (nquads must be set before the
    quadrature is built, so this runs first)."""
    s = parse(profile)
    P = model.parameters
    P.set_nquads(s['nquads'])
    P.pop_prec = s['pop_prec']
    P.sum_opacity_emissivity_over_all_lines = s['sum_all_lines']
    return s


def apply_after_read(model, profile):
    """Settings Magritte does NOT persist in the .hdf5 (or may reset on read); applied
    explicitly after every `Model(file)` so legacy never depends on a compile-time default."""
    s = parse(profile)
    P = model.parameters
    P.sum_opacity_emissivity_over_all_lines = s['sum_all_lines']
    P.pop_prec = s['pop_prec']
    if s['max_dist'] is not None:
        P.max_distance_opacity_contribution = s['max_dist']
    if s['ng_adaptive'] is not None:
        P.use_adaptive_Ng_acceleration = s['ng_adaptive']
    if s['ng_mem'] is not None:
        P.Ng_acceleration_mem_limit = s['ng_mem']
    if s['min_rel_pop'] is not None:
        P.min_rel_pop_for_convergence = s['min_rel_pop']
    if s['max_interp_diff'] is not None:
        P.max_interpolation_diff = s['max_interp_diff']
    return s
