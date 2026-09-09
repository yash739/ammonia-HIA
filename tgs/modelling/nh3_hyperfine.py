"""
NH3 inversion-line hyperfine structure: component identities, velocity offsets,
and the machinery to assign fitted Gaussians to physical components.

WHY THIS MODULE EXISTS
----------------------
`nh3_NLTE_analysis.analyse_spectra` used to assign hyperfine amplitudes by
their *position index* in the curve_fit parameter vector (`amps11[0]` -> A_10,
`amps11[1]` -> A_21, ...). That is wrong in two independent ways:

  1. It mislabelled the outer pair. On the velocity axis the code produced, the
     most-negative component is F1 = 0->1 but was labelled `A_10`, and the
     most-positive is F1 = 1->0 but was labelled `A_01`. The hyperfine
     *anomaly* is precisely the asymmetry between those two lines, so the swap
     inverts the quantity being measured.
  2. It is fragile. With unbounded Gaussian centres, components can drift,
     cross or collapse onto one another under noise, silently scrambling the
     index -> component mapping with no error raised.

Both are fixed here by identifying components from their fitted *centre
velocity* against the known offsets below.

PROVENANCE OF THE OFFSETS
-------------------------
Derived from the LAMDA collision file's own F1-resolved level list
(`p-nh3@loreau.dat.txt`, level labels `J_K_sym_F1`), by
`derive_offsets_from_lamda()`. The tabulated constants below are that
function's output, kept as literals so there is no runtime file dependency;
`test_nh3_hyperfine.py` re-derives them from the file and asserts agreement, so
the two cannot silently drift apart.

They are independently corroborated twice over:
  - Stutzki & Winnewisser (1985) Fig. 1 gives the (1,1) velocity scale as
    -19.50, -7.62, 0, +7.61, +19.49 km/s.
  - Camarata, Jackson & Chambers (2015, ApJ 806, 74) Sect. 2 explicitly define
    F1 = 0->1 as "right-outer", F1 = 1->0 as "left-outer", F1 = 1->2 as
    "left-inner" and F1 = 2->1 as "right-inner" -- matching the table below
    component for component.

VELOCITY CONVENTION
-------------------
Everything here uses the standard radio/LSR convention

    v = c (nu_rest - nu) / nu_rest

so positive velocity is redshifted (lower frequency). Note the consequence that
trips people up: F1 = 1->0 sits at *higher* frequency than the main line and
therefore at *negative* velocity, while F1 = 0->1 is at lower frequency and
positive velocity -- the opposite of what the label ordering suggests.
"""

import os
import numpy as np

C_KMS = 2.99792458e5

# Rest frequencies of the inversion transitions [Hz].
FREQ_HZ = {
    '1,1': 23694.4955e6,
    '2,2': 23722.6333e6,
}

# Component keys, in the amplitude-name convention this pipeline already uses
# downstream (A_10, A_21, A_MAIN, A_12, A_01). The numbers are the F1 quantum
# numbers of the transition, F1_upper -> F1_lower.
#
# offsets are km/s in the radio convention (see module docstring).
# `camarata` is the left/right naming of Camarata et al. (2015), retained
# because it makes the sign convention unambiguous when comparing to their
# published anomaly ratios.
NH3_11_COMPONENTS = (
    # key,      F1 transition, v_offset [km/s], group,   Camarata label
    ('A_10',    (1, 0),        -19.497,         'outer', 'left-outer'),
    ('A_12',    (1, 2),         -7.590,         'inner', 'left-inner'),
    ('A_MAIN',  None,            0.000,         'main',  'main'),
    ('A_21',    (2, 1),         +7.598,         'inner', 'right-inner'),
    ('A_01',    (0, 1),        +19.492,         'outer', 'right-outer'),
)

# (2,2) hyperfine offsets [km/s], same convention, same derivation. The (2,2)
# satellites are far weaker and sit much further out in velocity than the (1,1)
# ones. This pipeline only uses the (2,2) main-line brightness, but the offsets
# are needed to seed and bound its five-Gaussian fit.
NH3_22_COMPONENTS = (
    ('A22_10',  (2, 1),        -26.019),   # left-outer
    ('A22_12',  (2, 3),        -16.358),   # left-inner
    ('A22_MAIN', None,           0.000),
    ('A22_21',  (3, 2),        +16.370),   # right-inner
    ('A22_01',  (1, 2),        +26.022),   # right-outer
)

# Keys in ascending velocity order -- the order a left-to-right spectrum shows.
NH3_11_KEYS_BY_VELOCITY = tuple(k for k, _, _, _, _ in NH3_11_COMPONENTS)
NH3_11_OFFSETS_KMS = np.array([v for _, _, v, _, _ in NH3_11_COMPONENTS])

NH3_22_KEYS_BY_VELOCITY = tuple(k for k, _, _ in NH3_22_COMPONENTS)
NH3_22_OFFSETS_KMS = np.array([v for _, _, v in NH3_22_COMPONENTS])

# The LTE optically-thin satellite/main intensity ratios, for reference and for
# regression tests: each inner satellite group is ~0.28 of the main and each
# outer ~0.22 (Ho & Townes 1983; quoted as 26% and 22% by Wu et al. 2024, whose
# values fold in the finer hyperfine sums).
LTE_RATIO_INNER = 0.278
LTE_RATIO_OUTER = 0.222


def freq_to_radio_velocity_kms(freq_hz, freq_rest_hz):
    """Standard radio/LSR convention: v = c (nu_rest - nu) / nu_rest [km/s].

    Positive velocity is redshifted, i.e. LOWER frequency. This is the opposite
    sign to `c (nu - nu_rest) / nu_rest`, which is what this pipeline used
    before and which left the FITS `CTYPE1 = 'VELO-LSR'` header misdescribing
    its own data.
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    return C_KMS * (freq_rest_hz - freq_hz) / freq_rest_hz


def derive_offsets_from_lamda(lamda_path, J=1, K=1, freq_rest_hz=None, tol_kms=0.5):
    """Re-derive the hyperfine velocity offsets from the LAMDA file itself.

    Returns a list of (F1_upper, F1_lower, freq_MHz, v_kms) for the radiative
    transitions between the two inversion sub-states of (J,K), sorted by
    velocity. Used by the unit tests to verify the literal tables above; not
    called at import time, so the module has no runtime file dependency.
    """
    if freq_rest_hz is None:
        freq_rest_hz = FREQ_HZ[f'{J},{K}']
    with open(lamda_path) as f:
        lines = f.read().split('\n')

    # Level block: "! LEVEL | ENERGY | WEIGHT | J_K_sym_F"
    i_lev = next(i for i, l in enumerate(lines) if 'LEVEL' in l and '|' in l)
    i_ntr = next(i for i, l in enumerate(lines) if 'NUMBER OF RADIATIVE' in l)
    levels = {}
    for line in lines[i_lev + 1:i_ntr]:
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit():
            levels[int(parts[0])] = parts[3]

    # Upper inversion sub-state has sym = -1, lower has sym = +1.
    upper = {i: lab for i, lab in levels.items() if lab.startswith(f'{J}_{K}_-1_')}
    lower = {i: lab for i, lab in levels.items() if lab.startswith(f'{J}_{K}_1_')}

    i_tr = next(i for i, l in enumerate(lines) if 'TRANS' in l and 'A_ul' in l)
    n_tr = int(lines[i_ntr + 1].split()[0])

    out = []
    for line in lines[i_tr + 1:i_tr + 1 + n_tr]:
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            u, lo, freq_mhz = int(parts[1]), int(parts[2]), float(parts[4]) * 1e3
        except ValueError:
            continue
        if u in upper and lo in lower:
            f1_u = int(upper[u].rsplit('_', 1)[1])
            f1_l = int(lower[lo].rsplit('_', 1)[1])
            v = freq_to_radio_velocity_kms(freq_mhz * 1e6, freq_rest_hz)
            out.append((f1_u, f1_l, freq_mhz, float(v)))
    return sorted(out, key=lambda r: r[3])


def identify_components(centres_kms, offsets_kms=None, keys=None, max_sep_kms=6.0):
    """Map fitted Gaussian centres to hyperfine components by nearest offset.

    This replaces assignment-by-parameter-index. Returns a dict
    {component_key: index_into_centres}. Raises ValueError if the assignment is
    not one-to-one or if any component is further than `max_sep_kms` from its
    nearest fitted centre -- a fit that has collapsed two components onto one
    peak, or drifted a component out of its window, is a *failure* and must not
    be silently returned as if it were a measurement.

    `max_sep_kms` default of 6 km/s is comfortably below the 7.6 km/s spacing
    between the main line and the inner satellites, so a mis-assignment cannot
    masquerade as a valid one.
    """
    if offsets_kms is None:
        offsets_kms = NH3_11_OFFSETS_KMS
    if keys is None:
        keys = NH3_11_KEYS_BY_VELOCITY
    centres = np.asarray(centres_kms, dtype=float)
    if len(centres) != len(offsets_kms):
        raise ValueError(f"expected {len(offsets_kms)} centres, got {len(centres)}")

    # Greedy nearest assignment is sufficient and predictable here because the
    # components are well separated relative to the fit windows; verify
    # one-to-one afterwards rather than trusting it.
    assignment = {}
    used = set()
    order = np.argsort([np.min(np.abs(centres - off)) for off in offsets_kms])
    for oi in order:
        d = np.abs(centres - offsets_kms[oi])
        for ci in np.argsort(d):
            if ci not in used:
                if d[ci] > max_sep_kms:
                    raise ValueError(
                        f"component {keys[oi]} (expected {offsets_kms[oi]:+.2f} km/s) has no "
                        f"fitted centre within {max_sep_kms} km/s; nearest is "
                        f"{centres[ci]:+.2f} km/s. The fit has drifted or collapsed."
                    )
                assignment[keys[oi]] = int(ci)
                used.add(ci)
                break
    if len(assignment) != len(keys):
        raise ValueError("hyperfine component assignment is not one-to-one")
    return assignment


def estimate_v_sys_kms(velos_kms, tmb):
    """Crude systemic-velocity estimate: the velocity of the brightest channel.

    For an NH3 (1,1) spectrum the main line is by far the strongest feature, so
    the global maximum locates it reliably. Used only to *centre* the fit
    windows below, not as a fitted quantity -- the fit still refines each
    component's centre within its own window.
    """
    velos_kms = np.asarray(velos_kms, dtype=float)
    return float(velos_kms[int(np.argmax(np.asarray(tmb)))])


# Half-width of each component's centre window [km/s]. Must stay below half the
# smallest component separation or two windows would overlap and the fit could
# swap components -- exactly the failure mode this module exists to prevent.
# Smallest (1,1) separation is main <-> inner at 7.59 km/s, so the ceiling is
# 3.795; 3.0 leaves a 1.59 km/s guard band between adjacent windows.
DEFAULT_CENTRE_WINDOW_KMS = 3.0


def max_safe_window_kms(offsets_kms=None):
    """Largest non-overlapping half-window for a given offset table."""
    if offsets_kms is None:
        offsets_kms = NH3_11_OFFSETS_KMS
    seps = np.diff(np.sort(np.asarray(offsets_kms, dtype=float)))
    return float(np.min(seps) / 2.0)


def initial_centres_and_bounds(offsets_kms=None, v_sys_kms=0.0,
                                 window_kms=DEFAULT_CENTRE_WINDOW_KMS):
    """Seed centres and per-centre bounds for the five-Gaussian fit.

    Returns (p0_centres, lower, upper), each a length-5 array in km/s.

    Centres are seeded at the true hyperfine offsets shifted by `v_sys_kms`
    (see `estimate_v_sys_kms`) rather than spread uniformly across the window,
    and each is bounded to +/- `window_kms` about its seed. This is the
    "slightly free" behaviour: free enough to track the observed hyperfine
    centres exactly, tight enough that two components cannot swap or collapse
    onto the same peak under noise.

    Raises ValueError if `window_kms` is wide enough for adjacent windows to
    overlap, since that would silently reopen the swap failure mode.
    """
    if offsets_kms is None:
        offsets_kms = NH3_11_OFFSETS_KMS
    ceiling = max_safe_window_kms(offsets_kms)
    if window_kms >= ceiling:
        raise ValueError(
            f"window_kms={window_kms} would let adjacent centre windows overlap "
            f"(must be < {ceiling:.3f} km/s for this offset table); components "
            f"could then swap, which is the bug this bounding exists to prevent."
        )
    centres = np.asarray(offsets_kms, dtype=float) + v_sys_kms
    return centres, centres - window_kms, centres + window_kms
