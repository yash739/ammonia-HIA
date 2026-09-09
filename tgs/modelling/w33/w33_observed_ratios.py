"""Observed NH3 hyperfine ratio vectors for W33, from Tursun et al. (2022).

Combines two sources:
  - `observed_data/W33_line_heights.csv`: digitized (1,1) and (2,2) hyperfine
    satellite peak heights, read directly off Figs. 2-5 of Tursun et al. (2022)
    at the central (0,0) position of each source, for the five NON-absorbing
    sources (W33 Main is excluded: it shows NH3 absorption against its own H II
    region continuum in (1,1)/(2,2)/(4,4)/(5,5)/(6,6), a different physical
    regime our emission-only forward model does not address -- see
    params.LINES['W33_Main'] and Tursun et al.'s Table 3).
  - `params.LINES`: the paper's own Table 4 fitted main-line T_mb (and (2,1)
    peaks where detected), reused as the denominator for consistency with the
    rest of this pipeline and cross-checked against the digitized main peak
    below (they agree to within ~2-9%, consistent with digitization precision;
    see test_w33_observed_ratios.py).

RATIO CONVENTION -- must match nh3_hyperfine.py exactly
--------------------------------------------------------
"Blue"/"Red" in the CSV means blueshifted/redshifted (radio convention,
v = c(nu_rest-nu)/nu_rest), NOT which side of the panel a peak visually sits on
read left-to-right in frequency. Blueshifted = negative velocity = HIGHER
frequency:
    Outer_Blue_Sat -> F1 = 1->0  -> A_10  (at -19.5 km/s)
    Inner_Blue_Sat -> F1 = 1->2  -> A_12  (at  -7.6 km/s)
    Inner_Red_Sat  -> F1 = 2->1  -> A_21  (at  +7.6 km/s)
    Outer_Red_Sat  -> F1 = 0->1  -> A_01  (at +19.5 km/s)
This is the same mapping nh3_hyperfine.NH3_11_COMPONENTS uses, independently
confirmed there via the LAMDA level list and Camarata et al. (2015).

NON-DETECTIONS
--------------
Where the CSV says "non-detection", no peak was distinguishable above the
noise floor at that position. These are NOT dropped or silently treated as
zero: `observed_ratio_vector` returns them as (value=None, upper_limit=3*rms)
and the caller must decide how to handle a partially-specified ratio vector
(e.g. exclude that ratio from a chi^2 sum, or use the upper limit as a one-
sided constraint). See ANOMALY_SENSE_FLAGS below for a documented case where
this matters.

ANOMALY SENSE -- flagged, not silently accepted
-------------------------------------------------
The published/expected sense is R_01 > R_10 (outer) and R_12 > R_21 (inner).
Checked directly against the digitized peaks: the outer pair shows the
expected sense for all 5 sources. The INNER pair is reversed (R_21 > R_12,
weakly) for W33 A1 and W33 B1, and is a near-exact tie for W33 Main1. Given
the peak differences involved are ~0.05-0.15 K, comparable to plausible
digitization precision, these are flagged rather than either corrected or
hidden -- do not read them as three additional detections of the CE/expansion
mechanism without checking against a re-digitization or the archival spectra.
"""

import os
import csv

import params as p

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'observed_data', 'W33_line_heights.csv')

# CSV transition label -> (LINES key, our component key or None for the main line)
_TRANS_MAP = {'(1+1)': '1,1', '(2+2)': '2,2', '(2+1)': '2,1', '(3+2)': '3,2'}
_SAT_COLS = {'Outer_Blue_Sat_K': 'A_10', 'Inner_Blue_Sat_K': 'A_12',
             'Inner_Red_Sat_K': 'A_21', 'Outer_Red_Sat_K': 'A_01'}

# Sources with a documented anomaly-sense discrepancy in the digitized inner
# pair -- see module docstring. Populated by _load(); read-only after import.
ANOMALY_SENSE_FLAGS = {}


def _parse_val(s):
    s = s.strip()
    return None if s.lower() == 'non-detection' else float(s)


def _load():
    table = {}
    with open(CSV_PATH, newline='') as f:
        for row in csv.DictReader(f):
            src = row['Source'].replace(' ', '_')  # 'W33 A' -> 'W33_A'
            trans = _TRANS_MAP[row['Transition']]
            table.setdefault(src, {})[trans] = dict(
                rms=float(row['Noise_Floor_rms_K']),
                main=_parse_val(row['Main_Line_K']),
                **{comp: _parse_val(row[col]) for col, comp in _SAT_COLS.items()},
            )
    return table


DIGITIZED = _load()

# W33 Main excluded -- see module docstring (absorption source).
W33_SOURCES = ('W33_A', 'W33_B', 'W33_Main1', 'W33_A1', 'W33_B1')


# --------------------------------------------------------------------------- #
# L1: model-applicability gate (quadrant test)
# --------------------------------------------------------------------------- #
# Zhou et al. (2020) / Wu et al. (2024) convention: HIA_IS/HIA_OS are the
# redshifted/blueshifted intensity ratios of the inner/outer satellite pairs.
# Static hyperfine selective trapping predicts HIA_IS < 1, HIA_OS > 1
# (quadrant II) -- the ONLY quadrant a static constant-density sphere can
# ever reproduce. Quadrant III (both < 1) is infall, quadrant I (both > 1) is
# expansion, and quadrant IV (HIA_IS > 1, HIA_OS < 1) is forbidden by both
# mechanisms. A source outside quadrant II cannot be fit by this pipeline's
# model at ANY parameters -- this is therefore a gate to run before spending
# any Magritte time on a source, not a diagnostic to run after.
QUADRANT_LABELS = {
    'II': 'static trapping (fittable by a constant-density sphere)',
    'III': 'infall signature (NOT fittable by a static sphere)',
    'I': 'expansion signature (NOT fittable by a static sphere)',
    'IV': 'forbidden by both mechanisms (data inconsistency -- check the source)',
}


def quadrant_gate(source):
    """Classify a source's (HIA_IS, HIA_OS) quadrant directly from the
    digitized ratios in observed_data/W33_line_heights.csv, taken as given.

    HIA_IS = R(2->1)/R(1->2) = obs['R_21_MAIN']/obs['R_12_MAIN']
    HIA_OS = R(0->1)/R(1->0) = obs['R_01_MAIN']/obs['R_10_MAIN']
    (redshifted/blueshifted, matching Zhou et al. 2020's own convention.)

    Returns dict(HIA_IS, HIA_OS, quadrant, label).
    """
    obs, err, _ = observed_ratio_vector(source)
    HIA_IS = obs['R_21_MAIN'] / obs['R_12_MAIN']
    HIA_OS = obs['R_01_MAIN'] / obs['R_10_MAIN']
    if HIA_IS < 1 and HIA_OS > 1:
        q = 'II'
    elif HIA_IS < 1 and HIA_OS < 1:
        q = 'III'
    elif HIA_IS > 1 and HIA_OS > 1:
        q = 'I'
    else:
        q = 'IV'
    return dict(source=source, HIA_IS=HIA_IS, HIA_OS=HIA_OS,
                quadrant=q, label=QUADRANT_LABELS[q])


def observed_ratio_vector(source, sigma_frac_floor=0.10):
    """Return (obs, err, meta) for one source's (1,1)/(2,2)/(2,1) ratios.

    obs, err: dicts with keys R_01_MAIN, R_10_MAIN, R_21_MAIN, R_12_MAIN,
    R_22_MAIN, R_21_11 -- a value is None (not a number) where the underlying
    peak was a non-detection; err uses `sigma_frac_floor` of the observed
    value where no fit uncertainty is available (digitized peaks carry no
    formal error, only a noise floor), consistent with scoring.weighted_chi2's
    own default floor.

    meta: dict with the two main-line readings (digitized vs. Table 4 fitted)
    and their fractional disagreement, plus the resolved anomaly-sense flags.
    """
    d11 = DIGITIZED[source]['1,1']
    d22 = DIGITIZED[source]['2,2']
    d21 = DIGITIZED[source].get('2,1')
    line_key = source.replace('W33_', 'W33_')
    tab4 = p.LINES[line_key]

    main_dig = d11['main']
    main_tab4 = tab4['1,1']['T_mb']
    main = main_tab4  # prefer the paper's own fitted value as the denominator

    obs, err = {}, {}
    for comp, rkey in (('A_10', 'R_10_MAIN'), ('A_12', 'R_12_MAIN'),
                       ('A_21', 'R_21_MAIN'), ('A_01', 'R_01_MAIN')):
        v = d11[comp]
        if v is None:
            obs[rkey] = None
            err[rkey] = 3 * d11['rms'] / main  # 3-sigma upper limit on the ratio
        else:
            obs[rkey] = v / main
            err[rkey] = max(sigma_frac_floor * v, d11['rms']) / main

    main22_tab4 = tab4.get('2,2', {}).get('T_mb')
    if main22_tab4 is not None:
        obs['R_22_MAIN'] = main22_tab4 / main
        err['R_22_MAIN'] = max(sigma_frac_floor * main22_tab4, d22['rms']) / main
    else:
        obs['R_22_MAIN'] = None
        err['R_22_MAIN'] = None

    if '2,1' in tab4:
        obs['R_21_11'] = tab4['2,1']['T_mb'] / main
        err['R_21_11'] = max(sigma_frac_floor * tab4['2,1']['T_mb'],
                              (d21['rms'] if d21 else 0.04)) / main
    else:
        obs['R_21_11'] = None
        rms21 = d21['rms'] if d21 else 0.04
        err['R_21_11'] = 3 * rms21 / main  # upper limit; (2,1) not detected

    outer_sense_ok = (d11['A_01'] is not None and d11['A_10'] is not None
                       and d11['A_01'] > d11['A_10'])
    inner_sense_ok = (d11['A_12'] is not None and d11['A_21'] is not None
                       and d11['A_12'] > d11['A_21'])
    meta = dict(main_digitized=main_dig, main_table4=main_tab4,
                main_disagreement_frac=(abs(main_dig - main_tab4) / main_tab4
                                         if main_dig is not None else None),
                outer_sense_ok=outer_sense_ok, inner_sense_ok=inner_sense_ok)
    ANOMALY_SENSE_FLAGS[source] = meta
    return obs, err, meta
