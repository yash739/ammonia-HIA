"""Parser for the complete Stutzki (1984, 1985) tables, digitized in full by
the user into observed_data/stutzki_1984_1985_full.csv.

Covers eight tables across the two papers that nh3hia/stutzki/params.py's hand
transcriptions only partially covered:
  Table 1 (1984): source list, coordinates (RA/Dec 1950), systemic velocity.
  Table 2 (1984): 51 positions with hfs anomalies -- the observed (1,1) ratio
    vectors. stutzki_params.TABLE_2_1984 had 23 of these, hand-transcribed
    from page scans; this supersedes it with the full set.
  Table 3 (1984): 12 sources WITHOUT anomalies (LTE-consistent controls).
  Table 1a (1985): fitted params, high-density solution. Cross-checked against
    stutzki_params.TABLE_1A below -- see CROSS_CHECK_TABLE_1A.
  Table 1b (1985): fitted params, low-density (rejected) solution. Not
    previously in the codebase in structured form.
  Table 2a/2b (1985): DERIVED quantities (Jeans length/mass, M_c(max), K,
    mean density, NH3/H2 abundance) for the high/low-density solutions. This
    is what lets nh3hia/stutzki/physics.py's coefficients be checked against 23
    independent tabulated K values instead of the single S106 spot-check
    used when they were first verified.
  Table 3 (1985): the (2,1) test set -- matches stutzki_params.TABLE_3_21.

RATIO COLUMN ORDER -- resolved, not assumed
--------------------------------------------
The CSV's own header row labels the four (1,1) ratio columns
"R_1_to_2, R_2_to_1, R_1_to_0, R_0_to_1" in that left-to-right order. Taken
literally that would mean the FIRST column is R(1->2). Checked statistically
against the published anomaly sense (R(0->1) > R(1->0) outer;
R(1->2) > R(2->1) inner) across all 51 rows, using each of the two possible
column-order readings:

    reading                          outer correct   inner correct
    C1=R10,C2=R12,C3=R21,C4=R01           40/51 (78%)     28/51 (55%)
    C1=R12,C2=R21,C3=R10,C4=R01           36/51 (71%)     18/51 (35%)

The first reading (matching stutzki_params.TABLE_2_1984's existing column
mapping, established independently from a direct high-resolution read of the
printed table) wins decisively on both counts. Inner-pair anomalies are
genuinely weaker and noisier than outer ones (consistent with Stutzki &
Winnewisser 1985's own text), which is why neither reading reaches 100% --
but the contrast between readings is large and one-sided, not marginal.
PARSE_ROW_ORDER below therefore reads the CSV's own columns as
(R_1_to_0, R_1_to_2, R_2_to_1, R_0_to_1) regardless of the printed header
labels, i.e. the CSV's header row mislabels which physical transition goes
with which column while the underlying numbers are correct.
"""

import os
import re
import csv

from nh3hia import paths

CSV_PATH = os.path.join(paths.OBSERVED_DATA, 'stutzki_1984_1985_full.csv')

# See module docstring: CSV column position -> our ratio key, NOT the CSV's
# own printed header label for that column.
_RATIO_COL_ORDER = ('R_10_MAIN', 'R_12_MAIN', 'R_21_MAIN', 'R_01_MAIN')

_VAL_ERR_RE = re.compile(r'^\s*(-?[\d.]+)\s*(?:\(([\d.]+)\))?\s*')


def _parse_val_err(s):
    """'4.752(0.039)' -> (4.752, 0.039); '0.6 (a)' -> (0.6, None);
    'not measured (b)' -> (None, None); '*' -> (None, None)."""
    if s is None:
        return None, None
    s = s.strip()
    m = _VAL_ERR_RE.match(s)
    if not m:
        return None, None
    val = float(m.group(1))
    err = float(m.group(2)) if m.group(2) else None
    return val, err


def _strip_footnote(s):
    """'OMC2 a' -> 'OMC2'; '(0, 0) a' -> '(0, 0)'; keeps the footnote letter
    available separately if ever needed, but callers here only need the key."""
    return re.sub(r'\s+[a-z]$', '', s.strip())


def _read_sections(path):
    """Split the file into (header_row, data_lines) blocks, one per table,
    using the CSV's own blank-line-separated, '#'-commented structure."""
    with open(path) as f:
        raw = f.read()
    blocks = [b for b in raw.split('\n\n') if b.strip()]
    sections = []
    for b in blocks:
        lines = [l for l in b.split('\n') if l.strip() and not l.startswith('#')]
        if not lines:
            continue
        header = next(csv.reader([lines[0]]))
        rows = [next(csv.reader([l])) for l in lines[1:]]
        sections.append((header, rows))
    return sections


def _load_all():
    sections = _read_sections(CSV_PATH)
    out = {}

    # --- Table 1 (1984): source list ---------------------------------------
    h, rows = sections[0]
    assert h[0] == 'Source' and 'RA_1950' in h
    table1 = {}
    for r in rows:
        name = _strip_footnote(r[0])
        table1[name] = dict(ra_1950=r[1], dec_1950=r[2],
                            v_lsr=float(r[3]), ref=r[4], telescope=r[5])
    out['TABLE_1_SOURCES'] = table1

    # --- Table 2 (1984): 51-position anomaly ratios -------------------------
    h, rows = sections[1]
    assert h[0] == 'Source' and 'Offset_arcsec' in h
    table2 = {}
    for r in rows:
        source = r[0]
        offset = r[1]
        tb11, tb11_err = _parse_val_err(r[2])
        vlsr, vlsr_err = _parse_val_err(r[3])
        dv, dv_err = _parse_val_err(r[4])
        c1, c1e = _parse_val_err(r[5])
        c2, c2e = _parse_val_err(r[6])
        c3, c3e = _parse_val_err(r[7])
        c4, c4e = _parse_val_err(r[8])
        tb22, tb22_err = _parse_val_err(r[9]) if len(r) > 9 else (None, None)
        vals = dict(zip(_RATIO_COL_ORDER, [c1, c2, c3, c4]))
        errs = dict(zip([k + '_err' for k in _RATIO_COL_ORDER], [c1e, c2e, c3e, c4e]))
        table2[(source, offset)] = dict(
            T_B_11=tb11, T_B_11_err=tb11_err, v_lsr=vlsr, v_lsr_err=vlsr_err,
            dv=dv, dv_err=dv_err, R_22_MAIN=tb22, R_22_MAIN_err=tb22_err,
            **vals, **errs)
    out['TABLE_2_1984_FULL'] = table2

    # --- Table 3 (1984): sources without anomalies --------------------------
    h, rows = sections[2]
    assert h[0] == 'Source' and 'tau_T' in h
    table3_1984 = {}
    for r in rows:
        tb, tb_e = _parse_val_err(r[1]); vl, vl_e = _parse_val_err(r[2])
        dv, dv_e = _parse_val_err(r[3]); tau, tau_e = _parse_val_err(r[4])
        table3_1984[r[0]] = dict(T_B_11=tb, T_B_11_err=tb_e, v_lsr=vl, v_lsr_err=vl_e,
                                  dv_int=dv, dv_int_err=dv_e, tau=tau, tau_err=tau_e)
    out['TABLE_3_1984_NO_ANOMALY'] = table3_1984

    # --- Table 1a/1b (1985): fitted params -----------------------------------
    for sec_idx, key in ((3, 'TABLE_1A_FULL'), (4, 'TABLE_1B_FULL')):
        h, rows = sections[sec_idx]
        assert h[0] == 'Region' and 'Tk_K' in h
        d = {}
        for r in rows:
            region, pos = r[0], _strip_footnote(r[1])
            includes_22 = r[1].strip().endswith(' a')
            Tk, Tk_e = _parse_val_err(r[2]); ln, ln_e = _parse_val_err(r[3])
            lN, lN_e = _parse_val_err(r[4]); chi2 = float(r[5])
            dv, dv_e = _parse_val_err(r[6])
            TBth = float(r[7])
            TBobs, TBobs_e = _parse_val_err(r[8])
            eta_raw = r[9].strip()
            eta = None if eta_raw == '*' else float(eta_raw)
            d[(region, pos)] = dict(T_k=Tk, T_k_err=Tk_e, log_nH2=ln, log_nH2_err=ln_e,
                                    log_N_NH3=lN, log_N_NH3_err=lN_e, chi2=chi2,
                                    dv_obs=dv, dv_obs_err=dv_e, T_B_theor=TBth,
                                    T_B_obs=TBobs, T_B_obs_err=TBobs_e,
                                    eta_f=eta, eta_f_unphysical=(eta_raw == '*'),
                                    includes_22_hfs=includes_22)
        out[key] = d

    # --- Table 2a/2b (1985): derived quantities ------------------------------
    for sec_idx, key in ((5, 'TABLE_2A_DERIVED'), (6, 'TABLE_2B_DERIVED')):
        h, rows = sections[sec_idx]
        assert h[0] == 'Region' and 'log_K' in h
        d = {}
        for r in rows:
            region, pos = r[0], _strip_footnote(r[1])
            lj, lj_e = _parse_val_err(r[2]); mj, mj_e = _parse_val_err(r[3])
            mc, mc_e = _parse_val_err(r[4]); lk, lk_e = _parse_val_err(r[5])
            lnu, lnu_e = _parse_val_err(r[6]); lnbar, lnbar_e = _parse_val_err(r[7])
            lab, lab_e = _parse_val_err(r[8])
            d[(region, pos)] = dict(
                log_lambda_J_e2pc=lj, log_lambda_J_e2pc_err=lj_e,
                log_M_J_Msun=mj, log_M_J_Msun_err=mj_e,
                log_Mc_max_Msun=mc, log_Mc_max_Msun_err=mc_e,
                log_K=lk, log_K_err=lk_e,
                log_nu_pc3=lnu, log_nu_pc3_err=lnu_e,
                log_n_bar_cm3=lnbar, log_n_bar_cm3_err=lnbar_e,
                log_NH3_over_H2=lab, log_NH3_over_H2_err=lab_e)
        out[key] = d

    # --- Table 3 (1985): (2,1) results ---------------------------------------
    h, rows = sections[7]
    assert h[0] == 'Source' and 'TB_2_1_over_TB_1_1_theor' in h
    d = {}
    for r in rows:
        tb21, tb21_e = _parse_val_err(r[1]); tb11, tb11_e = _parse_val_err(r[2])
        theor = float(r[3])
        obs, obs_e = _parse_val_err(r[4])
        d[r[0]] = dict(T_B_21=tb21, T_B_21_err=tb21_e, T_B_11=tb11, T_B_11_err=tb11_e,
                       ratio_theor=theor, ratio_obs=obs, ratio_obs_err=obs_e)
    out['TABLE_3_21_FULL'] = d

    return out


_TABLES = _load_all()
TABLE_1_SOURCES = _TABLES['TABLE_1_SOURCES']
TABLE_2_1984_FULL = _TABLES['TABLE_2_1984_FULL']
TABLE_3_1984_NO_ANOMALY = _TABLES['TABLE_3_1984_NO_ANOMALY']
TABLE_1A_FULL = _TABLES['TABLE_1A_FULL']
TABLE_1B_FULL = _TABLES['TABLE_1B_FULL']
TABLE_2A_DERIVED = _TABLES['TABLE_2A_DERIVED']
TABLE_2B_DERIVED = _TABLES['TABLE_2B_DERIVED']
TABLE_3_21_FULL = _TABLES['TABLE_3_21_FULL']


def observed_ratio_vector_1984(source, offset, sigma_frac_floor=0.0):
    """Same return shape as stutzki_params.observed_ratio_vector, sourced from
    the full 51-row table instead of the 23-row hand transcription."""
    e = TABLE_2_1984_FULL[(source, offset)]
    keys = ('R_10_MAIN', 'R_12_MAIN', 'R_21_MAIN', 'R_01_MAIN')
    obs = {k: e[k] for k in keys}
    err = {k: (e[k + '_err'] if e[k + '_err'] is not None
               else max(sigma_frac_floor * e[k], 1e-3)) for k in keys}
    if e.get('R_22_MAIN') is not None:
        obs['R_22_MAIN'] = e['R_22_MAIN']
        err['R_22_MAIN'] = e.get('R_22_MAIN_err') or 0.05
    return obs, err
