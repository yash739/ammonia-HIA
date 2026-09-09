"""
Benchmark fits from Stutzki & Winnewisser (1985, A&A 144, 13) -- "On the
interpretation of hyperfine-structure intensity anomalies in the NH3 (J,K)=(1,1)
inversion transition".

Used as Stage 0 of the W33 pipeline: before trusting our full 3D NLTE + hyperfine
Gaussian-fit machinery on W33 (where there's no independent size/density check),
reproduce this paper's own fits for real anomalous sources with our own code and
compare. Per the user: the goal is not to reimplement Stutzki's escape-probability
formalism (his Appendix A) -- it's to build a sphere at *his* published
(T_k, n_H2, N_NH3) and see whether Magritte's own NLTE solve + a Gaussian hyperfine
fit (nh3_NLTE_analysis.fit_five_gaussians) reproduces his T_B/eta_f, using modern
Loreau et al. (2023) NH3-H2 collision rates instead of his Green (1981) NH3-He
rates scaled by alpha=1.5.

All numbers below are Table 1a (the paper's own favoured "high-density solution" --
Table 1b's low-density branch is flagged in the paper itself as self-inconsistent,
see Sect. 4 discussion of clump number K). Fixed model parameters used throughout
Table 1a: individual-clump intrinsic linewidth Δv = 0.3 km/s (NOT the same quantity
as Δv_obs below -- see note on vturb translation).
"""

import numpy as np

# ----------------------------------------------------------------------------- #
# Source distances and the paper's assumed reference source angular size, from
# the header of Table 2a: "<source> (r=<dist> kpc, Θs=<size>')".
# ----------------------------------------------------------------------------- #
SOURCE_DISTANCE_PC = {
    'S106': 600.0,
    'S87': 1300.0,
    'W48': 3400.0,
    'OMC': 500.0,
}
SOURCE_THETA_S_ARCMIN = {
    'S106': 1.5,
    'S87': 1.5,
    'W48': 3.0,
    'OMC': 1.5,
}

# Intrinsic clump linewidth assumed throughout Table 1a's radiative transfer calc.
CLUMP_DV_KMS = 0.3

# ----------------------------------------------------------------------------- #
# Table 1a: fit results to the observed (1,1) spectra, high-density solution.
#
# Fields: T_k [K], log(n_H2'/cm^-3), log(N_NH3/cm^-2), chi2, dv_obs [km/s]
# (the *observed/blended* linewidth -- this is what should be used as our
# sphere's vturb Doppler parameter, NOT CLUMP_DV_KMS, since our homogeneous
# sphere has one coherent velocity dispersion rather than Stutzki's ensemble of
# unresolved narrow-line clumps), T_B_theor [K], T_B_obs [K], eta_f = T_B_obs/T_B_theor.
# `includes_22_hfs`: whether the fit also used the (2,2) hyperfine satellites
# (marked '(a)' in the paper).
# ----------------------------------------------------------------------------- #
TABLE_1A = {
    'S106': {
        '200,40':  dict(T_k=21.89, log_nH2=6.528, log_N_NH3=14.29, chi2=8.61, dv_obs=1.43, T_B_theor=12.705, T_B_obs=3.04, eta_f=0.237, includes_22_hfs=False),
        '200,80':  dict(T_k=17.72, log_nH2=7.099, log_N_NH3=14.33, chi2=0.27, dv_obs=1.52, T_B_theor=11.044, T_B_obs=1.32, eta_f=0.120, includes_22_hfs=False),
        '160,40':  dict(T_k=23.78, log_nH2=6.770, log_N_NH3=14.23, chi2=0.92, dv_obs=1.91, T_B_theor=12.045, T_B_obs=2.49, eta_f=0.207, includes_22_hfs=False),
        '160,0':   dict(T_k=23.05, log_nH2=6.296, log_N_NH3=14.12, chi2=0.29, dv_obs=1.58, T_B_theor=10.487, T_B_obs=1.96, eta_f=0.187, includes_22_hfs=False),
        '90,30':   dict(T_k=34.25, log_nH2=4.960, log_N_NH3=13.78, chi2=0.32, dv_obs=2.37, T_B_theor=5.484,  T_B_obs=0.55, eta_f=0.100, includes_22_hfs=False),
        '40,0':    dict(T_k=27.46, log_nH2=6.499, log_N_NH3=14.23, chi2=0.70, dv_obs=1.58, T_B_theor=12.703, T_B_obs=0.84, eta_f=0.066, includes_22_hfs=False),
        '0,40':    dict(T_k=21.91, log_nH2=6.537, log_N_NH3=14.26, chi2=0.46, dv_obs=1.81, T_B_theor=12.255, T_B_obs=1.38, eta_f=0.113, includes_22_hfs=False),
        '0,0':     dict(T_k=21.81, log_nH2=6.255, log_N_NH3=14.36, chi2=0.88, dv_obs=1.47, T_B_theor=13.708, T_B_obs=4.71, eta_f=0.344, includes_22_hfs=True),
        '0,-40':   dict(T_k=22.5,  log_nH2=4.839, log_N_NH3=14.07, chi2=0.28, dv_obs=2.00, T_B_theor=8.150,  T_B_obs=1.42, eta_f=0.174, includes_22_hfs=False),
        '-40,0':   dict(T_k=20.73, log_nH2=5.496, log_N_NH3=14.31, chi2=4.83, dv_obs=1.21, T_B_theor=12.286, T_B_obs=3.57, eta_f=0.291, includes_22_hfs=False),
    },
    'S87': {
        '0,0_21.0kms': dict(T_k=23.99, log_nH2=6.477, log_N_NH3=14.47, chi2=0.57, dv_obs=1.36, T_B_theor=15.936, T_B_obs=1.10, eta_f=0.069, includes_22_hfs=False),
        '0,0_23.5kms': dict(T_k=26.03, log_nH2=7.258, log_N_NH3=14.30, chi2=4.23, dv_obs=2.05, T_B_theor=12.968, T_B_obs=1.28, eta_f=0.099, includes_22_hfs=False),
    },
    'W48': {
        '-80,40':     dict(T_k=20.76, log_nH2=6.143, log_N_NH3=14.16, chi2=1.08, dv_obs=1.59, T_B_theor=10.567, T_B_obs=1.56, eta_f=0.148, includes_22_hfs=False),
        '-80,80':     dict(T_k=18.88, log_nH2=7.227, log_N_NH3=14.07, chi2=0.50, dv_obs=1.72, T_B_theor=8.827,  T_B_obs=2.60, eta_f=0.295, includes_22_hfs=False),
        '-40,40':     dict(T_k=22.34, log_nH2=6.830, log_N_NH3=14.37, chi2=1.30, dv_obs=2.66, T_B_theor=13.696, T_B_obs=1.40, eta_f=0.102, includes_22_hfs=False),
        '0,40_45kms': dict(T_k=32.20, log_nH2=5.525, log_N_NH3=14.41, chi2=1.59, dv_obs=1.95, T_B_theor=16.768, T_B_obs=1.01, eta_f=0.060, includes_22_hfs=False),
        '0,80':       dict(T_k=25.68, log_nH2=6.425, log_N_NH3=14.42, chi2=0.10, dv_obs=2.07, T_B_theor=15.695, T_B_obs=1.43, eta_f=0.091, includes_22_hfs=False),
        '80,-80':     dict(T_k=19.86, log_nH2=6.218, log_N_NH3=14.15, chi2=0.82, dv_obs=1.22, T_B_theor=10.212, T_B_obs=0.81, eta_f=0.079, includes_22_hfs=False),
    },
    'OMC': {
        'OMC2': dict(T_k=25.90, log_nH2=7.391, log_N_NH3=13.95, chi2=1.45, dv_obs=1.256, T_B_theor=7.583,  T_B_obs=4.75, eta_f=0.626, includes_22_hfs=True),
        'S1':   dict(T_k=19.01, log_nH2=7.576, log_N_NH3=14.17, chi2=8.94, dv_obs=0.944, T_B_theor=9.923,  T_B_obs=9.41, eta_f=0.948, includes_22_hfs=True),
        'S2':   dict(T_k=23.77, log_nH2=6.743, log_N_NH3=14.14, chi2=4.76, dv_obs=0.637, T_B_theor=10.699, T_B_obs=5.13, eta_f=0.479, includes_22_hfs=True),
        'S3':   dict(T_k=26.33, log_nH2=7.135, log_N_NH3=14.17, chi2=0.41, dv_obs=1.506, T_B_theor=10.983, T_B_obs=7.12, eta_f=0.648, includes_22_hfs=True),
        'S4':   dict(T_k=27.82, log_nH2=7.317, log_N_NH3=14.16, chi2=0.59, dv_obs=1.106, T_B_theor=10.714, T_B_obs=6.56, eta_f=0.612, includes_22_hfs=True),
    },
}

# Representative subset the plan's Stage 0 proposes running first (cheapest,
# broadest coverage in T_k/n_H2/eta_f before committing to the full table):
# S106 center (highest S/N, includes (2,2) hfs), S87 (both velocity components),
# W48 -80,80 (most beam-diluted, eta_f=0.295), OMC S3 and S4 (high eta_f, dense).
STAGE_0_SUBSET = [
    ('S106', '0,0'),
    ('S87', '0,0_21.0kms'),
    ('S87', '0,0_23.5kms'),
    ('W48', '-80,80'),
    ('OMC', 'S3'),
    ('OMC', 'S4'),
]


def fit(field, position):
    """Look up one Table 1a fit result, e.g. fit('OMC', 'S3')."""
    return dict(TABLE_1A[field][position])


def iter_stage_0_subset():
    """Yield (field, position, fit_dict) for the Stage 0 representative subset."""
    for field, position in STAGE_0_SUBSET:
        yield field, position, fit(field, position)


def iter_all():
    """Yield (field, position, fit_dict) for every Table 1a entry."""
    for field, positions in TABLE_1A.items():
        for position, d in positions.items():
            yield field, position, dict(d)


def n_H2_cm3(entry):
    return 10.0 ** entry['log_nH2']


def N_NH3_cm2(entry):
    return 10.0 ** entry['log_N_NH3']


# ----------------------------------------------------------------------------- #
# Table 3: "Results of NH3(2,1) observations" -- the paper's own genuine
# observational (2,1)/(1,1) ratio check (Sect. 4, "A precise indicator for high
# densities is the intensity of the non-metastable (2,1) inversion line"). This
# is the ONE place in the paper where a hyperfine/rotational intensity RATIO is
# published as an actual number (both his escape-probability theor. value and
# the real observed value) rather than only being implicit in a fitted
# (T_k, n_H2, N_NH3) triple (as Table 1a is). Text confirms these three T_B(1,1)
# values match Table 1a's '(a)'-flagged entries: S106 '0,0' (4.71 vs 4.713),
# OMC 'S3' (7.12 vs 7.115), OMC 'S4' (6.56 vs 6.558) -- same positions, so the
# matching Table 1a fit parameters (T_k, n_H2, N_NH3) are the right sphere to
# build for this check.
# ----------------------------------------------------------------------------- #
TABLE_3_21 = {
    ('S106', '0,0'): dict(T_B_21=0.065, T_B_21_err=0.014, T_B_11=4.713, T_B_11_err=0.060,
                           ratio_theor=0.018, ratio_obs=0.014, ratio_obs_err=0.003),
    ('OMC', 'S3'):   dict(T_B_21=0.109, T_B_21_err=0.017, T_B_11=7.115, T_B_11_err=0.049,
                           ratio_theor=0.052, ratio_obs=0.015, ratio_obs_err=0.002),
    ('OMC', 'S4'):   dict(T_B_21=0.186, T_B_21_err=0.014, T_B_11=6.558, T_B_11_err=0.080,
                           ratio_theor=0.068, ratio_obs=0.028, ratio_obs_err=0.002),
}


def iter_table_3_21():
    """Yield (field, position, table3_entry, table1a_fit) for the 3 positions
    with a real published (2,1)/(1,1) observation."""
    for (field, position), entry in TABLE_3_21.items():
        yield field, position, entry, fit(field, position)


# ----------------------------------------------------------------------------- #
# Stutzki, Jackson, Olberg, Barrett & Winnewisser (1984), A&A 139, 258, Table 2:
# "Sources with hfs-anomalies" -- the OBSERVED (1,1) satellite/main intensity
# ratios, with errors, plus T_B(2,2)/T_B(1,1).
#
# This is the measured five-ratio vector, i.e. exactly the observable set the
# inversion pipeline fits (RATIO_KEYS in invert_ratios.py). The 1985 paper does
# NOT publish these -- it gives fitted parameters (its Table 1a) and the (2,1)
# data (its Table 3) -- so this table is what makes an independent retrieval
# possible: fit these ratios, recover (T_k, n_H2, N/dv), compare against
# Stutzki's own fit, and then predict (2,1) as a held-out test.
#
# Effelsberg 100 m, 40" beam at the inversion frequencies; satellite/main ratios
# quoted to better than 3% (their Sect. II).
#
# Ratio keys use OUR convention, which matches theirs: R_10 is F1 = 1->0
# (blueshifted outer), R_01 is F1 = 0->1 (redshifted outer), R_12 is 1->2
# (blueshifted inner), R_21 is 2->1 (redshifted inner). See nh3_hyperfine.py.
#
# TRANSCRIBED FROM A SCAN, then cross-validated: T_B(1,1) for S106 (0,0), OMC S3
# and OMC S4 reproduces TABLE_3_21 exactly, and dv reproduces TABLE_1A's dv_obs
# exactly, for all three. Rows without that overlap have not been independently
# checked -- verify against the paper before relying on them quantitatively.
#
# Positions (Table 1, B1950): OMC2 05 32 58.6 -05 11 42; S1 05 32 54.7 -05 14 04;
# S2 05 32 53.5 -05 16 25; S3 05 32 49.3 -05 21 02; S4 05 32 48.0 -05 22 02;
# S106 20 25 25 +37 12 30. The OMC positions convert to roughly Dec -05 09 to
# -05 20 in J2000, i.e. inside the GAS Orion A footprint (-05 30 to -05 00), so
# modern high-S/N spectra exist at these same positions.
# ----------------------------------------------------------------------------- #
TABLE_2_1984 = {
    ('OMC', 'OMC2'): dict(T_B_11=4.752, T_B_11_err=0.039, v_lsr=11.2, dv=1.259, dv_err=0.012,
                           R_10=0.240, R_10_err=0.026, R_12=0.325, R_12_err=0.009,
                           R_21=0.313, R_21_err=0.009, R_01=0.329, R_01_err=0.005,
                           R_22_MAIN=0.572, R_22_MAIN_err=0.009, cross_checked=False),
    ('OMC', 'S1'):   dict(T_B_11=9.613, T_B_11_err=0.042, v_lsr=11.084, dv=1.084, dv_err=0.002,
                           R_10=0.329, R_10_err=0.005, R_12=0.409, R_12_err=0.005,
                           R_21=0.362, R_21_err=0.005, R_01=0.365, R_01_err=0.005,
                           R_22_MAIN=0.484, R_22_MAIN_err=0.005, cross_checked=False),
    ('OMC', 'S2'):   dict(T_B_11=5.132, T_B_11_err=0.068, v_lsr=10.797, dv=0.637, dv_err=0.004,
                           R_10=0.281, R_10_err=0.016, R_12=0.325, R_12_err=0.013,
                           R_21=0.367, R_21_err=0.014, R_01=0.412, R_01_err=0.014,
                           R_22_MAIN=0.518, R_22_MAIN_err=0.013, cross_checked=False),
    ('OMC', 'S3'):   dict(T_B_11=7.115, T_B_11_err=0.049, v_lsr=9.873, dv=1.506, dv_err=0.033,
                           R_10=0.245, R_10_err=0.008, R_12=0.372, R_12_err=0.007,
                           R_21=0.336, R_21_err=0.007, R_01=0.377, R_01_err=0.008,
                           R_22_MAIN=0.632, R_22_MAIN_err=0.007, cross_checked=True),
    ('OMC', 'S4'):   dict(T_B_11=6.558, T_B_11_err=0.084, v_lsr=9.698, dv=1.106, dv_err=0.021,
                           R_10=0.248, R_10_err=0.008, R_12=0.361, R_12_err=0.008,
                           R_21=0.317, R_21_err=0.008, R_01=0.378, R_01_err=0.008,
                           R_22_MAIN=0.693, R_22_MAIN_err=0.022, cross_checked=True),
    ('S106', '0,0'): dict(T_B_11=4.713, T_B_11_err=0.036, v_lsr=-1.24, dv=1.470, dv_err=0.021,
                           R_10=0.294, R_10_err=0.014, R_12=0.456, R_12_err=0.014,
                           R_21=0.428, R_21_err=0.014, R_01=0.505, R_01_err=0.015,
                           R_22_MAIN=0.522, R_22_MAIN_err=0.020, cross_checked=True),
}

# Positions where BOTH the observed (1,1) ratio vector (1984 Table 2) and the
# (2,1) measurement (1985 Table 3) exist -- the held-out-prediction test set.
TEST2_POSITIONS = [('S106', '0,0'), ('OMC', 'S3'), ('OMC', 'S4')]

# ----------------------------------------------------------------------------- #
# EXTENSION of TABLE_2_1984 to every position also present in TABLE_1A -- S106
# (10 positions), S87 (2), W48 (6) -- transcribed from a 600 dpi crop of the
# same Table 2 (Stutzki et al. 1984, A&A 139, 258), read cleanly this time
# (earlier low-resolution reads of this table were not trusted for these rows).
#
# Cross-validated against TABLE_1A's own T_B_obs and dv_obs (which the 1985
# paper states are carried over from this 1984 survey): T_B(1,1) matches to
# the last published digit for all 23 positions now in this table. dv matches
# for 21/23; two do not (S106 '200,40': 1.21 here vs 1.43 in TABLE_1A; OMC 'S1':
# 1.084 here vs 0.944 in TABLE_1A, with T_B also 2% off there). Given the
# otherwise exact reproduction elsewhere, these two are flagged as a probable
# genuine difference between the 1984 survey reduction and the 1985 refit
# (cross_checked=False), not assumed to be a transcription error in either
# direction -- do not silently trust either value for these two rows without
# checking the source again.
# ----------------------------------------------------------------------------- #
TABLE_2_1984.update({
    ('W48', '-80,40'):     dict(T_B_11=1.56, T_B_11_err=0.06, v_lsr=42.463, dv=1.59, dv_err=0.07,
                                 R_10=0.28, R_10_err=0.04, R_12=0.35, R_12_err=0.04,
                                 R_21=0.41, R_21_err=0.04, R_01=0.42, R_01_err=0.04,
                                 R_22_MAIN=0.42, R_22_MAIN_err=0.04, cross_checked=True),
    ('W48', '-80,80'):     dict(T_B_11=2.60, T_B_11_err=0.05, v_lsr=42.725, dv=1.72, dv_err=0.04,
                                 R_10=0.28, R_10_err=0.02, R_12=0.40, R_12_err=0.02,
                                 R_21=0.36, R_21_err=0.02, R_01=0.34, R_01_err=0.02,
                                 R_22_MAIN=0.41, R_22_MAIN_err=0.02, cross_checked=True),
    ('W48', '-40,40'):     dict(T_B_11=1.40, T_B_11_err=0.04, v_lsr=42.482, dv=2.66, dv_err=0.09,
                                 R_10=0.26, R_10_err=0.04, R_12=0.48, R_12_err=0.03,
                                 R_21=0.42, R_21_err=0.03, R_01=0.44, R_01_err=0.03,
                                 R_22_MAIN=0.59, R_22_MAIN_err=0.03, cross_checked=True),
    ('W48', '0,40_45kms'): dict(T_B_11=1.02, T_B_11_err=0.06, v_lsr=44.725, dv=1.95, dv_err=0.16,
                                 R_10=0.24, R_10_err=0.05, R_12=0.41, R_12_err=0.05,
                                 R_21=0.32, R_21_err=0.06, R_01=0.76, R_01_err=0.07,
                                 R_22_MAIN=0.67, R_22_MAIN_err=0.07, cross_checked=True),
    ('W48', '0,80'):       dict(T_B_11=1.43, T_B_11_err=0.06, v_lsr=42.394, dv=2.07, dv_err=0.11,
                                 R_10=0.27, R_10_err=0.05, R_12=0.43, R_12_err=0.05,
                                 R_21=0.41, R_21_err=0.05, R_01=0.54, R_01_err=0.06,
                                 R_22_MAIN=0.63, R_22_MAIN_err=0.04, cross_checked=True),
    ('W48', '80,-80'):     dict(T_B_11=0.81, T_B_11_err=0.06, v_lsr=42.713, dv=1.22, dv_err=0.17,
                                 R_10=0.34, R_10_err=0.08, R_12=0.35, R_12_err=0.06,
                                 R_21=0.38, R_21_err=0.10, R_01=0.44, R_01_err=0.08,
                                 R_22_MAIN=0.40, R_22_MAIN_err=0.06, cross_checked=True),

    ('S87', '0,0_21.0kms'): dict(T_B_11=1.097, T_B_11_err=0.041, v_lsr=20.88, dv=1.36, dv_err=0.07,
                                  R_10=0.32, R_10_err=0.05, R_12=0.44, R_12_err=0.04,
                                  R_21=0.44, R_21_err=0.05, R_01=0.56, R_01_err=0.04,
                                  R_22_MAIN=0.63, R_22_MAIN_err=0.04, cross_checked=True),
    ('S87', '0,0_23.5kms'): dict(T_B_11=1.279, T_B_11_err=0.034, v_lsr=23.49, dv=2.05, dv_err=0.07,
                                  R_10=0.28, R_10_err=0.03, R_12=0.46, R_12_err=0.03,
                                  R_21=0.31, R_21_err=0.03, R_01=0.39, R_01_err=0.03,
                                  R_22_MAIN=0.69, R_22_MAIN_err=0.03, cross_checked=True),

    ('S106', '200,80'): dict(T_B_11=1.32, T_B_11_err=0.09, v_lsr=-1.95, dv=1.52, dv_err=0.12,
                              R_10=0.39, R_10_err=0.08, R_12=0.49, R_12_err=0.08,
                              R_21=0.42, R_21_err=0.07, R_01=0.45, R_01_err=0.10,
                              R_22_MAIN=0.48, R_22_MAIN_err=0.10, cross_checked=True),
    # dv mismatches TABLE_1A (1.21 here vs 1.43 there); T_B matches exactly. See note above.
    ('S106', '200,40'): dict(T_B_11=3.04, T_B_11_err=0.06, v_lsr=-2.23, dv=1.21, dv_err=0.03,
                              R_10=0.32, R_10_err=0.02, R_12=0.44, R_12_err=0.02,
                              R_21=0.34, R_21_err=0.02, R_01=0.47, R_01_err=0.02,
                              R_22_MAIN=0.51, R_22_MAIN_err=0.05, cross_checked=False),
    ('S106', '160,40'): dict(T_B_11=2.49, T_B_11_err=0.08, v_lsr=-2.12, dv=1.91, dv_err=0.07,
                              R_10=0.27, R_10_err=0.03, R_12=0.41, R_12_err=0.03,
                              R_21=0.34, R_21_err=0.03, R_01=0.43, R_01_err=0.04,
                              R_22_MAIN=0.55, R_22_MAIN_err=0.04, cross_checked=True),
    ('S106', '160,0'):  dict(T_B_11=1.96, T_B_11_err=0.04, v_lsr=-1.49, dv=1.58, dv_err=0.04,
                              R_10=0.23, R_10_err=0.02, R_12=0.38, R_12_err=0.02,
                              R_21=0.37, R_21_err=0.02, R_01=0.40, R_01_err=0.02,
                              R_22_MAIN=0.46, R_22_MAIN_err=0.02, cross_checked=True),
    ('S106', '90,30'):  dict(T_B_11=0.55, T_B_11_err=0.03, v_lsr=-1.42, dv=2.37, dv_err=0.14,
                              R_10=0.22, R_10_err=0.06, R_12=0.36, R_12_err=0.06,
                              R_21=0.27, R_21_err=0.04, R_01=0.40, R_01_err=0.06,
                              R_22_MAIN=0.60, R_22_MAIN_err=0.06, cross_checked=True),
    ('S106', '40,0'):   dict(T_B_11=0.84, T_B_11_err=0.03, v_lsr=-1.29, dv=1.58, dv_err=0.07,
                              R_10=0.20, R_10_err=0.05, R_12=0.36, R_12_err=0.04,
                              R_21=0.40, R_21_err=0.05, R_01=0.45, R_01_err=0.05,
                              R_22_MAIN=0.60, R_22_MAIN_err=0.05, cross_checked=True),
    ('S106', '0,40'):   dict(T_B_11=1.38, T_B_11_err=0.03, v_lsr=-1.17, dv=1.81, dv_err=0.05,
                              R_10=0.26, R_10_err=0.02, R_12=0.43, R_12_err=0.02,
                              R_21=0.40, R_21_err=0.02, R_01=0.43, R_01_err=0.02,
                              R_22_MAIN=0.50, R_22_MAIN_err=0.02, cross_checked=True),
    ('S106', '0,-40'):  dict(T_B_11=1.42, T_B_11_err=0.12, v_lsr=-0.75, dv=2.00, dv_err=0.19,
                              R_10=0.20, R_10_err=0.09, R_12=0.42, R_12_err=0.09,
                              R_21=0.42, R_21_err=0.10, R_01=0.41, R_01_err=0.09,
                              R_22_MAIN=0.42, R_22_MAIN_err=0.07, cross_checked=True),
    ('S106', '-40,0'):  dict(T_B_11=3.57, T_B_11_err=0.04, v_lsr=-1.32, dv=1.21, dv_err=0.02,
                              R_10=0.30, R_10_err=0.01, R_12=0.43, R_12_err=0.01,
                              R_21=0.45, R_21_err=0.01, R_01=0.51, R_01_err=0.01,
                              R_22_MAIN=0.45, R_22_MAIN_err=0.01, cross_checked=True),

    # dv mismatches TABLE_1A (1.084 here vs 0.944 there); T_B also 2% off (9.613 vs 9.41). See note above.
    ('OMC', 'S1'):      dict(T_B_11=9.613, T_B_11_err=0.042, v_lsr=11.084, dv=1.084, dv_err=0.002,
                              R_10=0.329, R_10_err=0.005, R_12=0.409, R_12_err=0.005,
                              R_21=0.362, R_21_err=0.005, R_01=0.365, R_01_err=0.005,
                              R_22_MAIN=0.484, R_22_MAIN_err=0.005, cross_checked=False),
})

# Full set of positions present in BOTH TABLE_1A (fitted params) and
# TABLE_2_1984 (observed ratios) -- the complete retrieval-vs-Stutzki's-own-fit
# test set, 23 positions across S106/S87/W48/OMC. TEST2_POSITIONS (above)
# remains the 3-position subset that ALSO has a (2,1) measurement (Table 3),
# for the held-out (2,1) prediction test specifically.
RETRIEVAL_TEST_POSITIONS = [k for k in TABLE_2_1984.keys()
                            if k[1] in TABLE_1A.get(k[0], {})]



def observed_ratio_vector(field, position):
    """Observed 5-ratio vector and 1-sigma errors, in invert_ratios' key names."""
    e = TABLE_2_1984[(field, position)]
    obs = {'R_10_MAIN': e['R_10'], 'R_12_MAIN': e['R_12'],
           'R_21_MAIN': e['R_21'], 'R_01_MAIN': e['R_01'],
           'R_22_MAIN': e['R_22_MAIN']}
    err = {'R_10_MAIN': e['R_10_err'], 'R_12_MAIN': e['R_12_err'],
           'R_21_MAIN': e['R_21_err'], 'R_01_MAIN': e['R_01_err'],
           'R_22_MAIN': e['R_22_MAIN_err']}
    return obs, err
