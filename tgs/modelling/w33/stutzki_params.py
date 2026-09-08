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
