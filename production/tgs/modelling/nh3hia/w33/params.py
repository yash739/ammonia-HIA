"""
Observed parameters for the W33 NH3 survey of Tursun et al. 2022 (A&A 658, A34).

Everything here is transcribed directly from the paper so that the modelling code
never carries hard-coded numbers of its own.  Table references are given per field.

Important conventions
---------------------
* All column densities are per cm^-2, volume densities per cm^-3, temperatures in K,
  velocities in km/s, sizes in arcsec.
* ``N_para`` (Table 6) is the quantity a *para*-NH3 LAMDA file must reproduce.
  Table 6's para column is the sum over the para levels (1,1)+(2,2)+(4,4)+(5,5);
  the (0,0) state belongs to ortho-NH3 (K=0) and is counted in ``N_ortho``.
* ``tau_11`` from Table 4 is the *total* optical depth of all 18 NH3 (1,1) hyperfine
  components returned by the GILDAS "NH3(1,1)" fit.  The central (main) group carries
  a fraction ``MAIN_GROUP_FRACTION`` of that.
* ``T_kin`` (Table A.5) is derived from T_rot(1,2) via Tafalla et al. (2004), Eq. 5.
* ``n_H2`` (Table A.5) is derived from the (1,1) excitation via Ho & Townes (1983),
  Eq. 8 -- it is a *lower bound* assuming a beam filling factor of unity.
"""

import numpy as np

# ----------------------------------------------------------------------------- #
# Physical constants (SI unless stated)
# ----------------------------------------------------------------------------- #
H_PLANCK = 6.62607015e-34       # J s
K_BOLTZ  = 1.380649e-23         # J / K
C_LIGHT  = 2.99792458e8         # m / s
M_H2     = 2.01588 * 1.66053906660e-27   # kg
AU       = 1.495978707e11       # m
PC       = 3.085677581e16       # m
ARCSEC   = np.pi / (180.0 * 3600.0)      # radians
T_CMB    = 2.725                # K

# ----------------------------------------------------------------------------- #
# Survey configuration
# ----------------------------------------------------------------------------- #
DISTANCE_PC = 2400.0            # W33 distance, 2.4 kpc (Immer et al. 2013; Sect. 1)

# Effelsberg 100-m: FWHM ~40" at 23 GHz, scaling as 1/nu.
# Paper Sect. 2.1: "the full width at half maximum (FWHM) beam size varied from
# 35" (0.4 pc) to 50" (0.6 pc) (~0.5 pc at 23 GHz)" over 17.9-26.2 GHz.
BEAM_REF_FREQ_HZ    = 23.0e9
BEAM_REF_FWHM_ARCSEC = 40.0

# Fraction of the total NH3 (1,1) hyperfine optical depth carried by the main group.
MAIN_GROUP_FRACTION = 0.5

# ----------------------------------------------------------------------------- #
# Table 2 -- NH3 inversion transition rest frequencies [Hz]
# ----------------------------------------------------------------------------- #
FREQ_HZ = {
    '1,1': 23694.4955e6,
    '2,2': 23722.6333e6,
    '3,3': 23870.1292e6,   # ortho
    '4,4': 24139.4163e6,
    '5,5': 24532.9887e6,
    '6,6': 25056.0250e6,   # ortho
    '2,1': 23098.8190e6,   # non-metastable
    '3,2': 22834.1820e6,   # non-metastable
}

# Energy of the lower level above ground [K] (Table 2)
E_LOW_K = {
    '1,1': 23.2, '2,2': 64.1, '3,3': 122.9, '4,4': 199.3,
    '5,5': 293.6, '6,6': 405.6, '2,1': 80.4, '3,2': 149.9,
}

# NH3 nuclear-spin symmetry.  K divisible by 3 -> ortho, otherwise para.
SPECIES = {k: ('ortho' if int(k.split(',')[1]) % 3 == 0 else 'para') for k in FREQ_HZ}

# Lines carried by the p-NH3 LAMDA file used here (Loreau et al. 2023), which spans
# para levels up to J=4.  (5,5) has no levels in that file and cannot be modelled.
PARA_LINES_AVAILABLE = ('1,1', '2,2', '4,4', '2,1', '3,2')

# ----------------------------------------------------------------------------- #
# Table 1 -- clump parameters
# ----------------------------------------------------------------------------- #
CLUMPS = {
    'W33_Main':  dict(radec=('18:14:13.50', '-17:55:47.0'), L_bol=449e3, M_sun=4.0e3,
                      T_dust=42.5, N_H2=4.6e23, stage='Hot core with HII region',
                      obs_size_arcsec=(380., 320.)),
    'W33_A':     dict(radec=('18:14:39.10', '-17:52:03.0'), L_bol=41e3,  M_sun=3.4e3,
                      T_dust=28.6, N_H2=2.5e23, stage='Hot core',
                      obs_size_arcsec=(40., 40.)),
    'W33_B':     dict(radec=('18:13:54.40', '-18:01:52.0'), L_bol=22e3,  M_sun=1.9e3,
                      T_dust=26.5, N_H2=2.1e23, stage='Hot core',
                      obs_size_arcsec=(120., 120.)),
    'W33_Main1': dict(radec=('18:14:25.00', '-17:53:58.0'), L_bol=11e3,  M_sun=0.5e3,
                      T_dust=28.6, N_H2=0.9e23, stage='High-mass protostellar',
                      obs_size_arcsec=(40., 40.)),
    'W33_A1':    dict(radec=('18:14:35.00', '-17:55:00.0'), L_bol=6e3,   M_sun=0.4e3,
                      T_dust=25.0, N_H2=1.7e23, stage='High-mass protostellar',
                      obs_size_arcsec=(40., 40.)),
    'W33_B1':    dict(radec=('18:14:07.10', '-18:00:45.0'), L_bol=16e3,  M_sun=0.2e3,
                      T_dust=38.6, N_H2=0.5e23, stage='High-mass protostellar',
                      obs_size_arcsec=(40., 40.)),
}

# ----------------------------------------------------------------------------- #
# Tables 3 & 4 -- line parameters at the (0,0) reference offset.
# Keyed line -> dict(T_mb, V_LSR, dv, tau, N).  tau is the *total* hfs optical depth;
# "<0.3" style upper limits are stored as the limit value with ``tau_limit=True``.
# ----------------------------------------------------------------------------- #
LINES = {
    'W33_A': {
        '1,1': dict(T_mb=5.18, V_LSR=37.5, dv=3.3, tau=3.6,  N=1.3e15),
        '2,2': dict(T_mb=3.62, V_LSR=37.6, dv=4.0, tau=0.4,  N=1.5e14, tau_limit=True),
        '3,3': dict(T_mb=2.48, V_LSR=37.6, dv=4.6, tau=0.4,  N=1.1e14, tau_limit=True),
        '4,4': dict(T_mb=0.50, V_LSR=37.7, dv=5.9, tau=0.3,  N=2.5e13, tau_limit=True),
        '5,5': dict(T_mb=0.24, V_LSR=38.5, dv=3.8, tau=0.3,  N=7.3e12, tau_limit=True),
        '6,6': dict(T_mb=0.21, V_LSR=37.9, dv=3.9, tau=0.4,  N=6.3e12, tau_limit=True),
        '2,1': dict(T_mb=0.48, V_LSR=36.1, dv=3.7, tau=0.2,  N=7.7e13, tau_limit=True),
        '3,2': dict(T_mb=0.18, V_LSR=36.4, dv=3.7, tau=0.4,  N=1.4e13, tau_limit=True),
    },
    'W33_B': {
        '1,1': dict(T_mb=3.08, V_LSR=55.9, dv=2.5, tau=6.8,  N=1.4e15),
        '2,2': dict(T_mb=2.27, V_LSR=55.8, dv=3.1, tau=0.2,  N=7.4e13, tau_limit=True),
        '3,3': dict(T_mb=1.43, V_LSR=55.7, dv=3.8, tau=0.3,  N=5.1e13, tau_limit=True),
        '4,4': dict(T_mb=0.24, V_LSR=55.4, dv=4.9, tau=0.4,  N=1.0e13, tau_limit=True),
        '5,5': dict(T_mb=0.27, V_LSR=56.6, dv=3.5, tau=0.3,  N=5.8e12, tau_limit=True),
        '6,6': dict(T_mb=0.14, V_LSR=55.7, dv=6.3, tau=0.4,  N=6.7e12, tau_limit=True),
        '2,1': dict(T_mb=0.35, V_LSR=53.8, dv=1.9, tau=0.2,  N=2.9e13, tau_limit=True),
        '3,2': dict(T_mb=0.17, V_LSR=55.1, dv=2.8, tau=0.2,  N=1.0e13, tau_limit=True),
    },
    'W33_Main1': {
        '1,1': dict(T_mb=4.11, V_LSR=36.6, dv=2.1, tau=5.2,  N=1.0e15),
        '2,2': dict(T_mb=2.21, V_LSR=36.7, dv=2.6, tau=0.3,  N=5.9e13, tau_limit=True),
        '3,3': dict(T_mb=0.97, V_LSR=36.7, dv=3.6, tau=0.4,  N=2.4e13, tau_limit=True),
        '4,4': dict(T_mb=0.16, V_LSR=36.5, dv=2.9, tau=0.3,  N=4.0e12, tau_limit=True),
    },
    'W33_A1': {
        '1,1': dict(T_mb=2.92, V_LSR=37.2, dv=2.4, tau=4.8,  N=9.1e14),
        '2,2': dict(T_mb=1.63, V_LSR=37.2, dv=2.9, tau=0.4,  N=5.0e13, tau_limit=True),
        '3,3': dict(T_mb=0.65, V_LSR=36.9, dv=4.0, tau=0.3,  N=2.4e13, tau_limit=True),
        '4,4': dict(T_mb=0.14, V_LSR=38.2, dv=4.3, tau=0.2,  N=5.1e12, tau_limit=True),
    },
    'W33_B1': {
        '1,1': dict(T_mb=1.28, V_LSR=34.2, dv=3.6, tau=3.2,  N=6.6e14),
        '2,2': dict(T_mb=0.66, V_LSR=34.2, dv=4.6, tau=0.3,  N=3.2e13, tau_limit=True),
        '3,3': dict(T_mb=0.30, V_LSR=34.8, dv=6.9, tau=0.2,  N=1.9e13, tau_limit=True),
    },
    # W33 Main is seen in ABSORPTION against its own HII-region continuum (Table 3).
    # T_mb is negative; modelling it requires a hot background, not a CMB boundary.
    'W33_Main': {
        '1,1': dict(T_mb=-1.89, V_LSR=33.7, dv=2.2, tau=1.1,  N=1.9e14),
        '2,2': dict(T_mb=-1.54, V_LSR=33.8, dv=3.4, tau=0.1,  N=5.5e13, tau_limit=True),
        '4,4': dict(T_mb=-0.30, V_LSR=33.7, dv=7.8, tau=0.1,  N=2.0e13, tau_limit=True),
        '5,5': dict(T_mb=-0.16, V_LSR=33.4, dv=5.4, tau=0.1,  N=6.9e12, tau_limit=True),
        '6,6': dict(T_mb=-0.20, V_LSR=33.4, dv=6.9, tau=0.1,  N=1.1e13, tau_limit=True),
        '3,3': dict(T_mb=3.74, V_LSR=36.2, dv=5.8, tau=0.3,  N=2.0e14, tau_limit=True),
    },
}

# ----------------------------------------------------------------------------- #
# Table A.5 -- excitation analysis at the (0,0) reference offset.
# Table 5 -- rotation temperatures.  Table 6 -- column densities and abundances.
# ----------------------------------------------------------------------------- #
EXCITATION = {
    #                    Tab. A.5           Tab. A.5        Tab. A.5     Tab. A.5
    'W33_A':     dict(T_ex=8.9, T_rot_12=15.0, T_kin=16.0, n_H2=2.6e4),
    'W33_B':     dict(T_ex=5.9, T_rot_12=11.0, T_kin=13.0, n_H2=1.1e4),
    'W33_Main1': dict(T_ex=7.2, T_rot_12=12.0, T_kin=13.0, n_H2=1.8e4),
    'W33_A1':    dict(T_ex=5.9, T_rot_12=11.0, T_kin=13.0, n_H2=1.1e4),
    'W33_B1':    dict(T_ex=4.3, T_rot_12=11.0, T_kin=13.0, n_H2=0.5e4),
    'W33_Main':  dict(T_ex=7.5, T_rot_12=23.0, T_kin=30.0, n_H2=1.2e4),
}

COLUMNS = {
    #                 Table 6: ortho / para / total N(NH3), fractional abundance, o/p
    'W33_A':     dict(N_ortho=1.9e15, N_para=1.5e15, N_total=3.5e15, X_total=1.4e-8, op=1.3),
    'W33_B':     dict(N_ortho=1.9e15, N_para=1.5e15, N_total=3.4e15, X_total=1.6e-8, op=1.3),
    'W33_Main1': dict(N_ortho=2.0e15, N_para=1.1e15, N_total=3.1e15, X_total=3.4e-8, op=1.8),
    'W33_A1':    dict(N_ortho=1.8e15, N_para=9.7e14, N_total=2.8e15, X_total=1.6e-8, op=1.9),
    'W33_B1':    dict(N_ortho=1.3e15, N_para=6.9e14, N_total=2.0e15, X_total=4.0e-8, op=1.9),
    'W33_Main':  dict(N_ortho=1.5e14, N_para=2.7e14, N_total=4.2e14, X_total=0.9e-9, op=0.5),
}

# Sources detected in emission -- the ones a CMB-bounded sphere can reproduce.
EMISSION_SOURCES = ('W33_A', 'W33_B', 'W33_Main1', 'W33_A1', 'W33_B1')

# W33 Main: continuum against which the inversion lines are seen in absorption.
# Sect. 4.5: "we obtain a continuum flux density of 18.7 +- 0.3 Jy or 26.2 +- 0.3 K
# on a main beam brightness temperature scale".
W33_MAIN_CONTINUUM_K = 26.2


# ----------------------------------------------------------------------------- #
# Derived quantities
# ----------------------------------------------------------------------------- #
def J_nu(T, freq_hz):
    """Rayleigh-Jeans equivalent radiation temperature [K] (paper Eq. 9)."""
    T = np.asarray(T, dtype=float)
    x = H_PLANCK * freq_hz / (K_BOLTZ * T)
    return (H_PLANCK * freq_hz / K_BOLTZ) / np.expm1(x)


def beam_fwhm_arcsec(freq_hz):
    """Effelsberg 100-m FWHM at ``freq_hz``, scaling as 1/nu from 40" at 23 GHz."""
    return BEAM_REF_FWHM_ARCSEC * (BEAM_REF_FREQ_HZ / freq_hz)


def T_kin_from_T_rot(T_rot_12):
    """T_rot(1,2) -> T_kin, Tafalla et al. (2004) as quoted in paper Eq. 5.

    Calibrated over T_kin = 5-20 K; the paper flags use above that as a caveat.
    """
    T_rot_12 = np.asarray(T_rot_12, dtype=float)
    denom = 1.0 - (T_rot_12 / 41.5) * np.log1p(1.1 * np.exp(-16.0 / T_rot_12))
    return T_rot_12 / denom


def T_rot_from_columns(N11, N22):
    """Paper Eq. 4: T_rot(1,2) from the (1,1) and (2,2) column densities."""
    return -41.5 / (np.log(np.asarray(N22) / 5.0) - np.log(np.asarray(N11) / 3.0))


def n_H2_from_excitation(T_ex, T_kin, T_bg=T_CMB,
                         A=1.71e-7, C=8.5e-11, freq_hz=None):
    """Paper Eq. 8 (Ho & Townes 1983): n(H2) [cm^-3] from the (1,1) excitation.

    ``A`` is the Einstein coefficient and ``C`` the collisional de-excitation rate
    for the (1,1) line (Danby et al. 1988).  Valid only where T_ex < T_kin.
    """
    if freq_hz is None:
        freq_hz = FREQ_HZ['1,1']
    num    = J_nu(T_ex, freq_hz) - J_nu(T_bg, freq_hz)
    den    = J_nu(T_kin, freq_hz) - J_nu(T_ex, freq_hz)
    factor = 1.0 + J_nu(T_kin, freq_hz) / (H_PLANCK * freq_hz / K_BOLTZ)
    return (A / C) * (num / den) * factor


def column_density_from_tau(tau, dv_kms, T_ex, J, K, freq_hz):
    """Paper Eq. 2 (Mauersberger et al. 1986a): N(J,K) [cm^-2]."""
    return (1.65e14 / (freq_hz / 1e9)) * (J * (J + 1) / K**2) * dv_kms * tau * T_ex


def arcsec_to_m(theta_arcsec, distance_pc=DISTANCE_PC):
    """Angular size [arcsec] -> physical size [m] at the given distance."""
    return theta_arcsec * ARCSEC * distance_pc * PC


def m_to_arcsec(length_m, distance_pc=DISTANCE_PC):
    """Physical size [m] -> angular size [arcsec] at the given distance."""
    return length_m / (distance_pc * PC) / ARCSEC


def fwhm_kms_to_vturb_ms(fwhm_kms):
    """Observed Gaussian FWHM linewidth [km/s] -> Magritte's Doppler b-parameter [m/s].

    Magritte's `vturb2` is the 1/e half-width b in exp(-(v/b)^2); standard radio
    astronomy linewidths (this paper's Delta-v, Stutzki's dv_obs) are FWHM.
    FWHM = 2*sqrt(ln2)*b.
    """
    return (fwhm_kms * 1000.0) / (2.0 * np.sqrt(np.log(2.0)))


def gaussian_beam_solid_angle(fwhm_rad):
    """Solid angle [sr] of a Gaussian beam given its FWHM [rad]."""
    return (np.pi / (4.0 * np.log(2.0))) * fwhm_rad**2


def disk_beam_dilution(theta_source_arcsec, freq_hz):
    """Beam dilution factor at a uniform disk's center: T_obs/T_source.

    Exact result for a Gaussian beam (FWHM ``beam_fwhm_arcsec(freq_hz)``)
    convolved with a uniform circular disk of angular radius
    ``theta_source_arcsec``; -> 0 for a point source, -> 1 once fully resolved.

    Stage-A screening approximation only (params.py docstring / plan
    "Physical framework"): a uniform *sphere*'s projected column/intensity is
    dome-shaped, not flat-topped, so this is not exact for our actual source
    geometry -- Stage B's real Magritte imaging + convolution replaces it.
    """
    theta_b = beam_fwhm_arcsec(freq_hz)
    return 1.0 - np.exp(-4.0 * np.log(2.0) * (theta_source_arcsec / theta_b) ** 2)


def catalogue_radius_cm_to_m(radius_req_cm):
    """Convert a `ModelGrid.py`/`nh3hia/model3d/sphere.py` catalogue `radius_req` to meters.

    `nh3hia/model3d/sphere.py` does `r_out = radius_sphere / 100` -- i.e. the
    catalogue's radius column is in cm, and Magritte's mesh (SI, meters) uses
    that value already divided by 100. Dividing by 100 here reproduces the
    same conversion for any angular-size/dilution calculation done outside
    Magritte (Stage A screening) -- get this wrong and every beam-dilution
    factor computed from the catalogue is off by ~1e4 in solid angle.
    """
    return radius_req_cm / 100.0


def catalogue_peak_column_cm2(numberdensity_cm3, XNH3, radius_req_cm):
    """Peak (line-of-sight-through-center) column density [cm^-2] for a catalogue row.

    N0 = 2 * r * n(H2) * X -- the exact identity for a uniform sphere; distinct
    from the catalogue's own "N_NH3" column, which is n*X*r/vturb (a Sobolev
    opacity-parameter proxy, not a column density -- see
    `nh3_NLTE_analysis.escape_probability`).
    """
    r_m = catalogue_radius_cm_to_m(radius_req_cm)
    n_m3 = numberdensity_cm3 * 1e6
    N_m2 = 2.0 * r_m * n_m3 * XNH3
    return N_m2 / 1e4  # m^-2 -> cm^-2


def diluted_column_cm2(N0_cm2, theta_source_arcsec, freq_hz):
    """Beam-averaged column density [cm^-2] given the peak column and source size."""
    return N0_cm2 * disk_beam_dilution(theta_source_arcsec, freq_hz)


def source(name):
    """Collect every paper-derived quantity for one source into a single dict."""
    if name not in CLUMPS:
        raise KeyError(f"unknown source {name!r}; known: {sorted(CLUMPS)}")
    out = dict(name=name)
    out.update(CLUMPS[name])
    out.update(EXCITATION[name])
    out.update(COLUMNS[name])
    out['lines'] = LINES.get(name, {})
    out['emission'] = name in EMISSION_SOURCES
    return out
