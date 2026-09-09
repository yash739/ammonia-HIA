"""
Stutzki & Winnewisser (1985)'s own physical-consistency checks for resolving
the high-density/low-density chi2-degeneracy their fitting method produces --
reused here verbatim (his formulas, not ours) because our pipeline hits the
exact same kind of degenerate double-minimum (confirmed empirically: a
T=64K/n=2.75e16/18km-sphere candidate matched both the observed ratios AND a
column-rescaled tau_main about as well as the physically-sensible candidate
near the true answer). Stutzki resolves this with two checks, not arbitrary
search bounds:

  1. eta_f = T_B,obs / T_B,theor must be <= 1 (a beam filling factor above 1
     is unphysical by definition).
  2. Assuming clumps sit near their own Jeans mass, the number of clumps K a
     given (T_k, n_H2) can physically pack into the beam (from M_c(max)/M_J)
     must be large enough to explain the OBSERVED total linewidth via many
     independent clump velocities (K >= Delta_v_obs / Delta_v_clump). His own
     low-density branch failed exactly this: K<=1 predicted vs K>=5 required.

CONFIDENCE NOTE: every coefficient below (Jeans length 0.776e-2, Jeans mass
0.103, M_c(max) 347.4, and the K formula's derivation) is now verified against
ALL 23 positions of the paper's own Table 2a (derived quantities), not a
single spot check: max|log10 discrepancy| < 0.0005 dex for log(M_J),
log(M_c_max) and log(K) across all 23 rows, and < 0.0005 dex for
log(lambda_J) after fixing a real bug this cross-check found -- an earlier
transcription of the Jeans-length coefficient as 0.776e-3 (one digit off from
the paper's printed exponent) was exactly 10x too small at every single
position. That bug never affected clump_count_K or physically_consistent()
(the two functions the inversion pipeline actually calls), since neither
depends on jeans_length_pc -- jeans_mass_Msun and max_clump_mass_Msun use
their own independent coefficients, which matched to <0.001 dex even before
the fix. See tgs/modelling/w33/stutzki_tables_full.py, which parses the
paper's full Table 2a/2b directly from a complete user-provided digitization,
and clump_count_K's docstring for the K-formula derivation (the paper prints
it typeset-ambiguously; this cross-check is what resolved that ambiguity).

Units: T_k [K], n_H2 [cm^-3] (already true n_H2 for us -- unlike Stutzki's own
n', we use Loreau et al. NH3-H2 rates directly, not NH3-He rates scaled by
alpha=1.5, so the n'/1.75 pseudo-density correction in his Sect. 2 does NOT
apply to this pipeline and is deliberately not implemented here), distance in
pc, radius in pc.
"""

import numpy as np

X_HE_H2 = 0.25  # He abundance relative to H2, for reference only (not applied -- see module docstring)


def eta_f(T_B_obs, T_B_theor):
    """Beam filling factor. > 1 is unphysical -- a red flag on the candidate,
    not (necessarily) a coding bug."""
    if T_B_theor == 0:
        return np.nan
    return T_B_obs / T_B_theor


def jeans_length_pc(T_k, n_H2_cm3):
    """lambda_J [pc], Stutzki & Winnewisser (1985) Sect. 4, Eq. (their own
    numbering not preserved here -- see module docstring on coefficient
    confidence).

    Coefficient corrected from an earlier 0.776e-3 (a one-digit misread of the
    paper's printed exponent) to 0.776e-2, found and fixed by checking this
    function against all 23 positions of the paper's own Table 2a
    (log_lambda_J column, in units of 1e-2 pc): the old coefficient was
    exactly 10x too small at every single position (ratio 0.09999...==1e-1
    to 5 sig figs), while jeans_mass_Msun and max_clump_mass_Msun -- which do
    NOT depend on this function -- matched to <0.001 dex throughout, so this
    fix does not touch clump_count_K or physically_consistent(), the two
    functions actually used in the inversion pipeline.
    """
    T1 = T_k / 10.0
    n7 = n_H2_cm3 / 1e7
    return 0.776e-2 * np.sqrt(T1 / n7)


def jeans_mass_Msun(T_k, n_H2_cm3):
    """M_J [Msun]."""
    T1 = T_k / 10.0
    n7 = n_H2_cm3 / 1e7
    return 0.103 * np.sqrt(T1 ** 3 / n7)


def max_clump_mass_Msun(n_H2_cm3, eta_f_value, distance_pc):
    """M_c(max) [Msun] -- the largest clump mass consistent with the derived
    filling factor and distance, per Stutzki's Eq. for M_c(max)."""
    n7 = n_H2_cm3 / 1e7
    r_05kpc = distance_pc / 500.0
    if eta_f_value <= 0 or np.isnan(eta_f_value):
        return np.nan
    return 347.4 * n7 * r_05kpc ** 3 * eta_f_value ** 1.5


def clump_count_K(T_k, n_H2_cm3, eta_f_value, distance_pc, dv_obs_kms, dv_clump_kms):
    """Predicted number of clumps per beam, Stutzki & Winnewisser (1985)
    Sect. 4: setting M_c = M_J (clumps sit at their own Jeans mass) gives an
    intermediate quantity kappa = (M_c(max)/M_J)^(2/3), and the paper's own
    tabulated K (Table 2a/2b) is kappa * (Delta_v_obs/Delta_v) -- NOT kappa
    alone. This is confirmed by direct numerical reproduction of Table 2a's
    S106 (200,40) row (K=56.36 there vs 56.37 computed here from the row's
    own T_k/n'/eta_f/Delta_v_obs) -- the paper's printed formula is
    typeset-ambiguous (a fraction bar lost in OCR/scan) and this check is
    what resolves it: without the Delta_v_obs/Delta_v factor the same row
    computes to kappa=11.8, which does not match the paper's tabulated
    56.4=11.8*(1.43/0.3).

    Physically: kappa alone (>=1, i.e. M_c(max)>=M_J) is the actual
    consistency criterion -- it asks whether even one Jeans-mass clump fits
    in the derived mass budget. The Delta_v_obs/Delta_v factor is carried
    along only so K can be compared directly against
    clump_count_required(dv_obs_kms, dv_clump_kms) = Delta_v_obs/Delta_v, per
    the paper's own reporting convention (e.g. "10-100 clumps per beam").

    NaN if eta_f is unphysical (<=0) since M_c(max) is undefined there."""
    M_c_max = max_clump_mass_Msun(n_H2_cm3, eta_f_value, distance_pc)
    M_J = jeans_mass_Msun(T_k, n_H2_cm3)
    if np.isnan(M_c_max) or M_J <= 0:
        return np.nan
    kappa = (M_c_max / M_J) ** (2.0 / 3.0)
    return kappa * (dv_obs_kms / dv_clump_kms)


def clump_count_required(dv_obs_kms, dv_clump_kms):
    """K_min = Delta_v_obs / Delta_v_clump -- the number of independent,
    narrow-line clumps needed to statistically broaden the observed line
    profile to its actual width. Purely observational, no model dependence."""
    return dv_obs_kms / dv_clump_kms


def physically_consistent(T_k, n_H2_cm3, T_B_obs, T_B_theor, dv_obs_kms, dv_clump_kms,
                           distance_pc, eta_f_max=1.0, K_safety_margin=1.0):
    """Stutzki's own two-check filter. Returns (is_consistent: bool, detail: dict).

    is_consistent is False if EITHER:
      - eta_f > eta_f_max (default 1.0 -- a hard physical ceiling), or
      - K_predicted < K_safety_margin * K_required (the branch can't pack
        enough Jeans-mass clumps into the beam to explain the observed
        linewidth).
    """
    ef = eta_f(T_B_obs, T_B_theor)
    K_pred = clump_count_K(T_k, n_H2_cm3, ef, distance_pc, dv_obs_kms, dv_clump_kms) if ef > 0 else np.nan
    K_req = clump_count_required(dv_obs_kms, dv_clump_kms)

    eta_ok = (not np.isnan(ef)) and (0 < ef <= eta_f_max)
    K_ok = (not np.isnan(K_pred)) and (K_pred >= K_safety_margin * K_req)

    detail = dict(eta_f=ef, K_predicted=K_pred, K_required=K_req,
                   eta_ok=bool(eta_ok), K_ok=bool(K_ok))
    return bool(eta_ok and K_ok), detail
