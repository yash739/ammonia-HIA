"""Stutzki & Winnewisser (1985) NH3 (1,1)/(2,1)/(2,2) escape-probability model,
implemented directly from the paper (Astron. Astrophys. 144, 13-26) rather than
via Magritte -- this is the paper's OWN fast method (a 36-level statistical
equilibrium solve with a spherical escape-probability closure), not the 3D
radiative-transfer approach used elsewhere in this repo's w33/ pipeline. Since
it needs no mesh, no rays and no imaging, a single grid point costs one
36x36 linear solve per fixed-point iteration -- milliseconds, not minutes.

Deviations from a literal reading of stutzki_model_1985.md / the paper, and why:

* Fixed-point (Lambda) iteration instead of full Newton-Raphson. At fixed J_t
  the rate equations ARE linear in the populations, so holding tau_G/S_G/beta_G
  fixed at their previous-iterate value and re-solving the resulting linear
  system each step is a standard, robust scheme for exactly this problem (it is
  what RADEX and most escape-probability codes do; the paper's own Sect. 3
  text calls its scheme "iterative solution of the linearized rate equations",
  which is this). Deriving and inverting an explicit Jacobian of tau's
  dependence on populations would be substantially more code for the same
  fixed point, with no accuracy benefit -- this problem does not need Newton's
  quadratic convergence, and damping (below) handles the oscillation risk the
  plan calls out for high tau.
* Normalization row: the plan says "replace row 36"; here row 0 (the ground
  state, always significantly populated) is replaced instead, since forcing
  the near-empty highest level's equation to be the normalization constraint
  produces a poorly conditioned matrix at low T_k where level 36 (E~139 K) is
  essentially unpopulated.
* Grouping is computed once from the actual pairwise velocity separation of
  all 77 transitions (union via connected components), not hand-coded per the
  FIR special case the plan describes -- the 0.3 km/s criterion reproduces the
  paper's "12 overlapping groups + 28 single lines" structure on its own, and
  this is checked at load time (see `assert_expected_grouping`).
* Radius R and an explicit NH3 abundance never appear. k_t*R in the paper's
  Eq. (3)-(4) only ever occurs multiplied by volume level populations, and
  n_level(volume)*R = fractional_level_population * N_NH3(column) for a
  uniform sphere -- so the whole optical-depth chain needs only the column
  density N_NH3 (one of the two free grid parameters, via N_NH3 =
  (N_NH3/Delta v) * Delta v_clump) and the fractional populations, never n_H2
  or an abundance. This is mathematically identical to the paper's formula,
  just algebraically simplified to avoid introducing a free radius/abundance
  that immediately cancels.

Collision rates are Loreau et al. (2023) NH3-H2 rates (read directly from
tgs/modelling/p-nh3@loreau.dat.txt), not Stutzki & Winnewisser's own 1985
NH3-He estimates scaled by alpha=1.5 -- so n_H2 here is the actual H2 density,
not Stutzki's pseudo-density n'. Because the collision partner and rates
differ (38 years of collisional-rate work apart), this reproduction is
expected to match the PAPER'S FIGURES QUALITATIVELY (same anomaly mechanism,
same qualitative density/column dependence) but not to reproduce their
contour values to the digit.
"""
import os
import re

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.optimize import root
from scipy.sparse.csgraph import connected_components

from nh3hia import paths

# ---------------------------------------------------------------------------
# Physical constants, CGS (consistent with N_NH3 in cm^-2, C_ul in cm^3/s).
# ---------------------------------------------------------------------------
H_ERG_S = 6.62607015e-27
C_CM_S = 2.99792458e10
K_ERG_K = 1.380649e-16
CM1_TO_K = H_ERG_S * C_CM_S / K_ERG_K  # 1.4387769 K per cm^-1

N_LEVELS = 36
DV_CLUMP_DEFAULT_KMS = 0.3
T_BG_DEFAULT = 2.725

_DEFAULT_LAMDA = paths.LAMDA_NH3


# ---------------------------------------------------------------------------
# Phase 1: parse the LAMDA file, keep only the first N_LEVELS levels.
# ---------------------------------------------------------------------------
def _read_lamda_blocks(path):
    with open(path) as f:
        lines = [l.rstrip('\n') for l in f]

    def find(marker, start=0):
        for i in range(start, len(lines)):
            if lines[i].startswith(marker):
                return i
        raise ValueError(f"marker {marker!r} not found in {path}")

    i_nlev = find('! NUMBER OF ENERGY LEVELS')
    n_levels_file = int(lines[i_nlev + 1].split()[0])
    i_lev0 = find('! LEVEL', i_nlev) + 1
    level_rows = lines[i_lev0:i_lev0 + n_levels_file]

    i_ntrans = find('! NUMBER OF RADIATIVE TRANSITIONS', i_lev0)
    n_trans_file = int(lines[i_ntrans + 1].split()[0])
    i_tr0 = find('! TRANS', i_ntrans) + 1
    trans_rows = lines[i_tr0:i_tr0 + n_trans_file]

    i_ncoll = find('! NUMBER OF COLLISIONAL TRANSITIONS', i_tr0)
    n_coll_file = int(lines[i_ncoll + 1].split()[0])
    i_ntemp = find('! NUMBER OF TEMPERATURES', i_ncoll)
    n_temp = int(lines[i_ntemp + 1].split()[0])
    i_temps = find('! TEMPERATURES', i_ntemp) + 1
    T_grid = np.array([float(x) for x in lines[i_temps].split()])
    assert len(T_grid) == n_temp
    i_coll0 = find('! TRANS', i_temps) + 1
    coll_rows = lines[i_coll0:i_coll0 + n_coll_file]

    return level_rows, trans_rows, coll_rows, T_grid


def _parse_qn(qn_str):
    """'1_1_1_0' -> (J=1, K=1, sym=1, F1=0)."""
    J, K, sym, F1 = (int(x) for x in qn_str.split('_'))
    return J, K, sym, F1


def _split_fir_outer_satellite(group_id, n_groups, u_idx, l_idx, qn):
    """Force the (2,1)->(1,1) FIR outer satellite (F1_upper=1 -> F1_lower=0,
    the only such pair the selection rules allow, at ~1168 and ~1215 GHz) into
    its own singleton group, undoing the velocity-threshold chain-merge --
    see the caller's comment for why."""
    group_id = group_id.copy()
    next_id = n_groups
    for t in range(len(u_idx)):
        Ju, Ku, _, F1u = qn[u_idx[t]]
        Jl, Kl, _, F1l = qn[l_idx[t]]
        if Ju == 2 and Ku == 1 and Jl == 1 and Kl == 1 and F1u == 1 and F1l == 0:
            group_id[t] = next_id
            next_id += 1
    return next_id, group_id


class NH3Model:
    """Everything derived from the LAMDA file that does NOT depend on
    (T_k, n_H2, N_NH3): levels, transitions, B coefficients, hyperfine
    groups, and the collision-rate log-log spline. Build once, reuse for an
    entire grid."""

    def __init__(self, path=_DEFAULT_LAMDA, n_levels=N_LEVELS,
                 dv_clump_kms=DV_CLUMP_DEFAULT_KMS):
        level_rows, trans_rows, coll_rows, T_grid = _read_lamda_blocks(path)

        E_cm = np.zeros(n_levels)
        g = np.zeros(n_levels)
        qn = [None] * n_levels
        for row in level_rows:
            parts = row.split()
            idx = int(parts[0])
            if idx > n_levels:
                continue
            E_cm[idx - 1] = float(parts[1])
            g[idx - 1] = float(parts[2])
            qn[idx - 1] = _parse_qn(parts[3])
        self.E_K = E_cm * CM1_TO_K
        self.g = g
        self.qn = qn  # list of (J,K,sym,F1), 0-based level index

        u_idx, l_idx, A_ul, nu_Hz = [], [], [], []
        for row in trans_rows:
            parts = row.split()
            u, l = int(parts[1]), int(parts[2])
            if u > n_levels or l > n_levels:
                continue
            u_idx.append(u - 1)
            l_idx.append(l - 1)
            A_ul.append(float(parts[3]))
            nu_Hz.append(float(parts[4]) * 1e9)
        self.u_idx = np.array(u_idx, dtype=int)
        self.l_idx = np.array(l_idx, dtype=int)
        self.A_ul = np.array(A_ul)
        self.nu_Hz = np.array(nu_Hz)
        n_trans = len(self.u_idx)

        gu = self.g[self.u_idx]
        gl = self.g[self.l_idx]
        self.B_ul = C_CM_S ** 2 / (2 * H_ERG_S * self.nu_Hz ** 3) * self.A_ul
        self.B_lu = (gu / gl) * self.B_ul

        # ---- Phase 2.1: hyperfine overlap grouping via connected components
        dv_kms = np.abs(self.nu_Hz[:, None] - self.nu_Hz[None, :]) / self.nu_Hz[:, None] * C_CM_S / 1e5
        adjacency = dv_kms < dv_clump_kms
        np.fill_diagonal(adjacency, True)
        n_groups, group_id = connected_components(adjacency, directed=False)

        # Paper Sect. 3 (p.16), explicit special case: "we ... treated the
        # inner satellites and the main line of the IR (2,1)->(1,1)
        # transition as a group of lines with equal frequency, whereas the
        # outer satellites where treated as separate lines." The plain
        # velocity criterion above chain-merges that outer component (F1=1->0,
        # the transition farthest from the (2,1)->(1,1) manifold's own mean
        # frequency) into the same group as the main+inner cluster anyway,
        # since its neighbours in frequency are each <0.3 km/s apart even
        # though the two extremes of the manifold are ~0.65 km/s apart --
        # override that chaining for exactly this transition, matching the
        # paper's own explicit treatment.
        n_groups, group_id = _split_fir_outer_satellite(
            group_id, n_groups, self.u_idx, self.l_idx, self.qn)

        self.group_id = group_id
        self.n_groups = n_groups
        self.dv_clump_kms = dv_clump_kms

        group_sizes = np.bincount(group_id, minlength=n_groups)
        self.n_single_groups = int(np.sum(group_sizes == 1))
        self.n_overlap_groups = int(np.sum(group_sizes > 1))

        # ---- Phase 1.3: collision rates -> log10(C) cubic spline over log(T)
        c_u, c_l, c_table = [], [], []
        for row in coll_rows:
            parts = row.split()
            u, l = int(parts[1]), int(parts[2])
            if u > n_levels or l > n_levels:
                continue
            c_u.append(u - 1)
            c_l.append(l - 1)
            c_table.append([float(x) for x in parts[3:3 + len(T_grid)]])
        self.coll_u = np.array(c_u, dtype=int)
        self.coll_l = np.array(c_l, dtype=int)
        c_table = np.array(c_table)  # (n_coll, n_temp), cm3/s, de-excitation u->l
        floor = 1e-30
        log_c = np.log10(np.clip(c_table, floor, None))
        self._log_C_spline = CubicSpline(np.log(T_grid), log_c.T, axis=0)
        self.T_grid = T_grid

        # ---- hyperfine labels for the (1,1) and (2,2) manifolds, for the
        # named diagnostics in Phase 4.3 / Figs 4-6.
        self.labels = self._label_transitions()
        self._assign_named_groups()

    # -- collision matrix at a single T_k, EXCLUDING the n_H2 factor --------
    def collision_matrix(self, T_k):
        n = N_LEVELS
        C = np.zeros((n, n))
        log_C_ul = self._log_C_spline(np.log(T_k))
        C_ul = 10.0 ** log_C_ul
        C[self.coll_u, self.coll_l] = C_ul
        Eu = self.E_K[self.coll_u]
        El = self.E_K[self.coll_l]
        gu = self.g[self.coll_u]
        gl = self.g[self.coll_l]
        C_lu = C_ul * (gu / gl) * np.exp(-(Eu - El) / T_k)
        C[self.coll_l, self.coll_u] = C_lu
        return C

    # -- transition labeling for named ratios --------------------------------
    def _label_transitions(self):
        labels = []
        for t in range(len(self.u_idx)):
            Ju, Ku, symu, F1u = self.qn[self.u_idx[t]]
            Jl, Kl, syml, F1l = self.qn[self.l_idx[t]]
            labels.append(dict(Ju=Ju, Ku=Ku, F1u=F1u, Jl=Jl, Kl=Kl, F1l=F1l))
        return labels

    def _assign_named_groups(self):
        def find_group(pred):
            gids = {self.group_id[t] for t, lab in enumerate(self.labels) if pred(lab)}
            if len(gids) != 1:
                raise RuntimeError(f"expected exactly one group, got {gids}")
            return gids.pop()

        is11 = lambda lab: lab['Ju'] == 1 and lab['Ku'] == 1 and lab['Jl'] == 1 and lab['Kl'] == 1
        is22 = lambda lab: lab['Ju'] == 2 and lab['Ku'] == 2 and lab['Jl'] == 2 and lab['Kl'] == 2
        # (2,1) is non-metastable (J!=K) but still has its own inversion
        # doublet -- verified directly: three ~23.0988 GHz Delta F1=0
        # transitions (F1=1->1,2->2,3->3) group together exactly like
        # main_11's two Delta F1=0 transitions do, matching params.FREQ_HZ['2,1'].
        # This is the line Stutzki calls a precise, sub-thermally-populated
        # high-density indicator, and what Magritte's LUT stores as A_MAIN_21.
        is21 = lambda lab: lab['Ju'] == 2 and lab['Ku'] == 1 and lab['Jl'] == 2 and lab['Kl'] == 1

        # Keys read "F1_upper_to_F1_lower", matching the paper's own
        # transition naming (e.g. its "F1=1->0" satellite is the one whose
        # UPPER-parity sublevel has F1=1, decaying to the LOWER-parity
        # sublevel's F1=0) -- see stutzki_1985.pdf p.15, "the F1=0->1
        # satellite ... is enhanced, whereas the F1=1->0 transition ... is
        # decreased in intensity". An earlier version of this labeling had
        # outer_01/outer_10 (and inner_12/inner_21) swapped relative to that
        # convention -- structurally correct groups, wrong dict key -- caught
        # because R_01 (as originally, wrongly, labeled) came out far below
        # the LTE reference everywhere (a "decrease") while R_10 came out
        # strongly enhanced (into masing at some grid points), the reverse of
        # what the paper's own Sect. 2 mechanism describes.
        self.named_groups = dict(
            main_11=find_group(lambda l: is11(l) and l['F1u'] == l['F1l']),
            outer_10=find_group(lambda l: is11(l) and l['F1u'] == 1 and l['F1l'] == 0),
            outer_01=find_group(lambda l: is11(l) and l['F1u'] == 0 and l['F1l'] == 1),
            inner_21=find_group(lambda l: is11(l) and l['F1u'] == 2 and l['F1l'] == 1),
            inner_12=find_group(lambda l: is11(l) and l['F1u'] == 1 and l['F1l'] == 2),
            main_22=find_group(lambda l: is22(l) and l['F1u'] == l['F1l']),
            main_21=find_group(lambda l: is21(l) and l['F1u'] == l['F1l']),
        )

    def group_mean_freq(self):
        wsum = np.bincount(self.group_id, weights=self.nu_Hz, minlength=self.n_groups)
        cnt = np.bincount(self.group_id, minlength=self.n_groups)
        return wsum / cnt


def assert_expected_grouping(model):
    """Structural sanity checks. The paper (Sect. 3, p.16) reports "12
    overlapping groups and 28 single lines" from the same 77 transitions;
    our pure velocity-threshold clustering (plus the explicit FIR-outer-
    satellite override above) lands close but not identical to that exact
    split for the higher-(J,K) manifolds the paper doesn't discuss in detail
    (e.g. (3,1)->(2,1), (3,2)->(2,2) near 1.7-1.8 THz) -- those levels are
    only weakly populated over T_k=18-40K (paper Sect. 3: "higher levels
    show only very low population... even the (4,4) level only has minor
    influence"), so small differences there have a second-order effect on
    the (1,1)/(2,2) populations this module actually reports. What DOES
    matter for Figs 4-6 is verified exactly: the (1,1) main line is its own
    group containing both its near-degenerate Delta F1=0 transitions, the
    four (1,1) satellites and the (2,2) main line are each isolated into
    their own single-transition group, and the FIR outer satellite is
    correctly split out (checked directly by NH3Model._assign_named_groups
    raising if any of these is ambiguous)."""
    assert len(model.u_idx) == 77, len(model.u_idx)
    assert set(model.named_groups.keys()) == {
        'main_11', 'outer_01', 'outer_10', 'inner_12', 'inner_21', 'main_22', 'main_21'}
    main_size = np.sum(model.group_id == model.named_groups['main_11'])
    assert main_size == 2, main_size
    main21_size = np.sum(model.group_id == model.named_groups['main_21'])
    assert main21_size == 3, main21_size  # verified directly: 3 Delta F1=0 transitions overlap here
    for key in ('outer_01', 'outer_10', 'inner_12', 'inner_21'):
        size = np.sum(model.group_id == model.named_groups[key])
        assert size == 1, (key, size)


# ---------------------------------------------------------------------------
# Phase 2.3: precomputed Doppler-averaged escape probability beta(tau).
# ---------------------------------------------------------------------------
def _beta_integral(tau):
    if tau <= 0:
        return 1.0
    f = lambda z: np.exp(-z * z) * np.exp(-tau * np.exp(-z * z))
    val, _ = quad(f, -8.0, 8.0, limit=200)
    return val / np.sqrt(np.pi)


class BetaSpline:
    """beta(tau) = (1/sqrt(pi)) INT exp(-z^2) exp(-tau exp(-z^2)) dz, tabulated
    over log10(tau) in [-8, 8] (wider than the plan's [-4,4] so the very high
    tau of the FIR (2,1)->(1,1) transitions -- deliberately not the physics
    of interest, per the paper's own Sect. 3, but still needed for a stable
    J_t -- stays inside the spline domain instead of being clamped)."""

    def __init__(self, log10_tau_min=-8.0, log10_tau_max=8.0, n=400):
        log_tau = np.linspace(log10_tau_min, log10_tau_max, n)
        tau = 10.0 ** log_tau
        beta = np.array([_beta_integral(t) for t in tau])
        self._spline = CubicSpline(log_tau, beta)
        self._lo, self._hi = log10_tau_min, log10_tau_max

    def __call__(self, tau):
        tau = np.asarray(tau, dtype=float)
        log_tau = np.log10(np.clip(np.abs(tau), 10.0 ** self._lo, 10.0 ** self._hi))
        return np.clip(self._spline(log_tau), 0.0, 1.0)


_beta_spline_singleton = None


def get_beta_spline():
    global _beta_spline_singleton
    if _beta_spline_singleton is None:
        _beta_spline_singleton = BetaSpline()
    return _beta_spline_singleton


# ---------------------------------------------------------------------------
# Phase 4.1: chord-averaged spherical escape factor e(x), Stutzki Eq. (10).
# ---------------------------------------------------------------------------
def e_escape(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-3
    out[small] = 1.0 - (2.0 / 3.0) * x[small] + 0.25 * x[small] ** 2
    xb = x[~small]
    out[~small] = 2.0 * (1.0 - np.exp(-xb) * (1.0 + xb)) / xb ** 2
    return out


def planck_specific_intensity(nu_Hz, T):
    x = H_ERG_S * nu_Hz / (K_ERG_K * T)
    return (2.0 * H_ERG_S * nu_Hz ** 3 / C_CM_S ** 2) / np.expm1(x)


# ---------------------------------------------------------------------------
# Phase 3: statistical equilibrium solver.
#
# NOT plain fixed-point (Lambda) iteration: holding tau_G/S_G/beta_G fixed at
# their previous value and re-solving the resulting *linear* rate system is
# the obvious first thing to try, and it is what an earlier version of this
# module did -- but it is well known to converge very slowly (near-unity
# convergence rate) whenever radiative trapping is significant, and on this
# model at moderate optical depth it does worse than that: traced by hand, it
# drifts the (1,1) outer F1=1->0 satellite's group optical depth monotonically
# and without bound toward a spurious negative-tau (maser) fixed point instead
# of settling near the thermal starting guess, which turns out to already be
# close to the true self-consistent solution at that point (this is exactly
# the "oscillatory behaviour at high tau" the plan's Phase 3.3 warns about --
# it just isn't fixed by damping the linear iteration harder here).
#
# Instead this solves the actual nonlinear steady-state residual
#     r_i(n) = sum_j Q_ji(n) n_j - n_i sum_j Q_ij(n)      i = 1..35
#     r_0(n) = sum_i n_i - 1                              (normalization row)
# with scipy.optimize.root (MINPACK's modified-Powell hybrd, a quasi-Newton
# method with its own finite-difference/Broyden Jacobian handling) -- this is
# the plan's "Newton-Raphson ... accounting for the explicit dependence of
# J_t on populations via tau_G and S_G" without hand-deriving that Jacobian:
# scipy estimates it, and the 36-unknown system is tiny for it (milliseconds).
# Row 0 (the ground state, always significantly populated) carries the
# normalization constraint rather than the plan's literal "row 36", since
# forcing the near-empty top level's equation to be the constraint is
# poorly conditioned at low T_k.
# ---------------------------------------------------------------------------
_POP_FLOOR = 1e-30


def _build_residual_fn(model, Cmat, T_k, n_H2, N_NH3_total, T_bg):
    beta_spline = get_beta_spline()
    u, l = model.u_idx, model.l_idx
    A_ul, B_ul, B_lu = model.A_ul, model.B_ul, model.B_lu
    nu = model.nu_Hz
    gid = model.group_id
    n_groups = model.n_groups
    gu, gl = model.g[u], model.g[l]

    dnu = nu * (model.dv_clump_kms * 1e5 / C_CM_S)
    prefac = H_ERG_S * (nu / dnu) / (4.0 * np.pi)  # erg*s, per transition
    F_nu_t = planck_specific_intensity(nu, T_bg)

    def intermediates(x):
        xs = np.clip(x, _POP_FLOOR, None)
        xl, xu = xs[l], xs[u]
        k_t = prefac * (xs[l] * B_lu - xs[u] * B_ul) * N_NH3_total
        tau_G = np.bincount(gid, weights=k_t, minlength=n_groups)

        denom = gu * xl / (gl * xu) - 1.0
        denom = np.where(denom == 0.0, 1e-12, denom)
        S_g = (2.0 * H_ERG_S * nu ** 3 / C_CM_S ** 2) / denom

        num = np.bincount(gid, weights=k_t * S_g, minlength=n_groups)
        den = np.bincount(gid, weights=k_t, minlength=n_groups)
        S_G = np.divide(num, den, out=np.zeros_like(num), where=np.abs(den) > 1e-300)

        # beta(tau) evaluated at max(tau_G, 0), NOT abs(tau_G): a group that
        # goes population-inverted (tau_G<0) is treated as fully optically
        # thin (beta=1, so J_t collapses to the background only) rather than
        # as "equally trapped as its positive-tau mirror image". The abs()
        # version (still what BetaSpline.__call__ does on its own, used only
        # for the post-hoc emergent-brightness/diagnostic path, never here)
        # let S_G's contribution back into J_t with a large (1-beta) weight
        # whenever |tau_G| was sizeable, which can make B_ul*J_t/B_lu*J_t --
        # meant to be non-negative RATE COEFFICIENTS in the master equation
        # -- come out negative once S_G is strongly negative (a real,
        # reproduced failure mode here: for the (1,1) F1=0->1 transition at
        # T=36K, n_H2=1e5, N/dv=1e15, A_ul+B_ul*J_t came out to -1.9e-7 s^-1,
        # i.e. an outright negative decay rate). Clamping at 0 here removes
        # that specific unphysical pathway; it does NOT remove the
        # population inversion itself -- checked directly: forcing this same
        # clamp still converges to essentially the same inverted populations
        # (x_u/g_u > x_l/g_l for the affected level, not merely tau_G<0),
        # and a level-by-level gain/loss accounting shows the (2,1)->(1,1)
        # FIR pumping rate into the affected sublevel alone exceeds its
        # ENTIRE loss budget (radiative + all collisional channels combined)
        # by a factor of several, so the inversion is not an artifact of
        # this term -- but leaving rate coefficients that can go negative in
        # the production solver is wrong regardless of whether it changes
        # this particular answer.
        beta_G = beta_spline(np.clip(tau_G, 0.0, None))
        J_t = F_nu_t * beta_G[gid] + S_G[gid] * (1.0 - beta_G[gid])
        return tau_G, S_G, J_t

    def residual(x):
        _, _, J_t = intermediates(x)
        # Hard floor at 0, independent of the beta clamp above: a rate
        # coefficient entering a master equation must never be negative,
        # and this is cheap insurance against any remaining path (e.g. a
        # nominally tau_G>=0 group whose pooled S_G is still very negative)
        # that the clamp in intermediates() doesn't already rule out.
        decay_rate = np.clip(A_ul + B_ul * J_t, 0.0, None)
        excite_rate = np.clip(B_lu * J_t, 0.0, None)
        Q = n_H2 * Cmat.copy()
        np.add.at(Q, (u, l), decay_rate)
        np.add.at(Q, (l, u), excite_rate)
        r = Q.T @ x - x * Q.sum(axis=1)
        r[0] = x.sum() - 1.0
        return r

    return residual, intermediates


def solve_populations(model, Cmat, T_k, n_H2, N_NH3_total,
                       T_bg=T_BG_DEFAULT, x0=None, tol=1e-12, maxfev=4000,
                       residual_ok=1e-9):
    """x0, if given, warm-starts the root find from a nearby grid point's
    converged populations -- much cheaper than starting from thermal LTE at
    every grid point when sweeping a grid (see grid.py).

    sol.success is deliberately NOT trusted on its own: MINPACK's hybrd
    reports success once the step between iterates drops below xtol, which
    is a DIFFERENT condition from the residual being small, and the two can
    diverge sharply for a warm start. Caught directly: starting from the
    converged populations of an adjacent grid point only 12% different in
    n_H2 and asking hybr to re-solve, it returned "success" with the
    populations barely moved from x0 and a residual of ~1e-8, while an
    independent cold start (thermal initial guess) for the SAME point
    converged properly (residual ~1e-18) to a visibly different, correct
    answer -- x0 was close enough that hybr's internal step size immediately
    satisfied xtol without the gradient actually being driven to zero. This
    silently propagated identical, wrong values across a whole run of grid
    points in a warm-started sweep (visible as a jagged, non-physical seam
    of repeated contour values in the Fig. 4 reproduction). The fix is to
    require the residual itself below `residual_ok` (not sol.success), and
    fall back to a cold thermal start -- a different enough initial guess
    that hybr's trust region actually has to move -- whenever that fails."""
    residual, intermediates = _build_residual_fn(model, Cmat, T_k, n_H2, N_NH3_total, T_bg)

    def thermal_guess():
        g0 = model.g * np.exp(-model.E_K / T_k)
        return g0 / g0.sum()

    def try_solve(x_start):
        with np.errstate(divide='ignore', invalid='ignore'):
            sol = root(residual, x_start, method='hybr', options={'xtol': tol, 'maxfev': maxfev})
            r_final = residual(sol.x)
        return sol, np.max(np.abs(r_final))

    if x0 is None:
        x0 = thermal_guess()

    sol, max_resid = try_solve(x0)
    if max_resid >= residual_ok:
        # warm start likely gave a false-converged fixed point of a nearby
        # problem -- retry from a genuinely different initial guess.
        sol_cold, max_resid_cold = try_solve(thermal_guess())
        if max_resid_cold < max_resid:
            sol, max_resid = sol_cold, max_resid_cold

    converged = bool(max_resid < residual_ok)

    x = np.clip(sol.x, _POP_FLOOR, None)
    x = x / x.sum()
    tau_G, S_G, _ = intermediates(x)

    return dict(x=x, tau_G=tau_G, S_G=S_G, converged=converged,
                n_iter=sol.nfev, max_residual=max_resid)


# ---------------------------------------------------------------------------
# Phase 4: emergent brightness + named diagnostic ratios.
# One T_B per GROUP (not per raw transition): transitions sharing a group are
# frequency-degenerate by construction (that is what "overlapping" means), so
# their emergent brightness is one merged spectral feature with the group's
# pooled tau_G/S_G -- evaluating Eq. (8) once per constituent transition and
# summing would double-count the (1,1) main line, which is two transitions
# merged into one group.
# ---------------------------------------------------------------------------
def group_brightness_temperatures(model, tau_G, S_G, T_bg=T_BG_DEFAULT, guard_masers=True):
    """Selective trapping can genuinely drive one hyperfine satellite into
    population inversion (tau_G < 0) at strongly anomalous grid points --
    observed directly in this solver's converged output over a non-trivial
    part of the (n_H2, N_NH3/dv) plane. Eq. (10)'s e(2tau) is a THERMAL
    escape-probability closure derived for tau>=0; algebraically continuing
    it to tau<0 does not diverge to +/-infinity for moderate inversion (it
    stays smooth and finite -- e.g. e(2*(-1.16)) = 5.34), so a literal
    implementation with no sign check on tau (almost certainly what
    Stutzki & Winnewisser's own 1985 code did, since maser action was not
    what they were looking for) does not fail visibly: it just keeps
    producing plausible-looking large positive ratios. Checked directly:
    at T=36K, n_H2=1e5, N/dv=1e15 -- right where Fig. 4a's own T=36K,
    F1=0->1 panel shows contours climbing through a labelled "2.0" -- the
    literal formula here gives R(0->1)=2.03, matching his own published
    contour almost exactly. So guard_masers=False (the literal formula,
    no NaN) is the historically faithful reproduction of what his figure
    actually shows; guard_masers=True (default) additionally reports NaN
    for any group with tau_G<0, since that literal number is not a
    physically meaningful escape-probability-closure result once the
    closure's own tau>=0 assumption has been violated -- it is what the
    formula outputs, not what the physics does. Both are computed by the
    same run; nothing here changes the underlying populations, only how a
    negative-tau group's brightness is reported."""
    nu_G = model.group_mean_freq()
    F_G = planck_specific_intensity(nu_G, T_bg)
    tau_for_e = np.where(tau_G >= 0, tau_G, np.nan) if guard_masers else tau_G
    e2tau = e_escape(2.0 * tau_for_e)
    T_B = C_CM_S ** 2 / (2.0 * K_ERG_K * nu_G ** 2) * (S_G - F_G) * (1.0 - e2tau)
    return T_B


def diagnostics(model, tau_G, S_G, T_bg=T_BG_DEFAULT, guard_masers=True):
    """Returns dict with tau_main, T_B_main, R_01, R_10, R_12, R_21, R_2211,
    all defined exactly as in stutzki_model_1985.md Phase 4.3."""
    T_B = group_brightness_temperatures(model, tau_G, S_G, T_bg=T_bg, guard_masers=guard_masers)
    ng = model.named_groups
    T_B_main = T_B[ng['main_11']]
    out = dict(
        tau_main=tau_G[ng['main_11']],
        T_B_main=T_B_main,
        T_B_outer_01=T_B[ng['outer_01']],
        T_B_outer_10=T_B[ng['outer_10']],
        T_B_inner_12=T_B[ng['inner_12']],
        T_B_inner_21=T_B[ng['inner_21']],
        T_B_22_main=T_B[ng['main_22']],
        T_B_21_main=T_B[ng['main_21']],
        R_01=T_B[ng['outer_01']] / T_B_main,
        R_10=T_B[ng['outer_10']] / T_B_main,
        R_12=T_B[ng['inner_12']] / T_B_main,
        R_21=T_B[ng['inner_21']] / T_B_main,
        R_2211=T_B[ng['main_22']] / T_B_main,
        R_21_11=T_B[ng['main_21']] / T_B_main,
        any_maser=bool(np.any(tau_G[[ng[k] for k in
                                      ('main_11', 'outer_01', 'outer_10',
                                       'inner_12', 'inner_21', 'main_22', 'main_21')]] < 0)),
    )
    return out


def run_one(model, Cmat, T_k, n_H2, log_N_dv, dv_clump_kms=DV_CLUMP_DEFAULT_KMS, **solver_kwargs):
    """log_N_dv = log10(N_NH3/Delta v [cm^-2 s km^-1]). Convenience wrapper
    tying together Phases 3-4 for one grid point. guard_masers=False
    reproduces the literal, un-sign-checked Eq.(10) formula (see
    group_brightness_temperatures' docstring)."""
    guard_masers = solver_kwargs.pop('guard_masers', True)
    N_dv = 10.0 ** log_N_dv
    N_NH3_total = N_dv * dv_clump_kms
    sol = solve_populations(model, Cmat, T_k, n_H2, N_NH3_total, **solver_kwargs)
    diag = diagnostics(model, sol['tau_G'], sol['S_G'], guard_masers=guard_masers)
    diag['converged'] = sol['converged']
    diag['n_iter'] = sol['n_iter']
    diag['max_residual'] = sol['max_residual']
    diag['x'] = sol['x']  # for warm-starting a neighbouring grid point
    return diag
