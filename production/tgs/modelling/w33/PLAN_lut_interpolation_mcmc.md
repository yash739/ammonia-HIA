# Plan: precomputed grid → interpolator → χ² landscapes → posterior

## Why

Every inversion so far pays full Magritte cost *inside* the search loop. That is
the wrong place to spend it while we are still benchmarking: the same forward
model gets re-run for each new source, each new observed ratio vector, each
change to the scoring function. A precomputed lookup table (LUT) over the
parameters the ratios actually depend on, interpolated on the fly, moves all the
cost to one bounded, parallel, resumable run — after which an inversion is
microseconds, χ² landscapes are free, and full MCMC posteriors become possible
at all (they need ~10⁵–10⁶ forward evaluations, which is flatly impossible with
Magritte in the loop).

Four phases, each with its own verification. Phase 2's validity tests are the
ones that decide whether any of this is legitimate, so they are not optional.

---

## Phase 1 — the precomputed grid

### What the axes actually are

`run_model` takes `(XNH3, numberdensity, vturb, T_cloud, radius_sphere)`, but the
observables don't depend on all five independently. `X_NH3` and `radius_sphere`
enter *only* through the column `N_NH3 = 2·r·n·X`, so they are exactly
degenerate — one axis, not two. The physically distinct axes are:

| Axis | Range | Spacing | Points |
|---|---|---|---|
| `log₁₀ n_H2` [cm⁻³] | 3.0 → 8.0 | 0.25 dex | 21 |
| `T_k` [K] | 10 → 60 | 5 K | 11 |
| `log₁₀ (N_NH3/Δv)` [cm⁻²/(km/s)] | 13.5 → 16.0 | 0.25 dex | 11 |
| `Δv` [km/s] | {0.3} tier 1; {0.15, 0.3, 0.6} tier 2 | — | 1 → 3 |

**Tier 1: 21 × 11 × 11 = 2,541 models** at Δv = 0.3 km/s.

Range justification, not guesswork:
- `n_H2` upper bound of 10⁸ because Stutzki's own Table 1a fits reach log n' =
  7.576 (OMC S1) and our OMC S4 benchmark sits at 7.317 — *above* the 10^6.5
  ceiling of the existing `parallel_v2` catalogue, which is exactly why that
  catalogue could never have seeded this benchmark correctly.
- `T_k` 10–60 K brackets Stutzki's own 18–40 K grid with room for the warmer
  solutions our own fits have wandered into (up to ~64 K).
- `N/Δv` brackets Stutzki's 10^14.2–10^15.6 with a margin.

### What the existing `parallel_v2` catalogue teaches (878 rows, all SUCCESS)

Inspected directly. It got the density axis roughly right and the rest wrong:
- `log n_H2`: 25 points, 3.5→6.5, **0.103 dex** spacing — good resolution, but
  the ceiling is 1.5 dex below where the OMC benchmarks live.
- `T_cloud`: **only 3 values (18, 36, 54 K)** — an 18 K step. Hopeless for
  interpolation in T; this is the single biggest gap to fix.
- `vturb`: 2 values. `X_NH3`: 1 value (1e-8, fine — it's degenerate anyway).
- 72.3% of rows reach ≥90% convergence; **17.9% hit the 300-iteration cap**,
  and `tau_main` spans 0.0099 → 57.4. Cost and reliability are both strongly
  τ-dependent, which drives the sampling design below.

The lesson: trade the over-resolved density axis (0.103 dex) for a real
temperature axis. 0.25 dex in n is still finer than Stutzki's own effective
resolution, and Phase 2's tests will tell us empirically whether it's enough
rather than us guessing.

### Cost control

Per-model wall time is τ-dependent and must be **measured, not assumed** —
Phase 1 starts with a stratified timing probe of ~40 points spanning the
(τ, T) plane, from which the full-grid cost is projected before committing.
The current OMC S4 recovery run (125 models, 14 processes) supplies the first
real datapoint for this.

Reuse the established `multiprocessing.Pool(processes=14, maxtasksperchild=1)`
+ incremental-CSV pattern: resumable, auditable row-by-row, and a killed run
loses at most one batch.

### Storage — spectra, not FITS cubes

**Store the 1-D spectra per grid point, not just the derived ratios**, and not
FITS cubes. If only 5 ratios are stored, the LUT is frozen against the current
observable set — we could never add the (2,1) line, change the peak extractor,
or move from peak-ratios to full-profile fitting without recomputing
everything. A 500-channel spectrum for each imaged line is ~12 KB per model, so
all 2,541 models fit in **~30 MB** in one compressed `.npz`/HDF5. FITS cubes
for a grid this size would be hundreds of GB and nothing reads them back.

Alongside the spectra, store per point: the 6 amplitudes, the 5 ratios,
`tau_main`, `halting_iter`, `final_convergence`, and wall time.

---

## Phase 2 — interpolation, and proving it's valid

### What to interpolate

**Interpolate `log`-amplitudes, then form ratios afterward** — not the ratios
directly. Two reasons: amplitudes are smoother functions of the parameters than
ratios are (a ratio steepens badly wherever its denominator gets small), and the
likelihood in Phase 4 wants amplitudes anyway. Also interpolate `log tau_main`,
and `final_convergence` so the sampler can *know* when it is wandering into a
region the grid itself didn't converge in.

### Method

`scipy.interpolate.RegularGridInterpolator` on the regular grid, cubic where the
error map allows and linear elsewhere. Chosen over GP/RBF for one decisive
reason: MCMC needs ~10⁶ vectorized evaluations, which `RegularGridInterpolator`
does in seconds and a GP over 10⁴ points does not. The one thing a GP would have
given for free — an interpolation-uncertainty estimate, which Phase 4's
likelihood genuinely needs — we get instead from the empirical error map below.

### Validity tests (the part that decides whether this is legitimate)

1. **Hold-one-out, on the existing grid — zero new compute.** For every interior
   grid point, rebuild the interpolator without it and predict it. Yields a
   per-observable, per-region *empirical error map*. Cheapest and highest-value
   test; also directly feeds `σ_interp(θ)` into the Phase 4 likelihood.
2. **Grid-halving (Richardson-style) — zero new compute.** Build the
   interpolator on every *other* grid point and compare against the full grid's
   true values. This is the test that literally answers "at what density is
   interpolating valid": if halving the spacing changes the answer by far less
   than the observational uncertainty, the grid is dense enough — and if not, we
   know exactly which axis needs refinement.
3. **Off-grid audit — ~50 new models.** Tests 1 and 2 both interpolate *from*
   grid points, so they share any systematic the grid geometry induces. Running
   ~50 forward models at random *non*-grid parameter values is the only check
   that catches that. Small, bounded cost.
4. **Acceptance criterion, stated in advance**: interpolation error must be
   below ~⅓ of the observational ratio uncertainty, so it contributes <10% in
   quadrature. Regions failing this get Tier-2 local refinement (halve spacing
   in the offending axis) rather than a blanket global refinement.

**Where to expect failures** (worth predicting up front so we don't rationalize
afterward): near the τ ≈ 1 transition, where line-trapping behavior changes
character; and wherever a line drops below the numerical noise floor — this
session already found the (2,1) line does exactly that in W33's regime. The
error map from test 1 should show both. If it shows large errors somewhere we
*didn't* predict, that is a signal to investigate the physics, not to add grid
points until it goes away.

---

## Phase 3 — χ² landscapes

Once the interpolator exists this is nearly free, and it replaces "here is the
best-fit row" with something far more honest.

- **Profile-χ² maps**: for each (n_H2, T_k) cell, minimize over `N/Δv` → a
  contour map directly comparable to Stutzki's own Fig. 4. Same for the other
  axis pairs.
- **Show the degeneracy rather than resolving it silently.** The high-density /
  low-density double minimum this pipeline keeps hitting is *two basins on this
  map*. Overplot the η_f ≤ 1 allowed region and the Jeans-mass clump-count
  K ≥ Δv_obs/Δv allowed region, and the degeneracy-breaking becomes something
  you can see and check, instead of a boolean returned by a function.
- Regenerate Stutzki's Fig. 4/5/6 from our LUT and overlay his published
  contours — a far stronger validation than the current trend-level comparison,
  and it costs nothing extra once the LUT exists.

---

## Phase 4 — MCMC and the likelihood problem

The user flagged the likelihood as possibly non-trivial. It is, and here is
specifically why the obvious choice is wrong.

### Why the current `weighted_chi2` is not a likelihood

It treats the 5 ratios as independent Gaussians. They are not, for three
separate reasons:

1. **Shared denominator.** All five ratios divide by `A_MAIN`. A fluctuation in
   `A_MAIN` moves all of them together, so their errors are strongly positively
   correlated. Treating them as independent over-counts the information and the
   posterior comes out too tight — spuriously confident error bars are worse
   than none.
2. **Ratios of noisy quantities aren't Gaussian.** A ratio whose denominator has
   non-negligible noise follows a Marsaglia/Fieller distribution with heavy
   tails. For high S/N `A_MAIN` the Gaussian approximation is fine; this must be
   *checked per source* against the actual `A_MAIN` S/N, not assumed.
3. **The amplitudes are jointly fitted.** All six come from one simultaneous
   5-Gaussian fit to a single spectrum, so `curve_fit`'s covariance matrix is
   dense — partially blended components have correlated amplitude errors.

### Two framings, and why to build both

**Framing A — Stutzki-faithful, ratio space with the correct covariance.**
Keep the dilution-independence that motivated using ratios at all, but propagate
the fit covariance properly via the Jacobian `J_ij = ∂r_i/∂A_j`:

```
C_r = J C_fit Jᵀ            (dense, not diagonal)
−2 ln L = (r_obs − r_model(θ))ᵀ C_r⁻¹ (r_obs − r_model(θ)) + ln|2πC_r|
```

Cheap, no nuisance parameters, and it cross-checks directly against the existing
χ² pipeline. It discards the absolute-brightness information by construction.

**Framing B — fit absolute amplitudes, η_f as a nuisance parameter.**

```
A_pred(θ, η_f) = η_f · A_model(θ),     θ = (T_k, n_H2, N/Δv)
−2 ln L = [A_obs − η_f A_model(θ)]ᵀ C⁻¹ [A_obs − η_f A_model(θ)] + ln|2πC|
C = C_fit + C_interp(θ) + C_model
```

with priors `0 < η_f ≤ 1` (Stutzki's hard physical ceiling) and the Jeans-mass
constraint `K(θ, η_f) ≥ Δv_obs/Δv` as an indicator prior.

**This is strictly stronger than Stutzki's own two-step method**, and it is the
main methodological gain on offer here: he derives η_f *after* fitting and then
rejects solutions that violate η_f ≤ 1. Framing B puts the same physics
*inside* the inference — the absolute brightness contributes real information,
the filling factor is properly marginalized over rather than point-estimated,
and the degeneracy is broken by the posterior itself instead of by a post-hoc
filter. Bonus: η_f enters linearly, so with a truncated-flat prior it can be
marginalized **analytically** (a truncated-Gaussian integral in terms of `erf`),
which removes a dimension from the sampling problem.

Build A first (baseline, validates against existing machinery), then B. **The
difference between the two posteriors is itself the result** — it quantifies
whether absolute brightness actually helps break the degeneracy, which is
precisely the open question from the earlier `tau_main`-rescaling experiment
that gave an ambiguous answer.

### Interpolation error must enter the likelihood

`C_interp(θ)` comes straight from Phase 2's hold-out error map. Omit it and the
posterior is over-confident wherever the grid is coarse — this is the concrete
reason Phase 2's tests aren't optional bookkeeping.

### Sampler: dynesty primary, emcee cross-check

**Nested sampling (dynesty) is the right primary choice here specifically
because our posterior is known to be multimodal** — the high-density/low-density
degeneracy is literally two modes, and ensemble samplers like emcee are
notoriously unreliable at moving between separated modes. Nested sampling
handles multimodality natively *and* returns the Bayesian evidence, which lets
us compare the two branches quantitatively (an evidence ratio) instead of
asserting that one is right. Run `emcee` as an independent cross-check on the
dominant mode.

### Validating the posterior itself

A single recovery test (like the OMC S4 run now in flight) checks the point
estimate. It says nothing about whether the *error bars* are right. Once the LUT
exists, the rigorous version costs **zero Magritte compute**:

- **Coverage test / simulation-based calibration**: draw N sets of true
  parameters from the prior, generate synthetic observations with realistic
  noise, run the full inference on each, and check that the truth falls inside
  the x% credible interval x% of the time. Under-coverage means the likelihood
  is missing a variance term (almost certainly `C_interp` or `C_model`);
  over-coverage means we're being too conservative.

### Priors to state explicitly, not bury

Log-uniform on `n_H2` and `N/Δv` (scale parameters), uniform on `T_k`, uniform
on `η_f ∈ (0,1]`, plus the Jeans indicator. Log-uniform versus uniform on a
scale parameter genuinely changes the answer when the data are weak, so this
gets stated in any result, not hidden in a config file.

---

## Sequencing and what gates what

1. **Finish the OMC S4 recovery test** (in flight) — if the current
   search-in-the-loop pipeline can't recover a known answer, the LUT would just
   make a broken inversion faster. This gates everything.
2. **Phase 1 timing probe** (~40 models) → project full-grid cost → commit or
   re-scope the grid.
3. **Phase 1 full run** (~2,541 models, resumable).
4. **Phase 2 tests 1 & 2** (free) → error map → local refinement where needed →
   **test 3** (~50 models).
5. **Phase 3 landscapes** — also the natural point to regenerate Stutzki's
   Fig. 4/5/6 as a much stronger validation than we currently have.
6. **Phase 4**: Framing A → Framing B → coverage test → posteriors for the real
   W33 sources.

## Honest open risks

- **The absolute-brightness discrepancy is unresolved.** Our Magritte `A_MAIN`
  for OMC S4 is 3.86 K against Stutzki's `T_B_theor` = 10.71 K. The recovery
  test sidesteps this by building its synthetic observation self-consistently
  from our own model, which is correct *for testing the inversion* but means
  Framing B (which uses absolute amplitudes) inherits an unexplained factor
  before it is ever applied to real data. This needs resolving on its own terms
  before Framing B's posteriors on real sources mean anything.
- **`Δv` fixed at 0.3 km/s** throughout Tier 1, matching Stutzki — but his own
  Sect. 5(i) flags this as an assumption (turbulent fragmentation stopping at
  the sound speed), and broader clumps would lower the required density. Tier 2's
  Δv axis is the test, and it triples the grid cost, so it is deliberately
  deferred rather than dropped.
- **`X_NH3`/radius stay degenerate.** The LUT constrains `N_NH3/Δv`; converting
  that to an abundance or an angular size still needs an independent size
  constraint. The LUT doesn't fix this — it just stops us from pretending it's
  fixed.
