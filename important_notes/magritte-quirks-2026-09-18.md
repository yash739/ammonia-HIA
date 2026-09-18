# Session notes: 2026-09-18

Continuation of `magritte-quirks-2026-09-16.md` (imaging/mesh defects, the
radial-mesh fix). This file covers what changed today: a full per-position
retrieval diagnostic tool, a plotting bug found and fixed in that tool, a
policy change on the radius mask, and a parallel retrieval built on the 1D
escape-probability model with a genuine numerical finding of its own.

---

## 1. `tau_main` bug: causally confirmed, not just correlated

2026-09-16 established that `tau_main` is the optical depth of whichever line
was imaged *last* (Magritte's `compute_image_optical_depth_new` uses
whatever spectral grid `compute_spectral_discretisation` last set), based on
matching values. Today: **direct causal test** — same model, only the
imaging order changed:

| last line imaged | reported `tau_main` | that line's own τ |
|---|---|---|
| (2,2) | 0.13916 | 0.13916 |
| (2,1) | 0.01098 | 0.01098 |
| (1,1) | 0.42347 | 0.42347 |

Confirmed. Still not fixed in `nh3_NLTE_sphere.py` (gold is mid-build; fixing
the definition mid-run would mix two definitions in one column). Fig 4
equivalents below don't need `tau_main` at all (their axes are `log_n_H2` and
`log_N_dv` directly), so they're unaffected; anything using `tau_main` as an
axis (Figs 5/6) still needs the correction applied downstream.

### 1.1 Confirmed: re-imaging the same model twice gives slightly different τ

Re-derived while building the retrieval tool's spectrum panels — not
re-investigated further, still **unresolved** per the 09-16 note.

---

## 2. Full chi² retrieval diagnostic against Stutzki 1984/1985 — new tool

New: `w33/retrieve_stutzki1985_full.py`. For each of the 23 cross-checked
positions in `stutzki_params.RETRIEVAL_TEST_POSITIONS`, produces one PNG with:

1. Observed (1,1) spectrum reconstructed from Stutzki's reported `T_B_11` and
   four satellite ratios, at his own observed linewidth.
2. Best-fit model spectrum from the interpolated gold LUT, at the model's own
   intrinsic 0.3 km/s clump linewidth — plotted in absolute `T_B` **and** a
   second panel with both spectra divided by their own main-line peak, so
   ratio-shape agreement is visible independent of the η_f brightness
   mismatch (added mid-session on request).
3. Best-fit `(log_n_H2, T_k, log_N_dv)`, χ², η_f = T_B,obs/T_B,theor, K
   predicted vs required, and the `physically_consistent()` verdict.
4. Predicted (2,1)/(1,1) at the best fit (held out — never enters the fit),
   vs Stutzki's own theory/observation where Table 3 has them (3 of 23
   positions).
5. χ² landscape in all three 2D slices.

Reuses `lut_interpolator.py` (per the standing rule to never do
nearest-grid-point retrieval) and `stutzki_physics.py`'s η_f/K functions.
Output: `output_lut_gold/results/stutzki1985_retrieval/` (23 PNGs +
`summary.csv`).

### 2.1 Bug found and fixed: fixed 2D slices can hide a real branch

**Symptom**: asked "why isn't the low-density branch ever visible in the χ²
landscape panels?" First implementation cut each 2D panel through the
*global* best-fit point's coordinates on the third axis (e.g. `T_k` vs
`log(n_H2)` at the single `log_N_dv` value the global optimum happens to
want).

**Root cause, verified directly on OMC S4**: the low-density local minimum
sits at `log_N_dv=14.73`; the global best fit's own `log_N_dv=15.42`. A
slice through 15.42 literally does not pass near the low branch's true
minimum. Confirmed with a direct 3D scan: a genuine local minimum exists at
`log_n≈4.0-4.3`, χ²≈15-31, versus global χ²≈0.2-1.4 — 5-80× worse, but a
*real*, lower-than-all-6-neighbours local minimum, not an artefact.

**Fix**: switched every landscape panel to a **profile** (minimum-projected
over the third axis, not a fixed slice) — matches the profile-landscape
approach already used for the legacy-grid retrieval
(`legacy_grid_retrieval.tex`'s `chi2_landscapes` section). Also switched to
log-scaled colour (linear scale with a huge dynamic range made a χ²≈15-31
local minimum visually indistinguishable from the general background) and
added explicit local-minimum detection (4-neighbour, ranked, top 4 marked
with χ² labelled) so a secondary branch can't be missed by eye either.

**Lesson for any future landscape plot against this LUT**: always profile
over the third axis, never slice at a fixed value, unless the fixed value is
itself the point of the plot.

### 2.2 Policy change: radius mask off by default in `lut_interpolator.py`

Instructed: "ignore the radius mask, just keep converged>90% models."
`lut_interpolator.DEFAULT_MAX_RADIUS_PC` changed from `0.5` to `None`. The
0.5 pc corner (see 09-16 notes, 2.1) is still genuinely subcritically
thermalized and often unconverged, but `convergence_ok`
(`final_convergence>=90%`, `build_lut.CONV_THRESHOLD`) is what actually gates
data quality; the radius mask was an extra precaution on top of that, not a
substitute. Effect on the 23-position retrieval: two positions changed
branch. `W48 (0,40) 45km/s` moved from log n=6.7 (mask on) to log n=4.26
(mask off), now flagged `physically_consistent: False`. `S106 (0,-40)` now
has η_f=1.06 (>1, unphysical) — both now visibly real local-minimum
structure in their own plots rather than being silently excluded.

### 2.3 Final 23-position summary (mask off, profile plots)

Median χ²=0.447 (mean 1.92, pulled up by 3 poor fits: OMC S1 χ²=20.8, OMC S2
χ²=6.3, S106 (−40,0) χ²=4.6). Median |Δ log n| vs Table 1a = 0.58 dex.
87.0% pass `physically_consistent` (20/23) — the 3 failures (`W48
0,40_45kms`, `S106 90,30`, `S106 0,-40`) are exactly the 3 that land on the
low-density branch. 7/23 pin at the density ceiling (log n=7.5); 2/23 pin at
the temperature ceiling (48 K). The 3 positions with real (2,1) data all
still show the corroborating-not-resolving pattern from the 09-16/earlier
session: predicted (2,1)/(1,1) exceeds even Stutzki's own over-prediction at
all 3 (OMC S3: 0.081 vs theory 0.052 vs obs 0.015; OMC S4: 0.088 vs 0.068 vs
0.028; S106 (0,0): 0.068 vs 0.018 vs 0.014).

---

## 3. Fig. 4 equivalents, grid completeness, interpolation completeness — new tools

- `w33/reproduce_stutzki_fig4.py`: pointed at gold (was `output_lut_coarse`),
  `--temps` CLI added. Two outputs: the direct Stutzki-panel comparison
  (18/24/30/36 K) and all 14 gold temperatures. Contour axes are
  `(log_n_H2, log_N_dv)` only, so unaffected by the §1 `tau_main` bug.
  Radius mask explicitly left OFF per instruction (contours in this
  parameter space don't need it the way a τ-axis plot would).
- `w33/plot_grid_completeness.py` (new): per-temperature cell map against
  the full 14×14×9=1764 target, reusing `build_lut_gold`'s own axis
  constants so it can't drift from the real target grid. At the time of
  writing: 1263/1764 (71.6%) with the radius mask, 273 of the converged rows
  sit in the masked corner.
- `w33/plot_interpolation_completeness.py` (new): pooled 3D hold-one-out
  (807 interior points). Medians 1.1-2.0%, pass rate (< 3.3%, ⅓ of the ~10%
  observational floor) 82-98%.
- `w33/plot_interpolation_axis_completeness.py` +
  `plot_interpolation_axis_curves.py` (new): 1D hold-one-out along `T_cloud`
  and `log_n_H2` separately (other two axes fixed), plus representative
  curve plots (true vs leave-one-out, %-error annotated) at 4 slices per
  axis. T-axis: median 0.15-0.38%, pass 93.5-98.1% (1070 evals). log_n-axis:
  median 0.23-0.36%, pass 98.2-100% (1012 evals, but only 14/126 `T x
  log_Ndv` pairs have full 14-point density coverage — read as a
  best-case number on the well-sampled fraction of the grid, not a
  full-grid average).

All outputs moved into `output_lut_gold/results/` (not `scratch/`) — the
LUT's own directory is now the home for its validation products, matching
the existing `fits/images/results/spectra` layout.

---

## 4. 1D escape-probability model: extended with a (2,1) line, and a new numerical finding

Instructed to repeat the same retrieval exercise on `stutzki85/` (the 1D,
Stutzki's-own-method model) instead of gold, with the masing guard held off.

### 4.1 Added a `(2,1)` named group — verified

`nh3_escape_model.py` only tracked `main_11`/`main_22` and the four (1,1)
satellites; no `(2,1)` inversion line, needed for the same held-out
(2,1)/(1,1) prediction the gold retrieval makes. Checked directly: (2,1) is
non-metastable (J≠K) but still has its own inversion doublet — three
ΔF1=0 transitions (F1=1→1, 2→2, 3→3) at ≈23.0988 GHz group together exactly
like `main_11`'s two ΔF1=0 transitions do, matching
`params.FREQ_HZ['2,1']` exactly. Added as `main_21`; `assert_expected_grouping`
extended to check group size 3. `diagnostics()` now also returns
`T_B_21_main` and `R_21_11 = T_B_21_main/T_B_main`, plus the raw `T_B_*` for
every named group (previously only ratios were returned) so amplitude-level
interpolation/retrieval is possible.

### 4.2 `guard_masers=False` produces genuine numerical poles — new finding

Built the full grid on gold's exact axis values (1764 points, 15.6s total —
confirms the model's own docstring cost estimate) with `guard_masers=False`
as instructed. Result: **618/1695 converged rows have `T_B_outer_10<0`**,
and `R_01` reaches **1.9×10¹³** at one point (`log_n=4.25, T=48, log_Ndv=15.8`).
This is not a bug in the sense of wrong code — it is the literal,
un-sign-checked Eq.(10) formula doing exactly what its own docstring warns
it will do: `S_G ∝ 1/(g_u x_l/g_l x_u - 1)`, which has a genuine pole exactly
at the masing threshold (population equality), and `guard_masers=False`
means nothing catches it. 424/1695 rows (25%) have `R_01>10`; the 90th
percentile of `R_01` alone is already 2024.

**Consequence for methodology**: a `LinearNDInterpolator`-based retrieval
(the gold approach) would smear that pole across its *entire* local Delaunay
neighbourhood, not just the one point — corrupting nearby, otherwise-sane
grid cells. Since this model is cheap (~9 ms/point with warm-starting),
switched to a **fine direct grid** instead of interpolation: 70×50×36≈126,000
points on the same axis ranges as gold (`build_escape_grid.py --fine`,
~18 minutes), and `chi2_retrieve` scores raw grid rows directly. A pole then
stays an isolated single bad-χ² cell that simply never wins the argmin,
rather than poisoning its neighbours. `retrieve_stutzki1985_escape1d.py`
(new) mirrors the gold retrieval script exactly (same 5-panel-plus-landscape
layout, same `stutzki_physics`/`lut_interpolator.chi2_retrieve` reuse) but
reads this fine grid directly instead of interpolating a coarse one.

**Fine grid finished**: 126,000 points in 860.9s (14.3 min), 2723/126000
(2.2%) unconverged, 97083/126000 (77.0%) `any_maser`.

### 4.3 `retrieve_stutzki1985_escape1d.py` — bug found and fixed, then run

**Bug**: `load_grid` originally dropped unconverged rows before reshaping into
the `(70,50,36)` regular grid the landscape panels need — but 2.2% of rows
were unconverged, so `len(df) != prod(shape3d)` and the assert failed. Fix:
keep the full cross product; NaN the ratio columns (not drop the rows) for
unconverged points, so `chi2_retrieve`'s existing isnan check scores them
`+inf` (never wins the argmin) without breaking the grid shape needed for
`chi2_3d.reshape(shape3d)`.

**Result, 23 positions**: median χ²=0.956 (mean 2.63, worst: OMC S1 χ²=23.8 —
*same* worst-fit position as gold's χ²=20.8), median |Δ log n|=0.56 dex.
**100% pass `physically_consistent`** (vs 87.0% for gold) — every retrieved
point has η_f≤1 here (unlike gold's 2 unphysical/over-consistent-failing
cases). **No low-density branch found at all**: every one of the 23 global
best fits lands at `log n = 6.7-7.5` (median 7.15), 7/23 pinned exactly at
the density ceiling — where gold had 3 positions whose *global* best fit
itself sat on the low branch (χ²-favoured over the high branch). This
escape-probability model's likelihood surface evidently doesn't develop as
deep a low-density local minimum as the 3D model's does, at least not one
competitive with the high-density solution — worth a landscape-panel visual
check before treating this as settled, but it's what 23/23 argmins say.

**New finding — masing sits inside the retrieval, not just at grid extremes**:
39.1% (9/23) of best-fit points have `any_maser=True` — the closest match to
Stutzki's observed ratios, under the guard-off closure, quite often sits at a
point where some hyperfine group is population-inverted. This wasn't visible
in the coarse-grid characterisation (which only reported the *fraction of the
grid* that masers, not whether physically-preferred fits cluster there) — a
genuinely new result: **the retrieval is drawn toward the masing threshold**,
consistent with Stutzki's own discussion of hyperfine "flip-over" as a
high-density/low-temperature phenomenon, now seen concretely in a real
best-fit search rather than just in the grid's marginal statistics.

**Headline comparison — the (2,1) over-prediction is shared, not
geometry-specific**: the three positions with real Table 3 (2,1) data:

| position | escape1d pred | gold (3D) pred | Stutzki theory | Stutzki obs |
|---|---|---|---|---|
| OMC S3 | 0.0814 | 0.0811 | 0.052 | 0.015 |
| OMC S4 | 0.0869 | 0.0878 | 0.068 | 0.028 |
| S106 (0,0) | 0.0489 | 0.0679 | 0.018 | 0.014 |

OMC S3 and S4 predictions agree to within 1% between the two completely
different radiative-transfer methods (1D escape-probability vs full 3D NLTE)
— both sharing only the Loreau et al. (2023) collisional rates. This is
strong evidence that the (2,1)/(1,1) over-prediction relative to Stutzki's
1985 observations is **driven by the modern collisional rates themselves**,
not by an artefact of either radiative-transfer treatment (geometry, escape
approximation, or Magritte's imaging apparatus). S106 (0,0) is the one
position where the two methods diverge (0.049 vs 0.068) — worth a follow-up
look, but even there both still over-predict the observation.

### 4.4 Gold paused again

`output_lut_gold` paused a second time today at **1573/1764 rows** to free
the machine for the above. Resumable by grid key as always; no rows lost.

---

## 5. escape1d completeness suite, and a second collision-rate set

Instructed to (a) do the same interpolation-completeness and Fig. 4
analyses for escape1d that already existed for gold, "for completeness",
and (b) separately, repeat the *entire* escape1d exercise (grid, Fig. 4,
interpolation completeness, retrieval) with **Stutzki's own original
rates** instead of Loreau, after the user renamed
`output_escape1d/results` → `results_loreau_rates` (done directly on the
filesystem, not by this session) to make room for a second, parallel
`results_original_rates` tree.

### 5.1 New tools, mirroring the gold versions

- `stutzki85/reproduce_fig4_escape1d.py` — Fig. 4a-c contour reproduction
  directly from the coarse escape1d grid (18/24/30/36 K land exactly on
  its axis, same as gold's). Colour convention fixed mid-session per
  direct instruction to match the project's existing house style
  (`reproduce_raw_fig4_full.py`, 2026-09-10): 2nd-98th percentile clip
  **before** `griddata` interpolation (not after — clipping post-hoc
  still lets an extreme outlier drag the interpolated gradient toward it
  between itself and its neighbours), ratio panels always include the
  LTE/τ=0 floor in the colour range, one colorbar per panel.
- `stutzki85/interpolation_completeness_escape1d.py` — pooled 3D +
  axis-specific (T and log_n) hold-one-out, same method as
  `lut_interpolator.py` (interpolate log-amplitude, form ratios after),
  on the coarse 1764-point grid.
- `stutzki85/interp_curves_escape1d.py` — representative leave-one-out
  curves (true vs LOO, %-error annotated) at 4 slices per axis, same
  style as `plot_interpolation_axis_curves.py`.
- All three, plus `build_escape_grid.py`, now take `--rates
  {loreau,original}` and resolve their own grid/output paths via
  `build_escape_grid.out_csv_for`/`outdir_for` rather than hardcoded
  constants — needed once a second rates tree existed side by side with
  the first.

### 5.2 Fig. 4 for escape1d: masing sits exactly where the anomaly is strongest

The R(0→1) panel saturates hard in the low-density/high-column corner
(log n≈3.5-5, log N/Δv≳15.3) — precisely the region Stutzki's own model
also predicts the strongest outer-satellite anomaly. The other four
panels (R(1→0), R(2→1), R(1→2), (2,2)/(1,1)) are smooth and
S-shaped/Stutzki-like across the full grid. Per-panel masing fractions
printed in each subplot title range from 0% (T=18K, most of the grid) to
~97% (T=36K, high-column corner).

### 5.3 Interpolation completeness for escape1d (Loreau rates, coarse grid)

Pooled 3D hold-one-out (999 interior points): R_01 median 5.04%, pass
(<3.3%) 40.0%; **R_10 median 17.58%, pass 31.1%** (worst of the four);
R_21 median 1.45%, pass 97.6%; R_12 median 1.91%, pass 97.3%. Splitting
by masing status makes the mechanism explicit: R_10's non-masing median
is 2.17% (fine) vs its **masing median pinned at exactly 100.00%**
(clipping a near-zero or negative true amplitude to the 1e-12 floor
before taking its log means the interpolator's neighbourhood is
dominated by that floor value, so predictions collapse toward zero
regardless of the true curve). R_21/R_12 (inner satellites, never
masing in this dataset) stay under 2% median throughout — interpolation
is fine wherever masing isn't involved.

The leave-one-out curve plots make this visually unambiguous: at
T=48K/log N/Δv=15.8 (the coarse grid's own recorded pole, R_01 up to
1.9×10¹³), the true R_01 curve spikes sharply between two grid points
and the LOO prediction misses it by up to 285%; at the two fully-masing
T=36/48K slices, R_10's true curve is smooth (−0.1 to +0.3 across the
density axis) but every LOO prediction in the masing region collapses to
~0.

This is the quantitative confirmation of the design decision already
recorded in §4.2: a `LinearNDInterpolator`-based retrieval on this grid
would have been corrupted specifically on the outer-satellite ratios,
specifically in the masing regions — exactly why
`retrieve_stutzki1985_escape1d.py` uses a direct fine grid instead.

### 5.4 Second rate set: Stutzki's own (Green 1980 + his Table 1), via `rate_swap_full`

`rate_swap_full.build_full_hybrid_collision_matrix` already existed from
an earlier (09-10) side investigation: Loreau's matrix everywhere, except
the (1,1) level's full outgoing rotational network (to itself, (2,2),
(2,1), (3,2), (3,1), (4,4)) rebuilt from Green (1980) Table III NH3-He
rates × Stutzki & Winnewisser (1985a) Table 1 IOS relative factors,
scaled He→H2 by α=1.5, plus Table 2's 18 quasi-elastic intra-multiplet
rates — i.e. every collisional pathway Stutzki's own paper actually
tabulates, with Loreau filling in only what neither table covers
(rotational pathways not touching (1,1) directly). `build_escape_grid.py`
now wires this in as `--rates original` alongside `--rates loreau`
(default).

Coarse grid (1764 pts, 20.0s): 92/1764 (5.2%) unconverged, 1236/1764
(70.1%) any_maser — similar order to Loreau's 69/1764 (3.9%) unconverged,
1355/1764 (76.8%) masing, slightly less masing-prone. Fine grid (126,000
pts, 1127.5s ≈ 18.8 min): 3649/126000 (2.9%) unconverged, 88759/126000
(70.4%) any_maser (vs Loreau's 2723/126000, 2.2% unconverged, 97083,
77.0% masing).

**Interpolation completeness is measurably better with the original
rates.** Pooled 3D (980 interior points): R_01 median 2.51%/pass 57.9%
(vs Loreau 5.04%/40.0%); **R_10 median 2.50%/pass 66.9%** (vs Loreau's
17.58%/31.1% — the biggest single difference between the two rate sets);
R_21/R_12 essentially unchanged (~1.4-1.8%, >98% pass either way). The
non-masing/masing median split for R_10 is 2.11%/2.90% under original
rates versus 2.17%/**100.00%** under Loreau — the original rates don't
eliminate masing (43.5% of retrieval best-fits still land there, close to
Loreau's 39.1%) but the poles are evidently less numerically extreme, so
interpolation degrades gracefully instead of collapsing to the clip
floor.

**Retrieval, 23 positions, original rates:** median χ²=0.855 (vs Loreau
escape1d's 0.956, vs gold's 0.447), median |Δlog n|=0.611 dex (vs 0.556,
0.576), 95.7% physically consistent (22/23 — one more failure than
Loreau escape1d's 100%: `S106 (0,-40)` lands on the low-density branch,
log n=3.56, η_f=1.33 unphysical — the only position across all three
runs where the *original*-rates escape1d model finds a low-density
branch the Loreau-rates escape1d model missed entirely). Worst fit is
again OMC S1 (χ²=25.2, consistent with both other runs' ~21-24).

**(2,1)/(1,1) held-out test, all three methods side by side:**

| position | gold (3D, Loreau) | escape1d (Loreau) | escape1d (original) | theory (S85) | obs (S84) |
|---|---|---|---|---|---|
| OMC S3 | 0.0811 | 0.0814 | 0.0745 | 0.052 | 0.015 |
| OMC S4 | 0.0878 | 0.0869 | 0.0825 | 0.068 | 0.028 |
| S106 (0,0) | 0.0679 | 0.0489 | 0.0453 | 0.018 | 0.014 |

Switching from Loreau to Stutzki's own (partial) rate set nudges the
predictions down modestly (5-10%) but changes nothing qualitatively: all
three methods still over-predict the observed ratio by a wide margin at
all three positions, and the over-prediction persists even when the
model's own historical author's rates are used for the direct (1,1)-(2,1)
pathway. This further narrows where the 1985 discrepancy can be coming
from — not the choice of (1,1)-(2,1) collisional rate specifically, since
swapping that pathway back to Stutzki's own numbers doesn't resolve it
either; the remaining candidates are the still-Loreau-sourced rest of the
rate network, or (as Stutzki himself argued) his fitted densities being
too high.

---

## 6. Full transcription of Stutzki's own rates — Green (1980) Table III + Stutzki (1985a) Table 1, complete

Prompted by a direct question ("are green1980.pdf and Stutzki1984.pdf not
enough?") that caught two things worth recording as their own lesson:

1. **`Stutzki1984.pdf` is not a rates paper.** It is Stutzki, Jackson,
   Olberg, Barrett & Winnewisser (1984, A&A 139, 258) — the pure
   observational survey (23 sources, hyperfine ratio tables), zero
   collision-rate content. The actual rates/methods paper
   (Stutzki & Winnewisser 1985, A&A 144, 1, "Hyperfine selective
   collisional excitation of interstellar molecules") is a *separate*
   file already in `production/references/`: `Stutzki1985a.pdf`. Easy
   mixup given the similar filenames — worth a permanent note.
2. **The earlier `rate_swap_full.py` docstring undersold what's in these
   papers.** It claimed Table 1 only tabulates source=(1,1); a direct
   re-read (this session) showed Table 1 actually covers source states
   (1,1), (2,2), (2,1), (3,2), (3,1), (4,4) — every manifold the escape1d
   model tracks — and Green's Table III likewise gives the complete
   inter-manifold rotational network among the same six, not just
   pathways touching (1,1). The earlier "hybrid" matrix was a needless
   partial subset of what the source material actually supports, not a
   consequence of missing data.

### 6.1 Transcription and verification

User transcribed both tables in full as CSVs (`production/references/
Green_Table3_1980.csv`, 119 data rows; `Stutzki_1985a_table1.csv`, 326
data rows) — every entry, not just what the earlier partial hybrid used.
Checked directly before trusting either:

- The model's own level scheme is **exactly** 18 (J,K,F) states × 2
  parities = 36 levels, spanning (1,1):F=0,1,2; (2,1)/(2,2):F=1,2,3;
  (3,1)/(3,2):F=2,3,4; (4,4):F=3,4,5 — precisely the six manifolds both
  tables cover. No third table is needed (Stutzki's Table 2, the
  intra-multiplet quasi-elastic rates, was already fully transcribed for
  all six manifolds in an earlier session — checked directly, not
  assumed).
- Green's CSV covers all 15 possible unordered pairs among the six
  manifolds; Stutzki's CSV independently covers the same 15. Compared the
  two tables' (source manifold → target manifold) *direction* for every
  pair — **they match exactly**, with zero exceptions — confirming both
  papers tabulate the same "excitation" direction per pair (matching
  Green's own stated convention: "de-excitation rates can be obtained
  from detailed balance"), so no directional ambiguity exists in
  combining them.
- Only gap: `443U` and `444U` (the two highest sublevels) have no
  *outgoing* inter-multiplet row in Stutzki's Table 1 — expected, not a
  hole: (4,4) is the top manifold in this scheme, so there is nothing
  higher for those two upper-parity sublevels to excite into.

### 6.2 `rate_swap_transcribed.py` — new, general implementation

Unlike the old `rate_swap_full.py` (hand-coded, source=(1,1) only), this
parses both CSVs generically and builds the full matrix by manifold pair,
using the verified-identical tabulated direction, the same detailed-balance
formula and the same Eq.(22b)/(25) L-target symmetry the original (1,1)-only
code already used (`build_complete_original_collision_matrix`). Falls back
element-wise to Loreau only where the transcription genuinely has no entry.
**Coverage, measured not assumed**: `coverage_report()` at T=24K gives
1060 Stutzki-sourced vs 56 Loreau-fallback off-diagonal entries — **95.0%
Stutzki-sourced**, with the 56 fallback entries accounted for exactly by
the 443U/444U gap above. Smoke-tested against `run_one`: converges, no
negative or NaN rates.

Wired into the same `--rates` machinery as a third option,
`full_original` (`build_escape_grid.py`, `reproduce_fig4_escape1d.py`,
`interpolation_completeness_escape1d.py`, `interp_curves_escape1d.py`,
`retrieve_stutzki1985_escape1d.py` all updated). `original` (the old
partial hybrid) is kept, not replaced, so the three rate sets are directly
comparable side by side rather than the earlier partial result being
silently overwritten.

### 6.3 Full pipeline re-run — masing drops sharply with more complete physics

| | loreau | original (partial) | full_original (complete, 95%) |
|---|---|---|---|
| coarse grid masing | 76.8% | 70.1% | **58.8%** |
| fine grid masing | 77.0% | 70.4% | **57.1%** |
| fine grid unconverged | 2.2% | 2.9% | 2.4% |

Masing fraction drops monotonically as the rate network gets more
complete/correct — a real physical trend, not noise: going from Loreau's
fully-modern rates, to a small hand-picked subset of Stutzki's own rates,
to (nearly) his *complete* rate network, the model becomes steadily less
prone to population inversion under the guard-off closure.

**Interpolation completeness improves in lockstep** (pooled 3D
hold-one-out, 980-999 interior points): R_10 pass rate (<3.3% error)
31.1% (loreau) → 66.9% (original) → **90.7%** (full_original); R_01 pass
rate 40.0% → 57.9% → **70.5%**. The masing/non-masing median split for
R_10 goes from a catastrophic 2.17%/**100.00%** (loreau) to a mild
1.93%/2.90% (original) to a genuinely well-behaved **1.93%/1.67%**
(full_original, masing points interpolate *no worse* than non-masing
ones). More complete physics doesn't just change the answer — it makes
the numerics themselves better conditioned.

### 6.4 Retrieval: the low-density branch reappears, and (2,1) tells two different stories

23-position retrieval, full_original: median χ²=0.986, median
|Δ log n|=**0.242 dex** (well under both loreau's 0.556 and original's
0.611 — the high-density-branch fits now land much closer to Stutzki's
own Table 1a values), only **69.6%** physically consistent (16/23, down
from 100%/95.7%) — **7 positions now fail**, and unlike every earlier
escape1d run, this is not one or two edge cases: `S106 (0,0)`, `S106
(160,0)`, `S106 (40,0)`, `S106 (0,40)`, `S106 (0,-40)`, `S106 (-40,0)`,
and `W48 (-40,40)` all land on the **low-density branch**
(log n≈3.5-5.2) with the Jeans clump-count check failing (K_predicted <
K_required by a factor of a few to 35× at `S106 40,0`), not the η_f
check (η_f stays physical, 0.14-0.61, at all seven). This is qualitatively
closer to gold's 3D result (which found 3/23 positions on the low branch)
than either earlier escape1d run (0 and 1 respectively) — the more
complete rate network gives the low-density branch a real chance to
compete, where the earlier partial/Loreau rates didn't.

**The (2,1)/(1,1) held-out test now splits into two different stories**:

| position | gold (3D) | loreau | original | **full_original** | theory (S85) | obs (S84) |
|---|---|---|---|---|---|---|
| OMC S3 | 0.081 | 0.081 | 0.075 | **0.052** | 0.052 | 0.015 |
| OMC S4 | 0.088 | 0.087 | 0.083 | **0.063** | 0.068 | 0.028 |
| S106 (0,0) | 0.068 | 0.049 | 0.045 | **0.0015** | 0.018 | 0.014 |

OMC S3 and S4 both still land on the high-density branch under
full_original, and their predictions move markedly closer to (S3: now
*matches to 2 decimals*) Stutzki's own theoretical value — expected,
since the model is now using ~95% of his own tabulated rates, so it
should reproduce his own calculation closely at the same density. Both
still over-predict the *observation* by a wide margin (S3: 3.5×; S4:
2.3×), so the central historical result is unchanged: four independent
methods (3D Loreau, 1D Loreau, 1D partial-original, 1D full-original) now
all reproduce, rather than resolve, Stutzki's 1985 over-prediction at
these two positions. **S106 (0,0) inverts**: its retrieval moved to the
low branch (log n=5.24 vs Table1a's 6.255), and at that lower density the
strongly sub-thermal (2,1) line collapses to 0.0015 — now
*under*-predicting the observation (0.014) by ~9×, the opposite failure
mode from every other method/position in this whole exercise. This is a
genuinely new, unresolved result, not yet explained — flagged here rather
than smoothed over.

---

## 7. Checks worth adding to the mandatory-gate list (extends 09-16 §6)

5. Any χ² landscape plot must profile (minimum-project) over the axis not
   shown, never slice at a fixed value — see §2.1.
6. Any retrieval against a grid built with `guard_masers=False` (or any
   other un-guarded closure) must use a direct grid or otherwise check for
   poles before interpolating — see §4.2.
7. Any contour/heatmap plot in this project clips colour at the 2nd-98th
   percentile of ITS OWN panel, applied to the raw values *before*
   griddata/interpolation (not after), with ratio panels always including
   the τ=0/LTE floor — the house convention from
   `reproduce_raw_fig4_full.py`, reapplied in §5.1. A shared/global colour
   scale or post-hoc clipping both let one extreme point wash out or
   distort every other panel.

---

## 8. Scope decision: three rate/model combinations going forward

Instructed, after §6's results landed: from here on, only three
configurations are tracked — **Magritte 3D + Loreau** (gold), **escape1d +
Loreau**, and **escape1d + the full transcribed original rates**
(§6.2-6.4). The partial-hybrid `original` variant (§4-§5, the
hand-picked (1,1)-source-only subset) is retired from active comparison
— its outputs stay on disk (`output_escape1d/results_original_rates/`,
untouched) but won't be extended or referenced in new write-ups.

---

## 9. Paper draft correction: the Eq.11-departure claim was overclaimed

The Overleaf draft's §4.2 (comparison against Stutzki's Eq.~11) had
stated the 3D model's faster-than-Eq.11 saturation was "a genuine,
resolution-converged property of the model geometry... not a numerical
artifact," verified by checking insensitivity to angular quadrature,
mesh resolution, and image sampling. That check cannot actually rule out
a mesh-*type* artefact: the 09-16 cube-mesh defect (§1 of the
09-16 notes — non-monotonic τ(b), ~half the correct central value, does
not improve with resolution/nrays/pixels/pad) is, by construction,
invisible to a robustness check performed entirely *within* the same
cube topology. The user caught this independently and gutted the
overclaiming paragraph in the draft directly (via Overleaf), leaving a
placeholder figure — a new direct LTE comparison (cube vs radial mesh,
reproducing Stutzki's own Figs. 5-6, 3 densities, T=18K) they had
generated separately.

Read literally, that figure is reassuring rather than damning: both mesh
types show similar faster-than-Eq.11 saturation at low-to-moderate τ, so
the qualitative result is *not* primarily a mesh artefact — both a
geometrically correct and a geometrically defective mesh produce it. But
the cube mesh's departure runs measurably ahead of the radial mesh's at
the highest optical depths sampled (τ≳3, clearest in the (2,2)/(1,1)
panel), so the cube-mesh production grid likely overstates the effect's
*magnitude* somewhat at high τ, in a way this one LTE check doesn't fully
quantify (doesn't extend past τ~9, and hasn't been checked under NLTE
excitation at all). Rewrote §4.2 and the Summary to state precisely this
— genuine effect, magnitude uncertain at high τ, radial-mesh re-run
still needed to close it out — instead of either the original
overclaim or an equal-and-opposite overcorrection to "it's just a mesh
bug." Added an Ongoing Work item for the staged radial production grid
(mesh economy study + platinum1/2/3, per the plan file) that would
settle this properly.

**Lesson**: a robustness check's scope is bounded by what it varies.
"Insensitive to resolution" only rules out artefacts that resolution
itself would fix — it says nothing about a defect baked into the mesh's
*topology* (here: never sampling the sphere's surface at any
resolution). The right check is an independent method that doesn't share
the suspect assumption, which is exactly what the radial mesh comparison
provided once it existed.

---

## 10. New reproducible figure: optical depth vs. chord length

The 09-16 diagnostic behind §9 was numeric/prose only — no saved figure
existed. Instructed to add it; rather than dig up a dead scratchpad
session's script, wrote `w33/chord_experiment.py`, reusing the
already-validated `lte_probe.py` (τ(b) from brightness inversion *and*
independently from Magritte's own optical-depth image, both checked
against the exact analytic chord law, which is exact at LTE for a
uniform sphere). Ran fresh at a representative point (T=24K, log
n=6.0, log N/Δv=15.0) — different from the original diagnostic's point,
so this is an independent confirmation, not a re-plot.

**Result, cube vs radial**: cube fits the chord law at 24.6% RMS and is
visibly non-monotonic; limb/chord-law ratio 0.003 (essentially no flux
recovered at the limb, vs radial's 0.786). Cube's fitted central τ
(0.645) is ~0.84× the radial mesh's (0.766) — smaller than the original
diagnostic's ~2× gap, but at a different, less optically-thick point;
the raw (non-fitted) central pixel value tells the sharper story: cube
reads ≈0.38 there, almost exactly half of radial's fitted central value,
matching the original ~0.5 ratio. Radial mesh: 1.4% RMS, monotonic by
construction. Both brightness-inversion and Magritte's-own-image τ
extractions agree with each other to 3 decimal places on both meshes —
the two independent extraction methods aren't where the disagreement is;
the mesh is.

Figure now embedded in `paper1_draft.tex` §4.2 (`fig:chord_experiment`,
new — before the existing `fig:mesh_fig56` ratio-vs-tau consequence
figure) and saved at
`output_lut_gold/results/chord_experiment_cube_vs_radial.png`, force-added
to git as the primary evidence figure for this finding.
