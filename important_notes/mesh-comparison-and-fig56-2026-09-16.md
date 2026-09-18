# Cube vs radial mesh, mesh economy, and Stutzki Figs. 5/6 at T = 18 K

Session of 2026-09-16. Companion to `magritte-quirks-2026-09-16.md` (the
Magritte behaviours behind all of this). **Status when written: investigation
paused; gold LUT build resumed; radial "platinum" work to be picked up later.**

All ratio checks below use the LTE control (`max_NLTE=0`): with LTE populations
there is no hyperfine anomaly, so disc-averaged satellite/main ratios must follow
Stutzki & Winnewisser (1985) Eq. (11) exactly. Any departure is a mesh/imaging
artefact. Eq. (11) is computed analytically (`stutzki_physics.eq11_thermal_ratio`,
which takes the *radial* τ, so it is evaluated at τ_chord / 2).

---

## 1. Conventions settled this session

- **τ on the x-axis of Stutzki's Figs. 5–6 is the central-chord optical depth**
  of the (1,1) main line. Established by fitting his *plotted* Eq. (11) curve on
  the scanned figure against the exact formula: best τ scale factor 2.1
  (rms 0.037) vs 0.046 unscaled. His Eq. (9)–(10) text confirms his τ_G is
  radial, with 2τ in the emergent-intensity formula.
- **Our `tau11_chord`** = central-pixel value of Magritte's optical-depth image on
  the (1,1) spectral window. Verified to be (1,1) by its hyperfine fingerprint:
  five peaks at −19.54, −7.52, +0.08, +7.69, +19.41 km/s with τ shares
  0.103 / 0.137 / 0.508 / 0.143 / 0.109 (LTE: 0.111 / 0.139 / 0.500 / 0.139 / 0.111).
- **`tau_main` in every existing CSV is the wrong line** (last-imaged line):
  (2,1) in coarse/offset/gold, (2,2) in the legacy catalogue. At n = 10^7,
  log N/Δv = 14.0, gold's stored `tau_main` is 0.0005 while the true (1,1) chord
  τ is 0.0744 (~150×). Legacy plot
  `scratch/output/output_test_1e-6_parallel_12rays_v2/results/nh3_2x2_tau_satellite_grid.png`
  was confirmed (from code at commit `d05c35a`) to use τ(2,2), with the
  single-sightline formula as its reference rather than Eq. (11).

## 2. LTE check at T = 18 K: cube vs radial (pad 1.0, 32×32 images)

`w33/fig56_slice.py --mode lte`, n = 10^3.5/10^5/10^7, 9 columns
(log N/Δv 14.0–15.8). Output `scratch/output/fig56_T18/slice_lte.csv`.

| mesh | points | τ(b) RMS vs chord law | monotonic | disc/Eq.(11), outer (min–max) |
|---|---|---|---|---|
| cube (res 10) | 269 | 25% | 0% | 1.02 – 1.31 |
| radial 14 shells × base 260 | 1313 | 1.5% | 100% | 0.93 – 1.06 |
| radial 10 × 140 | 628 | 4.3% | 7% | 0.92 – 1.29 |
| radial 8 × 90 | 427 | 6.5% | 0% | 1.02 – 1.32 |

The radial 1313 mesh sits on Eq. (11) from τ ≈ 0.06 to 8.5, drifting ~7% low at
the highest τ. The cube leaves Eq. (11) at τ ≈ 0.3 and reaches ~0.64 vs ~0.48
(outer) and ~0.74 vs ~0.58 (inner) at τ ≈ 4.3; its curves stop at τ ≈ 4.3 for
the same columns because it loses half the central chord.

Figures: `scratch/plots/fig56_T18_LTE_check_direct.png` (direct plot) and
`scratch/plots/fig56_T18_overlay_LTE_check.png` (drawn on Stutzki's scan; radial
lands on his Eq. (11) line, cube does not).

## 3. Mesh economy: does the radial mesh need 1313 points?

`w33/fig56_slice.py --mode lte`, T = 18 K, n = 10^5, log N/Δv ∈
{14.225, 14.9, 15.35, 15.8}. Output `scratch/output/mesh_economy_lte/slice_lte.csv`.
Worst Eq.(11) deviation = max over the 4 columns of |ratio/Eq(11) − 1|, outer or inner.

| shells × base | points | central τ (correct 8.09) | τ(b) RMS | monotonic | Eq.(11) worst |
|---|---|---|---|---|---|
| 8 × 90 / 140 / 200 / 260 | 427–788 | **4.05** | 4.8–6.6% | no | 28–30% |
| 10 × 90 / 140 / 200 / 260 | 493–964 | 8.09 | 3.5–5.3% | no | 7–9% |
| 12 × 90 | 557 | 8.09 | 4.7% | yes | 8.5% |
| **12 × 140** | **726** | 8.09 | 2.9% | yes | 7.8% |
| 12 × 200 | 929 | 8.09 | 1.9% | yes | 7.7% |
| 14 × 140 | 821 | 8.09 | 2.6% | yes | 7.7% |
| 14 × 260 (current) | 1313 | 8.09 | 1.5% | yes | 7.1% |

Findings:
- **Shell count is the critical knob.** 8 shells fails regardless of points per
  shell, reproducing the cube's exact half-chord central τ (4.05).
- ≥ 10 shells: central τ correct and Eq. (11) matched as well as the 1313 mesh.
- Points per shell only refine the τ(b) shape and fill the limb.
- 10 shells is never monotonic within the 2% tolerance; 12 and 14 always are.
- **Candidate: 12 shells × base 140 = 726 points** (55% of 1313), passes every
  check the 1313 mesh does. **Not yet measured: its NLTE cost** (LTE wall time
  does not track NLTE cost).

## 4. Cube vs radial at matched point counts

Same LTE setup as §3. Cube size set by `resolution`; the density remesher caps
growth (res 10/14/18/22/26/30/34/38 → 269/408/522/720/820/964/937/1056 points).

| points | cube: central τ / τ(b) RMS / Eq.(11) worst | radial: central τ / τ(b) RMS / Eq.(11) worst |
|---|---|---|
| ~525 | 4.04 / 21% / 27% | 4.05 / 5.7% / 29% (8 shells) |
| ~725 | 4.05 / 20% / 30% | 8.09 / 2.9% / 8% (12 shells) |
| ~820 | 4.05 / 20% / 28% | 8.09 / 2.6% / 8% (14 shells) |
| ~965 | 4.05 / 20% / 28% | 8.09 / 3.5% / 7% (10 shells) |
| ~1060 | 4.05 / 20% / 28% | 8.09 / 1.5% / 8% (14 shells) |

**Adding points to the cube buys nothing**: from 269 to 1056 points its central τ
stays at 4.05, τ(b) ~20% off and never monotonic, Eq. (11) missed by 27–30%.
Point placement, not point count, is the problem. Even the densest cube has no
interior points inside ~0.26R (res 10: none inside 0.40R).

Point-cloud visualisation: `scratch/plots/mesh_point_clouds_cube_vs_radial.png`
(script `w33/plot_mesh_point_clouds.py`).

## 5. Can gold's dataset reproduce Figs. 5/6? (NLTE, T = 18 K)

Gold's exact configuration re-run (cube, pad 1.15, 16×16 images, max_NLTE 250),
n = 10^5 and 10^7, 9 columns, recording τ(1,1) correctly. n = 10^3.5 skipped
(too slow). Output `scratch/output/fig56_T18_goldcfg/slice_nlte.csv`; all 18
converged (≥ 99.6%).

**The re-run is gold:** same 377-point mesh; disc ratios match gold's stored
values to 0.1–1.8% (small drift from using a b ≤ R pixel mask vs gold's whole-image
mean).

Ratio / Eq.(11) at the correct τ(1,1):

| n | τ_chord | F1 0→1 | F1 1→0 | inner average |
|---|---|---|---|---|
| 10^7 | 0.07 | 1.05 | 1.00 | 1.02 |
| 10^7 | 0.58 | 1.26 | 1.02 | 1.10 |
| 10^7 | 1.61 | 1.48 | 1.09 | 1.23 |
| 10^7 | 4.48 | 1.68 | 1.20 | 1.35 |
| 10^5 | 0.09 | 1.06 | 1.00 | 1.02 |
| 10^5 | 0.70 | 1.36 | 0.98 | 1.12 |
| 10^5 | 1.93 | 1.84 | 0.94 | 1.25 |
| 10^5 | 5.28 | 2.50 | 0.90 | 1.39 |

Verdict: **gold does not give usable Fig. 5/6 equivalents.** At n = 10^7
(near-thermalised; Stutzki's curves lie close to Eq. (11)) all three ratios run
above Eq. (11), up to +68% / +20% / +35% at τ ≈ 4.5. The inner-satellite
average — the control, which Stutzki's model keeps on Eq. (11) — is 16–39% high
for τ ≳ 1 at both densities. At n = 10^5 the anomaly has the right *sense*
(0→1 enhanced, 1→0 suppressed below Eq. (11)), but its size cannot be trusted
while the control is inflated. The overshoot grows with τ: the same signature as
the cube mesh's LTE failure (§2).

Figures: `scratch/plots/fig56_T18_goldcfg.png` (direct, Eq. (11) computed);
`scratch/plots/fig56_T18_goldcfg_interim.png` (overlay on the scan, partial).

## 6. Parallelism (same models, gold config)

Gold ran 3 workers × 4 threads; the slice ran 8 workers × 2 threads. Per-model
wall time was ~2.7× longer at 2 threads (range 2.4–3.1× over 7 identical points).
Total throughput is about equal (3 vs 8/2.7 ≈ 3 models per unit time), but 3×4
did it on 12 cores and finishes each model ~2.7× sooner. The >2× slowdown from
halving threads points to contention on a saturated 16-core machine. 4×4 might
beat both — untested.

## 7. Where things stand / to resume later

- **Gold LUT build: resumed** (cube, pad 1.15, 3 workers × 4 threads) as the
  fallback/comparison dataset. Its `tau_main` column is τ(2,1) and its ratios
  carry the cube overshoot — do not use it as the science table.
- **Radial T = 18 K NLTE slice: paused**, nothing completed
  (`scratch/output/fig56_T18/`, resumable: `w33/fig56_slice.py --mode nlte --meshes radial`).
- **Next steps when resuming:**
  1. NLTE timing on the 12 × 140 (726-point) radial mesh vs the 1313 mesh
     (1313 took 3774 s vs 281 s for the cube at one point).
  2. Radial NLTE Fig. 5/6 slice at T = 18 K on the chosen mesh.
  3. Fix `tau_main` in `run_model` to use the (1,1) window (§1), and add the
     hyperfine-fingerprint check to the LTE gate.
  4. Re-measure `nrays` sensitivity on the radial mesh (the nrays-independence
     result was only shown for the cube).

## 8. Code added/changed this session

- `nh3_NLTE_sphere.py`: `mesh=` option (`None` = original cube, byte-identical;
  `{'kind':'radial','n_shell','base','seed'}`), mesh tag in model filenames,
  `return_image=True` per-line intensity + τ images (copied before re-imaging to
  avoid the `model.images` segfault).
- `w33/build_lut.py`: `mesh` threaded through tasks (HEADER unchanged).
- `w33/lte_probe.py`: τ(b) from brightness and from the τ image; chord-law metrics.
- `w33/fig56_slice.py`: slice runner (LTE/NLTE, named or ad-hoc meshes
  `radial_s<n>b<base>`, `cube_r<res>`), resumable.
- `w33/plot_fig56.py`: direct Figs. 5/6 plot with analytic Eq. (11).
- `w33/plot_fig56_overlay.py`: overlay on the scanned paper figure (calibrated axes).
- `w33/plot_mesh_point_clouds.py`: point-cloud visualisation.
- Backups of the two edited core files: `scratch/backups/pre_radial_mesh/`.
