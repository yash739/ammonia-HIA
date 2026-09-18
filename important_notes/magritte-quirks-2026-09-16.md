# Magritte quirks found while building the NH3 sphere LUT

Compiled 2026-09-16. Each entry says what happens, how it was established, and
how sure we are. **Verified** means reproduced by a direct, controlled test in
this project; **measured once** means a single measurement that has not been
repeated; **unresolved** means we saw it but do not know the cause.

Most tests used an LTE-forced run (`max_NLTE=0`). With LTE populations there is
no hyperfine anomaly, so the answers are known analytically: a uniform sphere
must give τ(b) = τ_centre·√(1−(b/R)²), and satellite/main ratios must follow
Stutzki & Winnewisser (1985) Eq. (11). Anything else is an artefact.

Code references are to `production/tgs/modelling/nh3_NLTE_sphere.py` unless
stated.

---

## 1. Imaging and optical depth

### 1.1 `compute_image_optical_depth_new` uses whatever spectral grid was set last — **verified**

The optical-depth image has no frequency argument. It is computed on the grid
from the most recent `compute_spectral_discretisation` call, and in `run_model`
that call belongs to whichever line was imaged last in the line loop.
`tau_main` is taken from that image after the loop, so **`tau_main` is the
optical depth of the last-imaged line, not the (1,1) line.**

Causal test (same model, only the imaging order changed):

| last line imaged | reported `tau_main` | that line's own τ |
|---|---|---|
| (2,2) | 0.13916 | 0.13916 |
| (2,1) | 0.01098 | 0.01098 |
| (1,1) | 0.42347 | 0.42347 |

Consequences:
- Every LUT build (coarse, offset, gold) passes `image_lines={'21': ...}`, so
  their `tau_main` column is the **(2,1)** optical depth.
- The legacy catalogue imaged (1,1) then (2,2), so its "Main Hyperfine Optical
  Depth" is the **(2,2)** optical depth.
- Any Fig. 5/6-style plot against `tau_main` from either source has the wrong
  line on the x-axis, and legacy and gold used *different* wrong lines.
- The optical-depth image itself is fine: Magritte's (1,1) τ image and τ
  recovered from (1,1) brightness agree to 0.2% on both meshes.

**Not yet fixed in code** (deliberately — gold is mid-build and its workers
re-import the module per model, so changing the definition now would mix two
definitions in one column). Fix: call
`compute_spectral_discretisation` on the (1,1) window immediately before the
optical-depth image.

### 1.2 The optical-depth image needs the narrow spectral window first — **verified**

After the population solve Magritte's spectral discretisation is the wide
default one. Computing the optical-depth image without first imaging the line
(which sets the ±3 MHz window) gave a central τ of 79.55 where ~1.2–1.3 was
expected. Same root cause as 1.1: the spectral grid is global state.

### 1.3 Holding a reference into `model.images` then imaging again segfaults — **verified**

`model.images` is a C++ vector. Computing another image appends to it and can
reallocate it, invalidating any Python object obtained from
`model.images[-1]` earlier. Accessing that object afterwards segfaulted
(UCX backtrace, signal 11). Copy what you need with `np.array(...)` and drop
the reference before calling any `compute_image_*` again.

### 1.4 Imaging the same line twice gives slightly different τ — **unresolved**

Re-imaging (1,1) on the same solved model gave a central τ of 0.42347 against
0.42804 the first time (1.1% lower). Same line, same spectral window, same
pixel grid. Cause unknown. Small, but it means the images are not a pure
function of the model state.

### 1.5 The image field of view is tied to the outer boundary radius — **verified**

The imager's field is set by the CMB boundary radius, which in this pipeline is
`r_boundary = r_out * fov_pad_factor`. There is no separate field-of-view
control. See 2.2 for why this matters.

### 1.6 Pixel centres span (npix−1)/npix of the nominal half-width — **verified**

Measured from the imager's own `ImX`/`ImY`: image half-extent / (pad·r_out) =
0.9375 = 15/16 at 16 pixels and 0.96875 = 31/32 at 32 pixels. The outermost
pixel *centre* sits half a pixel inside the edge.

`MAGRITTE_IMAGE_HALF_EXTENT_FACTOR = 15/16` is hard-coded, so it is only right
for 16×16 images (which is what the LUT uses — latent, not active). Better to
use `ImX`/`ImY` directly than any assumed factor.

At `fov_pad_factor=1.0` the pixels therefore only reach 0.94R (16 px) along the
axes, so the outer limb of the sphere is not sampled.

### 1.7 Imaging does not depend on `nrays` — **verified (cube mesh, LTE)**

`nrays` = 12, 48, 192 gave bit-identical images and τ(b). The imager traces its
own rays. Caveat: tested on the cube mesh in LTE only; whether the *NLTE
solution* is insensitive to `nrays` on the radial mesh has not been re-checked.

### 1.8 `tools.save_fits` writes a fixed 300×300 grid — **from code, earlier in session**

It re-interpolates onto its own hard-coded 300×300 grid regardless of the image
size, so each file is ~340 MB. Nothing in the pipeline reads these back;
leave `save_image_fits=False`.

---

## 2. Mesh

### 2.1 A Cartesian-cube seed gives a wrong τ(b) profile — **verified**

`build_point_cloud` seeds the Delaunay mesh with a cube of points spanning
±1.2R (`resolution` points per axis; 0.267R spacing at resolution 10) and never
places a point on the sphere's surface. In LTE:

- τ(b) is **non-monotonic** — it peaks at b/R ≈ 0.38 instead of the centre,
  which is impossible for a uniform sphere;
- it sits ~1.8× above the chord law mid-disc;
- the limb (b/R ≈ 0.95) emits ~0.6% of what it should;
- the central τ is about **half** the correct value for every line
  (cube 0.428 vs radial 0.855 for (1,1); 0.139 vs 0.278 for (2,2)).

It does **not** improve with mesh resolution (10, 14, 18, 24), `nrays`, image
pixel count (16² vs 32²) or `fov_pad_factor`.

### 2.2 `fov_pad_factor` changes the mesh, not just the view — **verified**

Because the boundary radius both sets the image field (1.5) and is passed to
`build_point_cloud`, padding changes the mesh the radiative transfer is solved
on: 269 points at pad 1.0, 317 at 1.0667, 377 at 1.15, 480 at 1.30
(resolution 10). Changing pad 1.0 → 1.15 moved the central τ (2.087 → 1.830 at
resolution 5) and centre-pixel ratios by up to 60%. A field-of-view setting
cannot legitimately change the optical depth through the centre of a sphere.

The mesh is deterministic: repeated calls with identical arguments give
byte-identical point clouds. An earlier "269 vs 377 points" puzzle was entirely
this pad dependence.

### 2.2b What the cube mesh actually contains — **verified**

Counted directly from the point clouds (resolution 10, dimensionless R = 1;
figure `scratch/plots/mesh_point_clouds_cube_vs_radial.png`):

| mesh | total | inner shell (0.01R) | outer shell | points inside R | radial span of those |
|---|---|---|---|---|---|
| cube, pad 1.0 | 269 | 108 | 108 | 53 | 0.40–0.87R |
| cube, pad 1.15 (gold) | 377 | 108 | 108 | 101 (+60 outside R) | 0.40–0.95R |
| radial, pad 1.0 | 1313 | 108 | 108 | 1097 | 0.07–0.94R |

In both cube meshes **there is no point at all between the 0.01R inner boundary
shell and 0.40R**, and the equatorial slab |z| < 0.12R contains no interior
point — only boundary-shell points. At pad 1.0 nothing sits between 0.87R and
the surface either. The largest empty radial gap is 0.39R for the cube versus
0.07R for the radial mesh. The two boundary shells make up 80% of the pad-1.0
cube's points (216 of 269) and 57% of gold's (216 of 377).

### 2.2c Seed points near the outer boundary are dropped — **observed**

The radial seed includes a shell at 0.999R, but after
`point_cloud_add_spherical_outer_boundary` the outermost interior points are at
0.937R: points close to the boundary radius are removed when the boundary shell
is added. Not yet characterised (tolerance unknown).

### 2.3 A purely radial seed fixes τ(b) — **verified**

Seed with shells uniform in r, points per shell ∝ r², ~0.4% radial jitter,
a point at the origin, and a thin fringe out to the boundary. Result in LTE:
τ(b) monotonic, RMS deviation from the chord law 5.6% at 1313 points and 3.2%
at 3514, converging with refinement. Implemented as
`build_point_cloud(..., mesh={'kind': 'radial', 'n_shell': 14, 'base': 260, 'seed': 7})`.

### 2.4 The density remesher destroys the radial structure — **verified**

Passing the radial seed through `mesher.remesh_point_cloud` (as the cube path
does) gave 515 points and a τ(b) as bad as the cube. The remesher must be
skipped for the radial mesh.

### 2.5 Clustering shells toward the surface fails — **measured once**

Shells uniform in r work. Shells clustered geometrically toward r = R
(1827 points) failed badly (RMS 88%, non-monotonic) — clustering starves the
interior. One configuration tested.

### 2.6 Co-spherical points need jitter — **design choice, not separately tested**

Perfectly co-spherical shells can give a degenerate Delaunay tessellation, so a
small radial jitter with a fixed random seed was added. We did not run the
unjittered version to confirm it fails.

### 2.7 The remesher responds to absolute scale — **from code, earlier in session**

Building the point cloud directly in SI units gave 53–78 interior points across
a 4-dex density sweep; building in dimensionless units (unit radius and
density) and scaling afterwards gives an identical cloud at every grid point.
Keep building dimensionless.

---

## 3. Solver and cost

### 3.1 The radial mesh converges far more slowly in NLTE — **measured once**

Same physical model, `max_NLTE=250`: cube (269 points) converged in 281 s;
radial (1313 points) took 3774 s — **13.4×**. The radial run crawled early
(12.6% converged at iteration 16, 25.6% at iteration 91) and reached 100%.
Whether this is simply point count or small cells near the centre (the origin
point plus the 0.01R inner boundary shell) is not yet known.

### 3.2 `max_NLTE=0` gives LTE populations — **verified**

`run_model` calls `compute_LTE_level_populations()` and then zero NLTE
iterations. Consistent with every LTE check above, including optically thin
satellite/main ratios matching the LTE hyperfine intensities to <0.2%.

### 3.3 Frozen convergence at `resolution=14` — **unresolved**

At resolution 14 one model reported exactly 35.7488% converged on several
successive iterations (the solve was genuinely re-running), reproduced in a
second clean run, and was killed after ~1000 s. Possibly a corrupted,
re-used model file from an earlier killed run (the `.hdf5` path is
deterministic from the parameters); possibly a genuine resolution-14 problem.
Not re-tested.

### 3.4 Model files are keyed only by physical and numerical parameters — **verified**

The `.hdf5` name encodes XNH3, density, radius, vturb, T, pad, resolution and
nrays. Before the radial mesh was added it did **not** encode the mesh type, so
a cube and a radial run at identical parameters would have written the same
file. A mesh tag is now appended for non-cube meshes (cube names unchanged).
Killed runs can also leave partial files at those deterministic paths.

---

## 4. Related bugs that were ours, not Magritte's

For context only — these were in our analysis code, now fixed:

- Outer hyperfine satellites were labelled by parameter index, swapping
  F1 0→1 and 1→0.
- The velocity axis was built with the opposite sign to the radio convention.
- `npoints`/`nboundary` were written as −1 to every LUT row.
- `run_model`'s 5-Gaussian fit was checked on synthetic spectra and is accurate
  to ≤8%, so it is not a source of the τ/ratio problems above.

---

## 5. What this means for existing results

- **Gold LUT** (cube mesh, pad 1.15): ratios carry the cube's τ(b) error, and its
  `tau_main` column is the (2,1) optical depth. Kept as a fallback/comparison
  set only.
- **Legacy catalogue** (cube mesh, pad 1.0): `tau_main` is the (2,2) optical
  depth, and it carries the cube error too. Its apparent agreement with Eq. (11)
  was measured against the wrong line and should not be relied on.
- **Earlier session claims to revise:** the "legacy matches Eq. (11), gold
  doesn't" comparison put two different lines on the x-axis. The case against
  the cube mesh stands independently, because the τ(b) shape test uses (1,1)
  brightness directly.

## 6. Checks worth making mandatory

Cheap (LTE runs take seconds) and would have caught most of the above:

1. τ(b) from (1,1) brightness must be monotonic and follow √(1−(b/R)²).
2. Reported `tau_main` must equal the (1,1) τ image at the centre.
3. Satellite/main ratios vs (1,1) τ must follow Eq. (11).
4. Point cloud identical across a density sweep (for each mesh type).
