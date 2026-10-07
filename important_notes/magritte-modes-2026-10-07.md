# Magritte solver and imaging modes: what exists, what helps (2026-10-07)

Read-only survey of the vendored Magritte (`Magritte/`, C++ in `src/`, bindings in
`src/bindings/pybindings.cpp`, Python helpers in `magritte/`). Cost statements are from reading the code
unless marked "measured" (scratch/nlte_speed/, important_notes sec. 25, production/output_speedup_tests/).
Constraint for our work: **parameters and Python-level use only, no edits to the solver source.**

## 0. Findings that matter most
1. **Our slow default was a local edit.** `Magritte/src/model/parameters/parameters.hpp:119` carries an
   uncommitted change (`git diff`) flipping `sum_opacity_emissivity_over_all_lines` from upstream `false` to
   `true` ("legacy behaviour, for those who wish to compare to previous results"); `sphere.py` also set it to
   `True` explicitly. `True` selects the `None` approximation: every opacity/emissivity evaluation loops over all
   115 lines (one `exp` each). `False` selects `CloseLines` (binary search over the sorted line list within
   `max_distance_opacity_contribution` = 10 widths; `solver.tpp:1219-1262`, `:2018-2143`). Measured: identical
   ratios (5 decimals), 7-9x faster.
2. The flag is **not persisted** in the HDF5 (`parameters.cpp:96-130` lists what is read), so it must be set after
   every `Model(file)`. (`nquads` is persisted but is a `SetOnce` value, so it must be chosen before the model is
   written.)
3. `OneLine` (`one_line_approximation`) is invalid for overlapping hyperfine lines and is blocked for all imaging.
4. Each solver iteration costs about hnrays x npoints x nlines x nquads x ray length (x lines summed per step in
   the `None` approximation). Each line's ray is traced twice per (point, ray), and overlapping lines' frequency
   bins are **not** shared. Removing that redundancy needs C++ edits, so it is excluded.

## 1. Level populations / radiation field
All need `compute_spectral_discretisation()` (no arguments) first.

| mode | entry point | what it does | relevance (no C++ edit) |
|---|---|---|---|
| sparse ALI (ours) | `compute_level_populations_sparse(use_Ng, max_iter)`, `model.cpp:756` | Feautrier order 2 per ray, effective mean intensity `Jeff` from a diagonal ALO, per-point dense long-double QR for statistical equilibrium, threaded over points | already right; requires `n_off_diag = 0`; CPU threads only |
| dense ALI | `compute_level_populations(use_Ng, max_iter)`, `model.cpp:654` | stores `u(hnrays, npoints, nfreqs)` and `J(p,f)`; one global `SparseLU` on a (npoints*nlev)^2 matrix, single-threaded | allows `n_off_diag > 0` (wider ALO, fewer iterations on thick lines) but costlier iterations: speculative, one-node test |
| short characteristics | `compute_level_populations_shortchar`, `model.cpp:863` | order-0 formal solution, diagonal ALO, global LU | first-order, no advantage for a static problem |
| radiation field only | `compute_radiation_field_feautrier_order_2[_sparse]`, `_anis`, `_uv`, `shortchar_order_0` | one formal solution at fixed populations | `_uv` is labelled bugged; `_anis` (J2 tensor) is not used by any population loop |
| single stat-eq step | `compute_level_populations_from_stateq()` | one statistical-equilibrium solve with current `Jeff` | none |
| LTE | `compute_LTE_level_populations()` | sets LTE populations and seeds `lspec.populations` (required before any Ng step or warm start) | -- |

No multigrid exists (`tests/test_multigrid.cpp` is a commented-out stub). `use_scattering` and
`adaptive_ray_tracing` are not connected to any solver code.

## 2. Parameters (defaults in `parameters.hpp`; set after `Model(file)`)
Persisted in the HDF5: `n_off_diag`, `max_width_fraction`, `convergence_fraction`, `min_rel_pop_for_convergence`,
`pop_prec`, `min_opacity`, `Ng_acceleration_mem_limit`, `store_intensities`, `one_line_approximation`,
`use_adaptive_Ng_acceleration`. Everything else (including `sum_opacity_emissivity_over_all_lines`,
`max_distance_opacity_contribution`, `max_interpolation_diff`, `min_line_opacity`) is **not**.

| parameter | default | effect | speed relevance |
|---|---|---|---|
| `sum_opacity_emissivity_over_all_lines` | true (local edit; upstream false) | None vs CloseLines | **high** (measured 7-9x, exact) |
| `max_distance_opacity_contribution` | 10 | CloseLines cutoff in Doppler widths | low; 5/10/20 identical |
| `pop_prec` | 1e-6 | per-level relative-change stop threshold | medium (fewer iterations; fixes the never-converging cold corner) |
| `convergence_fraction` | 0.995 | fraction of levels that must satisfy `pop_prec` | low-medium |
| `min_rel_pop_for_convergence` | 1e-10 | levels with smaller population are ignored by the convergence test | medium; changes the criterion |
| Ng: `use_adaptive_Ng_acceleration`, `Ng_acceleration_mem_limit` (9), `adaptive_Ng_acceleration_min_order` (2), `..._use_max_criterion` (true), `Ng_acceleration_remove_N_its` (0) | see left | acceleration path | medium in the iteration-bound cold corner only (measured 158 -> 82 iterations) |
| `max_interpolation_diff` | 1.4 | adds ray sub-points where normalised line opacity changes by more than this factor; compare printed `length:` with 2*npoints | low-medium, unmeasured |
| `max_width_fraction` | 0.35 | exact erf optical depth when the Doppler shift is large | none (zero velocity always takes the trapezoid path) |
| `min_opacity`, `min_line_opacity`, `min_line_emissivity`, `min_dtau` | floors | numerical floors | none |
| `store_intensities` | false | pre-allocate `u`/`J` | negligible, unused by sparse mode |
| `nquads` | 20 | Gauss-Hermite points per line; `nfreqs = nlines*nquads`; cost linear in it | medium-high but accuracy-limited (measured: 12 is safe, 8 is not) |
| `nrays`/`hnrays` | 12 | angular samples; cost linear in `hnrays` | accuracy-limited |
| `spherical_symmetry` | false | 1D mode (needs `dimension = 1`; analytic ray stepping) | enables the 1D solver path (prototype in scratch/nlte_speed/proto1d.py) |
| `rays.use_adaptive_directions` | false | per-point HEALPix directions (3D only) | none |

## 3. Parallelism
OpenMP only in the installed build (`configure.hpp`: `OMP_PARALLEL true`, MPI/GPU off; `-O3 -fopenmp -flto`, no
`-march=native`, no SIMD). Threads: `OMP_NUM_THREADS` or `magritte.core.pcmt_set_n_threads_avail(n)`. The sparse
solver uses `threaded_for` (static contiguous blocks), never the accelerator path. GPU would need a CUDA rebuild
(`-DGPU_ACCELERATION=ON`; only a CUDA header is vendored, the SYCL header is missing) and still could not run the
sparse solver; the machine's T400 (4 GB) has weak FP64. Not worth pursuing.

## 4. Imaging
| path | entry | notes |
|---|---|---|
| new imager | `compute_image_new(ray_nr[, Nx, Ny])`, `solver.tpp:576` | one ray per pixel; **default 256x256** if no pixel counts are given (we pass 16x16); in 1D the direction is fixed to (0,1,0); extent is +-R in both axes. A strip `Nx=1` samples the full radial profile. |
| old imager | `compute_image(ray_nr)`, `model.cpp:961` | one ray per **mesh point**: cheapest; in 1D (last ray) gives I(b) at b = each mesh radius (used in the 1D prototype) |
| optical-depth image | `compute_image_optical_depth[_new]` | tau only, no formal solution; uses whatever spectral grid was last set (the `tau_main` bug) |
| single point | `compute_image_for_point` | debug |

Spectral windows: `compute_spectral_discretisation()` (line-centred Gauss-Hermite bins, for the solver),
`(width)`, `(nu_min, nu_max)` (uniform, 2300 bins), `(nu_min, nu_max, n)` (uniform, n bins; the main imaging cost
control; we use 500). Each `Image` copies its frequencies at construction, so looping windows then imaging is safe.
Imaging uses the same opacity approximation flag, so CloseLines also speeds imaging.

## 5. Warm start
`lineProducingSpecies[l].population` is read-write from Python (long-double vector, npoints*nlev, **absolute**
populations). Required sequence: `compute_spectral_discretisation()` -> `compute_inverse_line_widths()` ->
`compute_LTE_level_populations()` (seeds the history the Ng code dereferences; skipping it is undefined behaviour)
-> overwrite `population` -> `model.lines.set_emissivity_and_opacity()` (else the first radiation field uses stale
LTE emissivity/opacity) -> solve; set `Ng_acceleration_remove_N_its = 1` so the first Ng residual is not
"LTE -> warm start" (inference, untested). `model.write()` stores populations; `Model(file)` reloads them;
`population_prev1..3` are not restored. Untested so far.

## 6. Summary
| mode | promise | needs C++? |
|---|---|---|
| CloseLines | **high, adopted-pending-validation** | no |
| sparse solver at `n_off_diag = 0` | keep | no |
| dense solver + `n_off_diag > 0` | speculative, one-node test | no |
| Ng / `pop_prec` / `min_rel_pop_for_convergence` / `convergence_fraction` | medium | no |
| `nquads`, 36-level LAMDA truncation | medium-high, accuracy-limited (being validated) | no |
| `max_interpolation_diff` | low-medium, unmeasured | no |
| warm start | medium-high for repeated runs, untested | no |
| 1D spherical mode + old imager | high for cost and mesh-free geometry; prototype agrees with radial 14x260 to 0-3% except R_01 (3-6%) | no |
| imaging economy (fewer pixels/bins) | low now, grows as the solve gets faster | no |
| one-line, shortchar, anis/uv fields, scattering, adaptive rays, multigrid, GPU/SYCL/MPI | none | n/a |
| per-line ray re-tracing, shared frequency bins, `-march=native`/`-ffast-math` | would help | **yes** |
