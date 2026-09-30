# NH3 hyperfine-anomaly modelling

Code for **Paper I**: a 3D non-LTE reassessment of the NH3 (1,1) hyperfine
intensity anomaly, revisiting Stutzki & Winnewisser (1985, A&A 144, 13;
"S85"). It does two things:

1. Solves the full 3D non-LTE problem for a homogeneous, bare NH3 clump with
   [Magritte](https://github.com/Magritte-code/Magritte) and hyperfine-resolved
   NH3–H2 rates (Loreau et al. 2023), over a grid of density, temperature and
   column. Those models are then used to retrieve physical parameters from
   observed hyperfine ratios and to predict the held-out (2,1)/(1,1) line
   ratio.
2. Re-implements Stutzki's own 1D escape-probability method, with either the
   modern rates or a reconstruction of his 1985 rates, as an independent
   check on the 3D results.

The science, and every number quoted from these scripts, is written up in
the Paper I draft (Overleaf) and in the dated notes in
[`../../../important_notes/`](../../../important_notes/).

---

## Layout

```
modelling/
├── nh3hia/                 importable library ("NH3 hyperfine intensity anomaly")
│   ├── paths.py            every filesystem location, derived from the repo root
│   ├── hyperfine.py        (1,1)/(2,2) hyperfine components, velocity offsets, labelling
│   ├── spectral_fit.py     5-Gaussian hyperfine fit; components identified by fitted centre
│   ├── spectrum_utils.py   peak extraction for the (2,2)/(2,1) main lines
│   ├── noise.py            synthetic channel noise for model spectra
│   ├── scoring.py          chi^2 of a model ratio vector against an observed one
│   ├── model3d/            the Magritte 3D non-LTE sphere
│   │   ├── sphere.py       run_model(): mesh, non-LTE solve, imaging, spectra, tau
│   │   └── lte_probe.py    LTE (max_NLTE=0) diagnostics: tau(b) vs the exact chord law
│   ├── lut/                the precomputed model grids ("lookup tables")
│   │   ├── build.py        resumable, multiprocess grid builder (one row + spectra per model)
│   │   ├── axes.py         grid axes (gold grid and Figs 5/6 grid)
│   │   └── interpolator.py LutInterpolator + chi2_retrieve: interpolation-based retrieval
│   ├── escape1d/           1D escape-probability model (S85's method)
│   │   ├── model.py        36-level statistical equilibrium + S85 Eq. (10) closure
│   │   ├── rates_stutzki.py   S85's own 1985 rates (Green 1980 x S85a Table 1), 95% complete
│   │   ├── rates_table2.py    S85a Table 2 quasi-elastic rates and level-index helpers
│   │   ├── grid.py         builds the 1D grids (CLI); output paths per rate set and linewidth
│   │   └── curves.py       dense S85 Fig. 4 grid and Fig. 5/6 curves
│   ├── stutzki/            S84/S85 data and physics
│   │   ├── params.py       S85 Table 1a fits, Table 3 (2,1)/(1,1) values, test positions
│   │   ├── physics.py      filling factor, Jeans clump count, Eq. (11) no-anomaly curve
│   │   └── tables.py       parser for the digitized S84/S85 tables
│   └── w33/                W33 (Tursun et al. 2022)
│       ├── params.py       observed line parameters transcribed from the paper
│       └── observed.py     observed ratio vectors and the quadrant gate
├── analysis/               runnable experiments; one script per figure/table/check
│   ├── grids/              build the model grids
│   ├── retrieval/          fits to S84 and W33, held-out (2,1) tests, comparisons
│   ├── validation/         interpolation accuracy, noise-injection recovery, quadrant check
│   ├── mesh/               the cube vs radial vs hybrid mesh investigation
│   └── figures/            S85 Fig. 4 and Figs 5/6 reproductions
├── tests/                  pytest suite (no Magritte solves; runs in ~20 s)
├── data/observed/          digitized observations (S84/S85 tables, W33 line heights)
├── archive/                superseded code, kept for provenance; not maintained (see its README)
└── pyproject.toml          package metadata and pytest configuration
```

Only `nh3hia/` is a library. Scripts in `analysis/` may import from each
other where one produces what another plots, but nothing in `nh3hia/`
imports from `analysis/`.

## Running

Everything runs as a module, **from this directory**:

```bash
cd production/tgs/modelling
python3 -m pytest                                    # tests
python3 -m analysis.retrieval.magritte_stutzki1984   # any analysis script
```

(Alternatively, `pip install -e .` here makes the packages importable from
anywhere.) Most scripts take `--help`.

**Requirements:** Python ≥ 3.9 with numpy, scipy, pandas, matplotlib,
astropy and healpy, plus Magritte for anything that solves or images a
model. Magritte is vendored at the repository root (`Magritte/`) and
installed from there as an editable package. The library code that reads
grids, retrieves parameters or runs the 1D model does not need Magritte.

**Compute conventions:** Magritte runs use 3 worker processes with
`OMP_NUM_THREADS=4` each (about 14 cube-mesh models per hour on the
16-core machine). Run long builds in `screen`. All builders resume by
grid key, so a stopped build restarts where it left off; never rerun a
whole grid.

## Data products

All outputs live outside this directory, in `production/` (tracked results)
or `scratch/` (disposable):

| Directory | Contents | Built by |
|---|---|---|
| `production/output_lut_gold/` | Main 3D grid: 14 log n × 14 T × 9 log(N/Δv) = 1764 models (1741 converged), Δv = 0.3 km/s. `results/lut_dv0.30.csv` + `spectra/*.npz` | `analysis.grids.build_gold_grid` |
| `production/output_lut_fig56/` | 3D grid at S85's Figs 5/6 temperatures (18, 26, 36 K), 8 densities: 216 models | `analysis.grids.build_fig56_grid` |
| `production/output_escape1d/results_{loreau,full_original}_rates[_dv*]/` | 1D grids (`grid.csv` on the gold axes, `grid_fine.csv` dense), retrievals, Fig. 4 | `nh3hia.escape1d.grid` and the `escape1d_*` scripts |
| `production/tgs/model_files/` | Magritte model cache (`.hdf5`, deterministic names) | `run_model` |

## Which script produces what (Paper I)

| Paper I item | Script |
|---|---|
| Quadrant check (fig:quadrant_grid) | `analysis.validation.quadrant_check` |
| Interpolation accuracy (sec:grid) | `analysis.validation.interpolation_completeness` (+ `interpolation_axis_*`, `grid_completeness`) |
| Noise-injection recovery (fig:noise_recovery) | `analysis.validation.noise_recovery`, then `noise_recovery_hist` |
| τ(b) chord-law test (fig:chord_experiment) | `analysis.mesh.chord_experiment` |
| Mesh economy (fig:mesh_economy_*) | `analysis.mesh.mesh_economy`, `mesh_economy_nlte_cost`, `mesh_economy_figure`, `mesh_economy_followup` |
| LTE cube vs radial Figs 5/6 (fig:mesh_fig56) | `analysis.grids.fig56_slice --mode lte`, then `analysis.figures.fig56_slice_plot` |
| S85 Fig. 4 counterpart (fig:fig4) | `analysis.figures.fig4_magritte` (1D: `fig4_escape1d`) |
| S85 Figs 5/6 counterpart (fig:fig56_nlte, fig:fig56_scan) | `analysis.figures.fig56_magritte` (LTE reference lines from `fig56_slice --mode lte`) |
| 3D vs 1D Figs 5/6 comparison | `analysis.grids.escape1d_fig56_curves`, then `analysis.figures.fig56_compare_escape1d` / `fig56_overlay_8dens` |
| 23-position S84 retrieval, (2,1) test (tab:21test, fig:branch) | `analysis.retrieval.magritte_stutzki1984` |
| 1D retrievals (tab:escape1d) | `analysis.retrieval.escape1d_stutzki1984 --rates {loreau,full_original}` |
| S106 (0,0) branches (fig:s106_reversal) | `analysis.retrieval.s106_branch_comparison` |
| Density ceiling to log n 8.5 (sec:robustness) | `analysis.grids.escape1d_extend_decade`, then the 1D retrievals |
| Linewidth sweep (sec:robustness) | `nh3hia.escape1d.grid --fine --dv X`, the 1D retrieval with `--dv X`, then `analysis.retrieval.dv_sweep_comparison`, `analysis.figures.fig4_r10_dv_sweep` |
| W33 (tab:w33) | `analysis.retrieval.magritte_w33`, `escape1d_w33`, `w33_rates_comparison` |
| W33 estimator noise check | `analysis.validation.noise_mc_w33_linewidth` |

## Conventions and known issues (read before using the outputs)

Details and evidence are in `important_notes/magritte-quirks-2026-09-16.md`,
`mesh-comparison-and-fig56-2026-09-16.md` and `magritte-quirks-2026-09-18.md`.

- **Hyperfine labels** are assigned by fitted centre, using the component
  table in `hyperfine.py`, never by parameter order. Radio velocity
  convention: F1 0→1 is at +19.49 km/s (enhanced in the anomaly) and F1 1→0
  at −19.50 km/s (suppressed).
- **Optical depth.** S85's Figs 5/6 plot the central-chord τ.
  `run_model`'s `tau_main` is the chord value; the 1D model's `tau_main` is
  S85's radial τ_G (chord = 2×); `stutzki.physics.eq11_thermal_ratio` takes
  the radial τ.
- **The gold grid's `tau_main` column is wrong.** It is the (2,1) optical
  depth, because the value was read from whichever line was imaged last.
  The code is fixed, but those rows were not re-solved. No ratio or
  retrieval uses this column; for τ axes, use the Figs 5/6 grid.
- **The production mesh under-reports central τ by about 2×.** The default
  mesh is a Cartesian seed coarsened by Magritte's density remesher, which
  leaves the sphere's core and limb nearly empty. At fixed physical
  parameters its ratios agree with a correct radial mesh to within 3% at
  LTE. Radial and hybrid meshes are available through
  `run_model(mesh=...)` but cost 13× and 7× more per model.
- **`fov_pad_factor` changes the mesh**, not only the image field, because
  it sets the boundary radius. The grids use 1.15; match it when comparing
  runs.
- **Retrieval always goes through `lut.interpolator`** (log-amplitudes
  interpolated, ratios formed afterwards), never nearest-row scoring. The
  1D retrievals use the dense fine grid directly, because the 1D model has
  masing poles that interpolation would smear.
- **The consistency filter** (filling factor η_f ≤ 1, Jeans clump count
  K ≥ Δv_obs/0.3) is S85's post-fit branch choice. It acts as a prior for
  the high-density branch, not a measurement.

## Renamed files (for reading older notes)

The dated notes and the plan predate the 2026-09-30 reorganisation. Old names
map as follows (`w33/` and `stutzki85/` were the old subdirectories):

| Old | New |
|---|---|
| `nh3_NLTE_sphere.py` | `nh3hia/model3d/sphere.py` |
| `nh3_NLTE_analysis.py` | `nh3hia/spectral_fit.py` |
| `nh3_hyperfine.py`, `noise.py` | `nh3hia/hyperfine.py`, `nh3hia/noise.py` |
| `w33/spectrum_utils.py`, `w33/scoring.py` | `nh3hia/spectrum_utils.py`, `nh3hia/scoring.py` |
| `w33/lte_probe.py` | `nh3hia/model3d/lte_probe.py` |
| `w33/build_lut.py`, `w33/lut_interpolator.py` | `nh3hia/lut/build.py`, `nh3hia/lut/interpolator.py` |
| `w33/build_lut_gold.py`, `w33/build_lut_fig56.py` | `analysis/grids/build_gold_grid.py`, `build_fig56_grid.py` (axes in `nh3hia/lut/axes.py`) |
| `w33/stutzki_params.py`, `stutzki_physics.py`, `stutzki_tables_full.py` | `nh3hia/stutzki/params.py`, `physics.py`, `tables.py` |
| `w33/params.py`, `w33/w33_observed_ratios.py` | `nh3hia/w33/params.py`, `nh3hia/w33/observed.py` |
| `w33/observed_data/` | `data/observed/` |
| `stutzki85/nh3_escape_model.py` | `nh3hia/escape1d/model.py` |
| `stutzki85/build_escape_grid.py`, `stutzki85/grid.py` | `nh3hia/escape1d/grid.py`, `nh3hia/escape1d/curves.py` |
| `stutzki85/rate_swap_transcribed.py`, `rate_swap_test.py` | `nh3hia/escape1d/rates_stutzki.py`, `rates_table2.py` |
| `w33/retrieve_stutzki1985_full.py`, `retrieve_w33_full.py` | `analysis/retrieval/magritte_stutzki1984.py`, `magritte_w33.py` |
| `stutzki85/retrieve_stutzki1985_escape1d.py`, `retrieve_w33_escape1d.py` | `analysis/retrieval/escape1d_stutzki1984.py`, `escape1d_w33.py` |
| `stutzki85/s106_00_branch_comparison.py` | `analysis/retrieval/s106_branch_comparison.py` |
| `stutzki85/w33_rates_comparison.py`, `dv_sweep_comparison.py` | `analysis/retrieval/` (same names) |
| `w33/noise_recovery_lut.py`, `plot_noise_recovery_hist.py` | `analysis/validation/noise_recovery.py`, `noise_recovery_hist.py` |
| `w33/plot_grid_completeness.py`, `plot_interpolation_*.py` | `analysis/validation/grid_completeness.py`, `interpolation_*.py` |
| `stutzki85/interpolation_completeness_escape1d.py`, `interp_curves_escape1d.py` | `analysis/validation/escape1d_interpolation_completeness.py`, `escape1d_interpolation_curves.py` |
| `w33/noise_mc_w33_linewidth.py` | `analysis/validation/noise_mc_w33_linewidth.py` |
| `w33/chord_experiment.py`, `mesh_economy*.py`, `plot_mesh_point_clouds.py` | `analysis/mesh/` (`point_clouds.py`) |
| `w33/fig56_slice.py` | `analysis/grids/fig56_slice.py` |
| `stutzki85/extend_grid_decade.py`, `run_fig5_6_grid.py` | `analysis/grids/escape1d_extend_decade.py`, `escape1d_fig56_curves.py` |
| `w33/reproduce_stutzki_fig4.py`, `stutzki85/reproduce_fig4_escape1d.py` | `analysis/figures/fig4_magritte.py`, `fig4_escape1d.py` |
| `stutzki85/fig4_r10_dv_sweep.py` | `analysis/figures/fig4_r10_dv_sweep.py` |
| `w33/compare_fig5_6_direct.py`, `compare_fig5_6_overlay.py` | `analysis/figures/fig56_compare_escape1d.py`, `fig56_overlay_8dens.py` |
| `w33/plot_fig56_magritte.py`, `plot_fig56_overlay.py`, `plot_fig56.py` | `analysis/figures/fig56_magritte.py`, `fig56_on_scan.py`, `fig56_slice_plot.py` |
| everything else | `archive/` (see `archive/README.md`) |

The `--rates original` option (a partial reconstruction of S85's rates) was
retired. `full_original` is the complete reconstruction.
