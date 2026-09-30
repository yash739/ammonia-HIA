# Archive

Code that is no longer part of the pipeline, kept so that the dated notes
and earlier reports can still be traced to the code that produced them.
**Nothing here is maintained.** These files still use the flat,
pre-2026-09-30 import names, so most will not run as they stand. Tests
here are not collected by pytest.

Results produced by several of these scripts are known to be defective
(computed before the hyperfine-labelling, velocity-sign and `tau_main`
fixes, or on superseded grids). Don't cite them; see
`important_notes/`.

## legacy_pre_w33/

The original sphere-model grid runners from before the W33/Stutzki
pipeline: `ModelGrid*.py`, `RatioConvergenceTest.py`, the LTE sphere, the
Crapsi-profile "decay" model and its analysis, `compare_NLTE_LTE.py` and a
visualisation notebook.

## pre_lut_pipeline/

The first W33/Stutzki pipeline, which ran Magritte inside the fitting loop
instead of using a precomputed grid, plus the patchwork grids that preceded
the gold grid:

- `invert_ratios.py` (+ its tests, `test_recovery_omc_s4.py`): live-Magritte
  ratio inversion. Superseded by `nh3hia/lut/interpolator.py`.
- `validate_stutzki.py`, `stutzki_21_test.py`: the Stage 0 validation and
  the first (2,1)/(1,1) test at S85's own Table 1a parameters. Superseded
  by the retrieval-based held-out test (`analysis/retrieval/`).
- `legacy_catalogue.py` (+ test), `plot_chi2_landscapes.py`: the pre-fix
  878-row catalogue with its outer-satellite swap corrected on read.
- `build_lut_offset.py`, `build_lut_offset2.py`, `_smoke_fast_point.py`:
  the coarse/offset grids, merged into the gold grid.
- `imaging.py`: beam-convolution helpers for a never-used "Stage B".
- `PLAN_lut_interpolation_mcmc.md`: the original grid/interpolation plan.

## escape1d_early/

Early work with the 1D escape-probability model:

- `compare_magritte.py`, `compare_magritte_heatmap.py`: comparison against a
  pre-fix Magritte run (quarantined).
- `make_figures.py`, `reproduce_raw_fig4a.py`, `reproduce_raw_fig4_full.py`:
  first Fig. 4–6 reproductions (with a hybrid rate matrix).
- `fit_observations.py`: first fit to S84. Superseded by
  `analysis/retrieval/escape1d_stutzki1984.py`.
- `rate_swap_full.py`: the partial reconstruction of S85's rates (the
  retired `--rates original`). Superseded by
  `nh3hia/escape1d/rates_stutzki.py`.
