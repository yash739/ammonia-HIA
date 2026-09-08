# Relegated scripts

These four scripts represent superseded lines of analysis, kept for record
but no longer part of the live pipeline. Nothing in `tgs/modelling/w33/`
imports from this directory.

- **`screen.py`** — Stage A T_mb/tau/N screening helpers. Only ever imported
  by `ratio_screen.py`, relegated alongside it.

- **`ratio_screen.py`** — ladder-ratio screening scored with cosine
  similarity. Superseded because cosine similarity was found (twice) to be a
  poor discriminator on hyperfine ratio vectors: it lets one dominant
  component mask a badly-wrong component, and it's nearly blind to ratios
  that move proportionally with density. Replaced by `weighted_chi2` in
  `../scoring.py`. The one piece of this file that was still load-bearing —
  `_native_peak_tmb`, used by `invert_ratios.py` to extract the (2,2)
  main-line peak — has been extracted as a public function,
  `native_peak_tmb`, into `../spectrum_utils.py`. `invert_ratios.py` imports
  from there now, not from this file.

- **`run_grid_w33.py`** — the W33 grid-builder script used to produce
  `output_w33_grid_extension/results/NLTE_nh3_w33_extension.csv`. The script
  itself has no live importers; its output CSV is still read (as data, not
  code) by `invert_ratios.py`'s `catalogue_lookup` for seeding search bounds.

- **`reprocess_21_32.py`** — one-off post-hoc CSV-fix script for the (2,1)/
  (3,2) columns of an earlier catalogue run. Fully obsolete, zero live
  imports.

The current pipeline lives in `invert_ratios.py`, `scoring.py`, and
`stutzki_physics.py`, replicating Stutzki & Winnewisser (1985)'s own
ratio-inversion and degeneracy-resolution method rather than the ad hoc
cosine-similarity screening these relegated scripts implemented.
