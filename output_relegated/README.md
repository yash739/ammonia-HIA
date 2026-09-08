# Relegated output directories

- **`invert_verify_prefix_runaway/`** — output of the first (v1) end-to-end
  verification run of `invert_ratios.py`, using a known Stutzki OMC S4
  benchmark as synthetic ground truth. This run was killed mid-flight after a
  stale-round bounds-expansion bug (fixed in v2 — see `stale_retries` in
  `invert_ratios.py`) drove the n_H2 search bounds to ~1e113 cm^-3. Kept only
  as a diagnostic record: this run is also what first exposed a real,
  reproducible degenerate-solution problem (a badly-wrong high-density branch
  scoring nearly as well as the correct answer under ratio-only chi2), which
  motivated replicating Stutzki & Winnewisser (1985)'s own eta_f/Jeans-mass
  degeneracy-resolution checks (now in `stutzki_physics.py`).
