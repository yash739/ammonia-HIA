# Solver speed-ups: what was validated, what was adopted (2026-10-07)

Data: `report.md` (stage/ablation tables vs gold), `mesh_report.md` (cube vs radial), per-variant CSVs in `v_*/`,
`mesh_*/`. Reference = the existing gold rows (cube mesh, legacy solver); acceptance pre-registered in
`analysis/validation/speedup_report.py` (|dR/R| <= 0.5% for the satellite and (2,2) ratios, A_MAIN <= 1%,
R_21_11 <= 1% where detected, convergence parity). Gold rows whose (2,2)/(2,1) values are ripple-floor upper
limits (extractor v1, undetected (2,1)) are not comparable and are excluded.

## Verdict per lever (48 stratified nodes unless stated)
| lever | justification (a priori) | result | decision |
|---|---|---|---|
| `close`: sum only nearby lines | Gaussian tail beyond 10 widths < e^-100; distinct lines are > 25 widths apart except true hyperfine blends, which stay inside the window; it is Magritte's upstream default (our vendored copy has a local edit to `true`) | 46 nodes, **0 failures, max change 0.04%**, 8.5x median (6.3-10x); cutoff 5/20 widths identical at the nodes tested | **adopt** |
| `lev36`: 36-level LAMDA (77 of 115 lines) | levels 37-48 are >= 263 K above ground; hold <= 1e-7 of the NLTE population except hot dense gas | **0.00-0.4% for log N/dv <= 15.8, but 2-4% at cold, very-high-column nodes** (6.5/18 K/16.25: A_MAIN +3.6%, R_01 -2.2%; confirmed with a 1e-8 precision ladder: lev36 differs, lev44 does not, so it is truncation not convergence; lev40 still 0.4-1.5%) | **adopt only for log N/dv <= 15.8** (`solver_profiles.regional`); all 48 levels above |
| `nq12` | 12-point Gauss-Hermite exact to degree 23 for a smooth profile; weak point saturated/masing profiles | 0 failures on 16 nodes, max 0.43% (R_21_11), 1.6x; nq8 earlier gave 1.7-2.9% at masing nodes (rejected); nq10 +1.0% at one node | **adopt** |
| `prec4` (stop at 1e-4) | residual after stop ~ delta*rho/(1-rho) = 0.7-3.5e-4 | max change **0.04%** vs a 1e-8 truth on the nodes completed (5 incl. cold); `close` (1e-6) and the 1e-8 truth are identical to 0.000%; also converges the cold corner that never converged at 1e-6 | **adopt** |
| fixed Ng (`ngfixed`, mem 15) | changes only the path to the same fixed point | identical results, **no net speed-up** over 16 nodes (1.0x, 0.7-1.5x) | **drop** |
| `max_distance_opacity_contribution` 5/20, `min_rel_pop_for_convergence`, `max_interpolation_diff` | -- | dist5/dist20 identical at the one node measured earlier (not re-run on more nodes: time); the other two were not tested | not adopted, untested |

Accepted fast profile: `close+lev36+nq12+prec4` for log N/dv <= 15.8 (38 validation nodes: **0 failures, max 0.48%,
median 25x faster than gold, range 16-228x**) and `close+nq12+prec4` (all levels) above (10 nodes: **0 failures, max 0.39%,
median 14x, 12-169x**). Known residual caveats, stated rather than hidden: (i) cold, thin or very-high-column nodes that
gold itself barely converged (e.g. log n 3.5, 9 K, log N/dv 16.25: 94.6% in gold, 86-99% here) are path dependent at the
percent level for any profile; (ii) at T = 48 K, log N/dv = 16.25 lev36 would have been +0.54%, which the region rule avoids.

## Cube vs radial with the fast profile (24 / 16 nodes; `mesh_report.md`)
Cube is 3.3% (R_01), 5.8% (R_10), 2.1% (R_12), 3.9% (R_21) median different from radial 12x140 over all 24 nodes
(up to ~37% at individual cold/masing nodes); radial 12x140 vs radial 14x260 (16 cheaper nodes): median 1.2% (R_01),
0.97% (R_10), 0.35% (R_12), 0.23% (R_21), max 15%; gold vs radial 14x260 behaves like cube vs radial 14x260. Radial 12x140
costs 4.4x the cube (935 vs 377 points); the full speed-up vs gold is 6.1x (radial 12x140) against 26.7x (cube).

## Rebuild (radial 12x140, regional fast profile, 2464 nodes)
Cost model (log-log fit of radial wall to gold wall, 24 nodes): 26 h on 3 processes (35 h by ratio-of-sums). Launched
Wed 2026-10-07 08:14 in `screen -S lut_gold_radial` into `production/output_lut_gold_radial/` (CSV, spectra,
populations; model files deleted), coarse-to-fine order, 4 processes x 3 threads after the first 5 minutes; progress and
ETA: `python3 -m analysis.grids.rebuild_progress`. Gold is untouched (checksum in `gold_csv.md5`).
