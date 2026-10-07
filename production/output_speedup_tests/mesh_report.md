# Mesh comparison with the fast solver profile (regional: lev36 for log N/dv <= 15.8)

Reference: radial 14x260 (16 nodes, cheaper nodes only). Differences in per cent.

| cube vs radial 14x260 | nodes | R_01_MAIN | R_10_MAIN | R_12_MAIN | R_21_MAIN | R_22_MAIN | R_21_11 | HIA_IS | HIA_OS |
|---|---|---|---|---|---|---|---|---|---|
| median / max \|Δ\| % | 16 | 3.81 / 82.7 (n=16) | 6.03 / 82.7 (n=16) | 1.63 / 75.2 (n=16) | 3.74 / 75.3 (n=16) | 1.06 / 91.8 (n=15) | 13.01 / 80.8 (n=11) | 0.97 / 6.2 (n=16) | 3.33 / 26.6 (n=16) |
| radial 12x140 vs radial 14x260 | nodes | R_01_MAIN | R_10_MAIN | R_12_MAIN | R_21_MAIN | R_22_MAIN | R_21_11 | HIA_IS | HIA_OS |
|---|---|---|---|---|---|---|---|---|---|
| median / max \|Δ\| % | 16 | 1.23 / 15.2 (n=16) | 0.97 / 15.2 (n=16) | 0.35 / 13.8 (n=16) | 0.23 / 13.9 (n=16) | 0.11 / 19.2 (n=15) | 0.59 / 19.3 (n=11) | 0.13 / 1.6 (n=16) | 0.64 / 5.8 (n=16) |

| gold (cube, legacy) vs radial 14x260 | nodes | R_01_MAIN | R_10_MAIN | R_12_MAIN | R_21_MAIN | R_22_MAIN | R_21_11 | HIA_IS | HIA_OS |
|---|---|---|---|---|---|---|---|---|---|
| median / max \|Δ\| % | 16 | 3.84 / 82.7 (n=16) | 6.07 / 82.7 (n=16) | 1.70 / 75.2 (n=16) | 3.75 / 75.3 (n=16) | 1.09 / 91.8 (n=15) | 12.99 / 80.8 (n=11) | 0.95 / 6.3 (n=16) | 3.33 / 26.7 (n=16) |

## Cube vs radial 12x140 over all 24 mesh nodes

| radial 12x140 vs cube | nodes | R_01_MAIN | R_10_MAIN | R_12_MAIN | R_21_MAIN | R_22_MAIN | R_21_11 | HIA_IS | HIA_OS |
|---|---|---|---|---|---|---|---|---|---|
| median / max \|Δ\| % | 24 | 3.26 / 37.0 (n=23) | 5.80 / 36.9 (n=23) | 2.12 / 35.0 (n=23) | 3.94 / 35.1 (n=23) | 1.35 / 37.9 (n=22) | 11.01 / 157.7 (n=15) | 1.20 / 36.9 (n=24) | 6.98 / 29.4 (n=24) |

By stratum (median |Δ| % of R_01 / R_10 / R_21_11):

- cold_lowdens: 4 nodes, 18.4 / 8.2 / nan
- corner: 4 nodes, 5.4 / 5.7 / 19.7
- fill: 1 nodes, 0.3 / 2.1 / nan
- hot_dense: 3 nodes, 1.1 / 5.8 / 7.2
- masing: 3 nodes, 2.3 / 19.8 / 106.4
- retrieval: 5 nodes, 3.7 / 5.3 / 10.6
- slice: 2 nodes, 2.8 / 7.2 / 80.4
- thin: 2 nodes, 1.5 / 0.7 / 15.0

## Measured cost (fast profile, 3 processes x 4 threads)

| mesh | nodes | points | median iterations | wall sum [s] | gold wall sum [s] | speed-up vs gold |
|---|---|---|---|---|---|---|
| cube | 24 | 377 | 22 | 1161 | 30992 | 26.7 |
| radial 12x140 | 24 | 935 | 20 | 5113 | 30992 | 6.1 |
| radial 14x260 | 16 | 1702 | 17 | 3227 | 7895 | 2.4 |

Radial 12x140 cost model (log-log fit to gold wall time, slope 0.68): full 2464-node table = 77 core-hours of jobs -> 25.8 h on 3 processes, 19.4 h on 4; ratio-of-sums estimate 34.8 h on 3.
