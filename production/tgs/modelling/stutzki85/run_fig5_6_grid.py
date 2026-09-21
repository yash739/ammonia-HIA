"""Paper I, Work item 3: a direct stand-in for Stutzki & Winnewisser (1985)'s
own Fig. 5/6 model curves -- no digitization needed, since the full_original
collision-rate reconstruction (rate_swap_transcribed.py, 95% Stutzki-sourced)
is a faithful reimplementation of his actual escape-probability method. This
runs it at his own exact Fig. 5/6 parameters (grid.FIG56_TEMPS/FIG56_LOG_N_H2,
matching his caption verbatim: T_k in {18,26,36} K, n'_H2 in
{10^3.5,10^5.0,10^7.0} cm^-3, N_NH3 swept over 10^13.7-10^15.1 cm^-2 at
Delta v = 0.3 km/s) via grid.compute_fig56_curves, which already implements
this exact sweep (built for the Loreau-rates case originally; only the
collision matrix changes here).

guard_masers=False throughout (per standing instruction across this
project's escape1d work): masing groups report the literal, un-sign-checked
Eq.(10) brightness.

Cheap: 9 (T,n) combos x 80 log_N points = 720 solves, seconds total.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nh3_escape_model as m
import grid
import rate_swap_transcribed as rst_full
from build_escape_grid import outdir_for

OUT_CSV = os.path.join(outdir_for('full_original'), 'fig56_curves.csv')


def main():
    model = m.NH3Model()
    m.assert_expected_grouping(model)

    t0 = time.time()
    df = grid.compute_fig56_curves(
        model, Cmat_builder=rst_full.build_complete_original_collision_matrix,
        solver_kwargs={'guard_masers': False})
    print(f"Fig5/6 escape1d (full_original rates, Stutzki's own exact params): "
          f"{len(df)} points in {time.time()-t0:.1f}s, "
          f"{(~df['converged']).sum()} unconverged, {df['any_maser'].sum()} maser points")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"saved {OUT_CSV}")


if __name__ == '__main__':
    main()
