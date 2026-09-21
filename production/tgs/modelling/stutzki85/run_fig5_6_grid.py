"""Paper I, Work item 3: model curves at Stutzki & Winnewisser (1985)'s own
exact Fig. 5/6 parameters (grid.FIG56_TEMPS/FIG56_LOG_N_H2: T_k in
{18,26,36} K, n'_H2 in {10^3.5,10^5.0,10^7.0} cm^-3, N_NH3 swept over
10^13.7-10^15.1 cm^-2 at Delta v = 0.3 km/s), via grid.compute_fig56_curves,
under two different collision-rate sets:

- 'full_original': the full_original (95% Stutzki-sourced) collision-rate
  reconstruction (rate_swap_transcribed.py) -- a faithful stand-in for his
  own published curves, no digitization needed.
- 'loreau': the modern hyperfine-resolved NH3-H2 rates (Loreau et al. 2023)
  used throughout the rest of this project's escape1d work (and matching
  what the Magritte 3D NLTE leg uses internally). Comparing the two rate
  sets at fixed method (1D escape probability) is the complementary test to
  gold-vs-escape1d (method-swap, rates-fixed) already in the paper -- it
  isolates how much of any escape1d-vs-Magritte residual is attributable to
  the rates versus the radiative-transfer treatment.

guard_masers=False throughout (per standing instruction across this
project's escape1d work): masing groups report the literal, un-sign-checked
Eq.(10) brightness.

Cheap: 9 (T,n) combos x 80 log_N points = 720 solves, seconds total per
rate set.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nh3_escape_model as m
import grid
import rate_swap_transcribed as rst_full
from build_escape_grid import outdir_for

CMAT_BUILDERS = {
    'loreau': lambda model, T: model.collision_matrix(T),
    'full_original': rst_full.build_complete_original_collision_matrix,
}


def out_csv_for(rates):
    return os.path.join(outdir_for(rates), 'fig56_curves.csv')


def run(rates):
    model = m.NH3Model()
    m.assert_expected_grouping(model)

    t0 = time.time()
    df = grid.compute_fig56_curves(
        model, Cmat_builder=CMAT_BUILDERS[rates],
        solver_kwargs={'guard_masers': False})
    print(f"Fig5/6 escape1d ({rates} rates, Stutzki's own exact params): "
          f"{len(df)} points in {time.time()-t0:.1f}s, "
          f"{(~df['converged']).sum()} unconverged, {df['any_maser'].sum()} maser points")

    out_csv = out_csv_for(rates)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--rates', choices=['loreau', 'full_original', 'both'], default='both')
    a = ap.parse_args()
    rates_list = ['loreau', 'full_original'] if a.rates == 'both' else [a.rates]
    for rates in rates_list:
        run(rates)


if __name__ == '__main__':
    main()
