"""Phase 5: grid generation for Stutzki & Winnewisser (1985) Figs. 4-6, using
nh3_escape_model directly (no Magritte, no meshing/imaging -- see that
module's docstring). Each grid point is one 36-level statistical-equilibrium
root-find; walking each (T_k) slice's (log n_H2, log N_NH3/dv) plane in a
boustrophedon (snake) path and warm-starting every point from its immediate
predecessor's converged populations is what makes a several-thousand-point
grid run in a couple of minutes on a laptop rather than tens of minutes: a
warm-started solve converges in ~30-60 residual evaluations instead of ~150
from a cold thermal guess (measured: ~7 ms vs ~200 ms per point).

Grid ranges match the paper's own stated parameter space (Sect. 3, p.17) and
Fig. 4's own axes -- NOT stutzki_model_1985.md's Phase 5 verbatim, which
misstates the N_NH3/Delta v lower bound as 13.7 (that is actually the lower
bound of N_NH3 ALONE, per the Fig. 5/6 caption's "1e13.7 - 1e15.1 cm^-2";
dividing by Delta v_clump=0.3 km/s gives log(N_NH3/dv) = 13.7 + log10(1/0.3)
= 14.22, matching Fig. 4's own y-axis exactly).
"""
import time

import numpy as np
import pandas as pd

import nh3_escape_model as m

FIG4_TEMPS = (18.0, 24.0, 30.0, 36.0)
FIG4_LOG_N_H2 = (3.5, 7.0, 0.05)       # (min, max, step), log10(cm^-3)
FIG4_LOG_N_DV = (14.2, 15.6, 0.05)     # (min, max, step), log10(cm^-2 s km/s)

FIG56_TEMPS = (18.0, 26.0, 36.0)
FIG56_LOG_N_H2 = (3.5, 5.0, 7.0)
FIG56_LOG_N_COLUMN_RANGE = (13.7, 15.1)  # log10(N_NH3 [cm^-2]) -- paper's own Fig.5/6 range
FIG56_N_POINTS = 80


def _arange_inclusive(lo, hi, step):
    n = int(round((hi - lo) / step)) + 1
    return np.linspace(lo, hi, n)


def _solve_with_fallback(model, Cmat, T_k, n_H2, log_N_dv, x0, solver_kwargs):
    """Warm start from x0, but fall back to (and keep, if better) a cold
    thermal-start solve when the warm start doesn't actually converge --
    never blindly overwrite a converged warm-start result with a possibly
    worse cold one."""
    out = m.run_one(model, Cmat, T_k=T_k, n_H2=n_H2, log_N_dv=log_N_dv,
                     x0=x0, **solver_kwargs)
    retried = False
    if not out['converged']:
        out_cold = m.run_one(model, Cmat, T_k=T_k, n_H2=n_H2, log_N_dv=log_N_dv,
                              x0=None, **solver_kwargs)
        retried = True
        if out_cold['max_residual'] < out['max_residual']:
            out = out_cold
    return out, retried


def compute_fig4_grid(model, temps=FIG4_TEMPS, log_n_h2_range=FIG4_LOG_N_H2,
                       log_n_dv_range=FIG4_LOG_N_DV, verbose=True,
                       solver_kwargs=None, Cmat_builder=None):
    """For each (T_k, n_H2) row, N_NH3/dv is ALWAYS swept ascending (low
    column density -> high), starting each row from a cold thermal guess at
    the lowest (safely unambiguous, low-tau) N. This model has genuine
    bistability at high optical depth in parts of this grid (independently
    consistent with what tgs/modelling/w33/stutzki_physics.py's own
    docstring calls "the exact same kind of degenerate double-minimum" found
    in the W33 inversion fits) -- an earlier version of this function used a
    boustrophedon path that alternated the N_dv sweep direction by row
    parity, so adjacent n_H2 rows approached a given bifurcation from
    opposite sides of N and could land on opposite branches, showing up as a
    jagged, unphysical-looking seam of discontinuous values in the Fig. 4
    reproduction. Always building up FROM the unphysical-choice-free
    low-column-density end tracks one consistent branch (the one continuously
    connected to the unique low-tau solution) instead of whichever branch a
    sweep happened to approach from. Warm-starting is still used within each
    row (ascending N) and carried across rows (in n_H2) for speed; only the
    N-direction is fixed, not the n_H2 order."""
    solver_kwargs = solver_kwargs or {}
    Cmat_builder = Cmat_builder or (lambda model, T: model.collision_matrix(T))
    log_n_h2_vals = _arange_inclusive(*log_n_h2_range)
    log_n_dv_vals = _arange_inclusive(*log_n_dv_range)

    rows = []
    for T in temps:
        t_start = time.time()
        Cmat = Cmat_builder(model, T)
        x0 = None
        n_reconverged = 0
        for log_n in log_n_h2_vals:
            n_H2 = 10.0 ** log_n
            for log_N in log_n_dv_vals:  # always ascending
                out, retried = _solve_with_fallback(model, Cmat, T, n_H2, log_N, x0, solver_kwargs)
                n_reconverged += int(retried)
                x0 = out['x']
                rows.append(dict(T_k=T, log_n_H2=log_n, log_N_dv=log_N, n_H2=n_H2,
                                  **{k: v for k, v in out.items() if k != 'x'}))
        if verbose:
            dt = time.time() - t_start
            n_pts = len(log_n_h2_vals) * len(log_n_dv_vals)
            print(f"T_k={T:.0f}K: {n_pts} points in {dt:.1f}s "
                  f"({1000 * dt / n_pts:.2f} ms/pt), {n_reconverged} cold retries")

    df = pd.DataFrame(rows)
    return df


def compute_fig56_curves(model, temps=FIG56_TEMPS, log_n_h2_list=FIG56_LOG_N_H2,
                          log_n_column_range=FIG56_LOG_N_COLUMN_RANGE,
                          n_points=FIG56_N_POINTS, dv_clump_kms=m.DV_CLUMP_DEFAULT_KMS,
                          verbose=True, solver_kwargs=None):
    """Sweeps N_NH3 (column density, NOT N/dv) over the paper's own Fig.5/6
    range at fixed (T_k, n_H2), converting to the N/dv the solver actually
    takes via the fixed Delta v_clump. tau_main is an OUTPUT here (the
    natural x-axis of Figs 5-6), not a grid input."""
    solver_kwargs = solver_kwargs or {}
    log_N_col_vals = np.linspace(log_n_column_range[0], log_n_column_range[1], n_points)
    log_N_dv_vals = log_N_col_vals - np.log10(dv_clump_kms)

    rows = []
    for T in temps:
        Cmat = model.collision_matrix(T)
        for log_n in log_n_h2_list:
            n_H2 = 10.0 ** log_n
            x0 = None
            for log_N_dv in log_N_dv_vals:  # always ascending, matches compute_fig4_grid
                out, _ = _solve_with_fallback(model, Cmat, T, n_H2, log_N_dv, x0, solver_kwargs)
                x0 = out['x']
                rows.append(dict(T_k=T, log_n_H2=log_n, log_N_dv=log_N_dv, n_H2=n_H2,
                                  **{k: v for k, v in out.items() if k != 'x'}))
        if verbose:
            print(f"Fig5/6: T_k={T:.0f}K done")

    df = pd.DataFrame(rows)
    return df


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='.')
    a = ap.parse_args()

    model = m.NH3Model()
    m.assert_expected_grouping(model)

    t0 = time.time()
    df4 = compute_fig4_grid(model)
    print(f"Fig4 grid: {len(df4)} points in {time.time()-t0:.1f}s, "
          f"{(~df4['converged']).sum()} unconverged, {df4['any_maser'].sum()} maser points")
    df4.to_csv(f"{a.out_dir}/stutzki85_fig4_grid.csv", index=False)

    t0 = time.time()
    df56 = compute_fig56_curves(model)
    print(f"Fig5/6 curves: {len(df56)} points in {time.time()-t0:.1f}s, "
          f"{(~df56['converged']).sum()} unconverged, {df56['any_maser'].sum()} maser points")
    df56.to_csv(f"{a.out_dir}/stutzki85_fig56_curves.csv", index=False)
