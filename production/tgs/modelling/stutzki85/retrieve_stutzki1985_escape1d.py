"""Same chi^2 retrieval diagnostic as w33/retrieve_stutzki1985_full.py, but
against the 1D escape-probability model (Stutzki & Winnewisser 1985's own
method, modern Loreau rates) instead of the Magritte 3D LUT -- to see whether
the retrieval behaviour (branch structure, best-fit values, (2,1) prediction)
looks the same on "his" model as it does on the full 3D non-LTE one.

guard_masers=False throughout (instructed): masing groups report the
literal, un-sign-checked Eq.(10) brightness rather than NaN. This can drive
some ratios to genuine numerical poles near the masing threshold (R_01 up to
~1e13 was seen on the coarse grid) -- a LinearNDInterpolator would smear that
pole across its entire local neighbourhood, not just the one point, so this
uses a FINE DIRECT grid (build_escape_grid.py --fine, ~126000 points) instead
of interpolation: the model is cheap enough (~9 ms/point) that a fine grid
costs minutes, and a pole then stays an isolated bad-chi2 cell that simply
never wins the argmin, rather than corrupting its neighbours.

No radius mask (doesn't apply -- this model has no mesh/geometry, only the
36-level statistical equilibrium + escape-probability closure) and no
convergence-percentage filter (this solver is pass/fail, not partial -- see
nh3_escape_model.solve_populations); rows are kept iff converged=True.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_W33 = os.path.join(os.path.dirname(_HERE), 'w33')
sys.path.insert(0, _HERE)
sys.path.insert(0, _W33)
sys.path.insert(0, os.path.dirname(_HERE))

import stutzki_params as sp
import nh3_hyperfine as hf
from lut_interpolator import chi2_retrieve
from stutzki_physics import eta_f as calc_eta_f, clump_count_K, clump_count_required, physically_consistent
from retrieve_stutzki1985_full import gaussian_spectrum, chi2_slice_plot, OFFSETS
from build_escape_grid import out_csv_for, outdir_for, DV_CLUMP_KMS

RATIO_KEYS = ('R_01', 'R_10', 'R_21', 'R_12', 'R_2211')
STUTZKI_KEY = {'R_01': 'R_01', 'R_10': 'R_10', 'R_21': 'R_21', 'R_12': 'R_12', 'R_2211': 'R_22_MAIN'}
RATIO_21 = 'R_21_11'


def load_grid(path):
    """Keep the full regular (log_n, T, log_Ndv) cross product -- the
    chi2 landscape reshape below needs a genuine full grid. Unconverged
    rows (2723/126000, 2.2%) get their ratio columns NaNed instead of
    being dropped, so chi2_retrieve's isnan check scores them +inf
    (never wins the argmin) without breaking the grid shape.
    """
    df = pd.read_csv(path)
    ratio_cols = ['R_01', 'R_10', 'R_12', 'R_21', 'R_2211', 'R_21_11', 'T_B_main']
    df.loc[~df['converged'], ratio_cols] = np.nan
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--rates', choices=['loreau', 'original', 'full_original'], default='loreau')
    ap.add_argument('--grid', default=None, help='override the grid path implied by --rates')
    ap.add_argument('--out-tag', default=None,
                    help='suffix for the output dir name (e.g. "ext8.5"), so results against a '
                         'non-standard grid (--grid) do not overwrite the standard ones')
    a = ap.parse_args()

    grid_path = a.grid or out_csv_for(a.rates, fine=True)
    dirname = 'stutzki1985_retrieval' + (f'_{a.out_tag}' if a.out_tag else '')
    OUTDIR = os.path.join(outdir_for(a.rates), dirname)
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"loading fine grid ({a.rates})...", flush=True)
    df = load_grid(grid_path)
    pts = df[['log_n_H2', 'T_cloud', 'log_N_dv']].values
    ratio_grid = {k: df[k].values for k in RATIO_KEYS}
    ratio_grid['A_MAIN'] = df['T_B_main'].values
    ratio_grid[RATIO_21] = df[RATIO_21].values

    log_n_ax = np.sort(df.log_n_H2.unique())
    T_ax = np.sort(df.T_cloud.unique())
    ndv_ax = np.sort(df.log_N_dv.unique())
    shape3d = (len(log_n_ax), len(T_ax), len(ndv_ax))
    print(f"  {len(df)} converged rows, grid shape {shape3d} "
          f"(any_maser: {df.any_maser.mean():.1%})", flush=True)
    # sanity: the grid must actually be a full regular cross product for the
    # reshape below to be valid.
    assert len(df) == np.prod(shape3d), (len(df), shape3d)
    order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
    pts, ratio_grid = pts[order], {k: v[order] for k, v in ratio_grid.items()}

    summary = []
    for region, pos in sp.RETRIEVAL_TEST_POSITIONS:
        entry = sp.TABLE_2_1984.get((region, pos))
        a1a = sp.TABLE_1A.get(region, {}).get(pos)
        if entry is None:
            continue
        tag = f"{region}_{pos}".replace(',', '_').replace(' ', '')
        print(f"=== {region} {pos} ===", flush=True)

        obs = {k: entry.get(STUTZKI_KEY[k], np.nan) for k in RATIO_KEYS}
        err = {k: entry.get(STUTZKI_KEY[k] + '_err', np.nan) for k in RATIO_KEYS}
        obs_vec = np.array([obs[k] for k in RATIO_KEYS])
        sigma_vec = np.array([err[k] for k in RATIO_KEYS])

        best_idx, chi2_best, chi2_all = chi2_retrieve(None, pts, ratio_grid, obs_vec, sigma_vec,
                                                       ratio_keys=RATIO_KEYS)
        log_n_fit, T_fit, ndv_fit = pts[best_idx]
        A_MAIN_fit = float(ratio_grid['A_MAIN'][best_idx])
        R21_11_pred = float(ratio_grid[RATIO_21][best_idx])
        model_ratios = {k: float(ratio_grid[k][best_idx]) for k in RATIO_KEYS}

        T_B_obs = entry['T_B_11']
        dist = sp.SOURCE_DISTANCE_PC.get(region, 500.0)
        dv_obs = entry['dv']
        ef = calc_eta_f(T_B_obs, A_MAIN_fit)
        K_req = clump_count_required(dv_obs, DV_CLUMP_KMS)
        K_pred = clump_count_K(T_fit, 10 ** log_n_fit, ef, dist, dv_obs, DV_CLUMP_KMS) if ef and ef > 0 else np.nan
        consistent, detail = physically_consistent(T_fit, 10 ** log_n_fit, T_B_obs, A_MAIN_fit,
                                                    dv_obs, DV_CLUMP_KMS, dist)

        v = np.linspace(-32, 32, 800)
        obs_amps = [T_B_obs * entry['R_10'], T_B_obs * entry['R_12'], T_B_obs,
                    T_B_obs * entry['R_21'], T_B_obs * entry['R_01']]
        obs_spec = gaussian_spectrum(v, obs_amps, dv_obs)
        model_amps = [A_MAIN_fit * model_ratios['R_10'], A_MAIN_fit * model_ratios['R_12'], A_MAIN_fit,
                      A_MAIN_fit * model_ratios['R_21'], A_MAIN_fit * model_ratios['R_01']]
        model_spec = gaussian_spectrum(v, model_amps, DV_CLUMP_KMS)

        t321 = sp.TABLE_3_21.get((region, pos))

        chi2_3d = chi2_all.reshape(shape3d)
        i_n = int(np.argmin(np.abs(log_n_ax - log_n_fit)))
        i_T = int(np.argmin(np.abs(T_ax - T_fit)))
        i_d = int(np.argmin(np.abs(ndv_ax - ndv_fit)))

        fig = plt.figure(figsize=(21, 10))
        gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.9)

        ax_spec = fig.add_subplot(gs[0, 0:3])
        ax_spec.plot(v, obs_spec, color='k', lw=1.6, label=f'observed (FWHM={dv_obs:.2f} km/s)')
        ax_spec.plot(v, model_spec, color='tab:blue', lw=1.4, ls='--',
                    label=f'best-fit model (FWHM={DV_CLUMP_KMS} km/s)')
        ax_spec.set_xlabel('v [km/s]', fontsize=9)
        ax_spec.set_ylabel(r'$T_B$ [K]', fontsize=9)
        ax_spec.set_title(f'{region} {pos}: (1,1) spectrum [escape-prob. model]', fontsize=10)
        ax_spec.legend(fontsize=7)
        ax_spec.grid(alpha=0.25)

        obs_spec_norm = gaussian_spectrum(v, [x / T_B_obs for x in obs_amps], dv_obs)
        model_spec_norm = gaussian_spectrum(v, [x / A_MAIN_fit for x in model_amps], DV_CLUMP_KMS)
        ax_spec_n = fig.add_subplot(gs[0, 3:6])
        ax_spec_n.plot(v, obs_spec_norm, color='k', lw=1.6, label='observed / $T_{B,obs}$')
        ax_spec_n.plot(v, model_spec_norm, color='tab:blue', lw=1.4, ls='--', label='model / $T_{B,theor}$')
        ax_spec_n.axhline(1.0, color='gray', lw=0.6, ls=':')
        ax_spec_n.set_xlabel('v [km/s]', fontsize=9)
        ax_spec_n.set_ylabel(r'$T_B\,/\,T_B(\mathrm{main})$', fontsize=9)
        ax_spec_n.set_title('main-line-normalized (ratio shape)', fontsize=10)
        ax_spec_n.legend(fontsize=7)
        ax_spec_n.grid(alpha=0.25)

        ax_txt = fig.add_subplot(gs[0, 6:9])
        ax_txt.axis('off')
        lines = [
            f"$\\chi^2$ = {chi2_best:.3f}",
            "",
            f"log $n_{{H_2}}$ = {log_n_fit:.3f}" + (f"   (Table 1a: {a1a['log_nH2']:.3f})" if a1a else ""),
            f"$T_k$ = {T_fit:.2f} K" + (f"   (Table 1a: {a1a['T_k']:.2f} K)" if a1a else ""),
            f"log($N_{{NH_3}}/\\Delta v$) = {ndv_fit:.3f}",
            "",
            f"$T_{{B,obs}}$ = {T_B_obs:.3f} K   $T_{{B,theor}}$ = {A_MAIN_fit:.3f} K",
            f"$\\eta_f$ = {ef:.3f}" + ("  (>1, UNPHYSICAL)" if (ef and ef > 1) else ""),
            f"K predicted = {K_pred:.2f}   K required = {K_req:.2f}",
            f"physically consistent: {'YES' if consistent else 'NO'}",
            f"any_maser at best fit: {'YES' if df['any_maser'].values[order][best_idx] else 'no'}",
        ]
        ax_txt.text(0.02, 0.95, "\n".join(lines), va='top', ha='left', fontsize=9.5,
                    transform=ax_txt.transAxes, family='monospace')

        ax21 = fig.add_subplot(gs[0, 9:12])
        labels, vals, colors = ['predicted'], [R21_11_pred], ['tab:blue']
        if t321 is not None:
            labels += ['Stutzki\ntheory', 'Stutzki\nobserved']
            vals += [t321['ratio_theor'], t321['ratio_obs']]
            colors += ['gray', 'black']
        ax21.bar(labels, vals, color=colors, alpha=0.85)
        ax21.set_ylabel(r'$T_B(2,1)/T_B(1,1)$', fontsize=9)
        ax21.set_title('(2,1)/(1,1): held-out prediction', fontsize=10)
        for i, val in enumerate(vals):
            ax21.text(i, val, f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        if t321 is None:
            ax21.text(0.5, 0.5, 'no Table 3 (2,1)\ndata for this position', ha='center', va='center',
                      transform=ax21.transAxes, fontsize=8, color='gray')

        chi2_slice_plot(fig.add_subplot(gs[1, 0:4]), log_n_ax, T_ax, chi2_3d, 2,
                        r'log $n_{H_2}$', r'$T_k$ [K]', (log_n_fit, T_fit), chi2_best,
                        (a1a['log_nH2'], a1a['T_k']) if a1a else None,
                        title=r'$T_k$ vs log($n_{H_2}$), profile min over log(N/dv)')
        chi2_slice_plot(fig.add_subplot(gs[1, 4:8]), T_ax, ndv_ax, np.moveaxis(chi2_3d, 0, -1), 2,
                        r'$T_k$ [K]', r'log($N_{NH_3}/\Delta v$)', (T_fit, ndv_fit), chi2_best,
                        None, title=r'log(N/dv) vs $T_k$, profile min over log($n_{H_2}$)')
        chi2_slice_plot(fig.add_subplot(gs[1, 8:12]), log_n_ax, ndv_ax, chi2_3d, 1,
                        r'log $n_{H_2}$', r'log($N_{NH_3}/\Delta v$)', (log_n_fit, ndv_fit), chi2_best,
                        None, title=r'log(N/dv) vs log($n_{H_2}$), profile min over $T_k$')

        out_png = os.path.join(OUTDIR, f'{tag}.png')
        plt.savefig(out_png, dpi=140)
        plt.close(fig)
        print(f"  saved {out_png}  chi2={chi2_best:.3f}  log_n={log_n_fit:.3f}  T={T_fit:.1f}  "
              f"eta_f={ef:.3f}  consistent={consistent}", flush=True)

        summary.append(dict(
            region=region, pos=pos, chi2=chi2_best,
            log_n_H2_fit=log_n_fit, T_k_fit=T_fit, log_N_dv_fit=ndv_fit,
            log_n_H2_table1a=a1a['log_nH2'] if a1a else np.nan,
            T_k_table1a=a1a['T_k'] if a1a else np.nan,
            delta_log_n=(log_n_fit - a1a['log_nH2']) if a1a else np.nan,
            T_B_obs=T_B_obs, T_B_theor=A_MAIN_fit, eta_f=ef,
            K_predicted=K_pred, K_required=K_req, physically_consistent=consistent,
            R21_11_predicted=R21_11_pred,
            R21_11_theor=t321['ratio_theor'] if t321 else np.nan,
            R21_11_observed=t321['ratio_obs'] if t321 else np.nan,
            any_maser_at_fit=bool(df['any_maser'].values[order][best_idx]),
            pinned_log_n_lo=np.isclose(log_n_fit, log_n_ax.min(), atol=0.05),
            pinned_log_n_hi=np.isclose(log_n_fit, log_n_ax.max(), atol=0.05),
            pinned_T_lo=np.isclose(T_fit, T_ax.min(), atol=0.5),
            pinned_T_hi=np.isclose(T_fit, T_ax.max(), atol=0.5),
        ))

    dfsum = pd.DataFrame(summary)
    csv_path = os.path.join(OUTDIR, 'summary.csv')
    dfsum.to_csv(csv_path, index=False)
    print(f"\nsaved {csv_path}  ({len(dfsum)} positions)")
    print(f"median chi2 = {dfsum.chi2.median():.3f}   median |delta log n| = {dfsum.delta_log_n.abs().median():.3f}")
    print(f"physically_consistent: {dfsum.physically_consistent.mean():.1%}")
    print(f"any_maser at best fit: {dfsum.any_maser_at_fit.mean():.1%}")


if __name__ == '__main__':
    main()
