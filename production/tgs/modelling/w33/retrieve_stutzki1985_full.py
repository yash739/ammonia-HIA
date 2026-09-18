"""Full chi^2 retrieval diagnostic against Stutzki's 1984/1985 observations,
using the interpolated gold LUT (per feedback_lut_retrieval_use_interpolator.md
-- never nearest-grid-point scoring).

For each of the 23 cross-checked positions (stutzki_params.RETRIEVAL_TEST_POSITIONS,
the properly key-aligned intersection of Stutzki et al. 1984's Table 2 ratios and
Stutzki & Winnewisser 1985's Table 1a fitted parameters), produces:

  1. The observed (1,1) spectrum reconstructed from his reported T_B_11 and the
     four satellite/main ratios, at his own observed (blended) linewidth.
  2. The best-fit model spectrum from the interpolated LUT, at the model's own
     intrinsic clump linewidth (0.3 km/s) -- NOT rescaled to match (1) in width
     or amplitude, so any mismatch in shape or brightness is directly visible.
  2b. The best-fit (log_n_H2, T_k, log_N_dv), its chi^2, beam filling factor
      eta_f = T_B_obs/T_B_theor, predicted clump count K vs the K required to
      explain the observed linewidth, and the physically_consistent() verdict.
  3. Chi^2 landscape through the best-fit point in all three orthogonal 2D
     slices (T x log_n, T x log_Ndv, log_n x log_Ndv), to show branch structure
     and whether the minimum is interior or pinned at a grid edge.
  4. Predicted (2,1)/(1,1) at the best-fit point (a genuine held-out quantity --
     never enters the fit), compared against Stutzki's own theoretical/observed
     values where Table 3 has them (S106 (0,0), OMC S3, OMC S4 only).

One PNG per position plus a summary CSV, both in
output_lut_gold/results/stutzki1985_retrieval/.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import stutzki_params as sp
import nh3_hyperfine as hf
from lut_interpolator import LutInterpolator, chi2_retrieve, RATIO_KEYS, RATIO_21
from stutzki_physics import eta_f as calc_eta_f, clump_count_K, clump_count_required, physically_consistent

DV_CLUMP_KMS = 0.3
OUTDIR = "/home/yasho379/magritte_rebuilt/production/output_lut_gold/results/stutzki1985_retrieval/"
QGRID = dict(n_log_n=200, n_T=73, n_log_Ndv=90)
OFFSETS = hf.NH3_11_OFFSETS_KMS  # order: A_10, A_12, A_MAIN, A_21, A_01
COMP_ORDER = ('A_10', 'A_12', 'A_MAIN', 'A_21', 'A_01')


def gaussian_spectrum(v, amps, fwhm_kms):
    sigma = fwhm_kms / 2.354820045
    y = np.zeros_like(v)
    for a, o in zip(amps, OFFSETS):
        y += a * np.exp(-0.5 * ((v - o) / sigma) ** 2)
    return y


def find_local_minima(Z, max_chi2=50.0):
    """2D local minima (4-neighbour) below max_chi2, sorted by chi2. Excludes
    NaN/inf. Returns list of (i, j, value)."""
    out = []
    ni, nj = Z.shape
    for i in range(ni):
        for j in range(nj):
            c = Z[i, j]
            if not np.isfinite(c) or c > max_chi2:
                continue
            nbrs = [Z[i+di, j+dj] for di, dj in ((1,0),(-1,0),(0,1),(0,-1))
                    if 0 <= i+di < ni and 0 <= j+dj < nj]
            nbrs = [n for n in nbrs if np.isfinite(n)]
            if nbrs and c <= min(nbrs):
                out.append((i, j, c))
    out.sort(key=lambda t: t[2])
    # de-duplicate minima that are adjacent to an already-kept, lower one
    kept = []
    for i, j, c in out:
        if all(abs(i - ki) > 2 or abs(j - kj) > 2 for ki, kj, _ in kept):
            kept.append((i, j, c))
    return kept


def chi2_slice_plot(ax, X, Y, Z3d, axis_i, xlabel, ylabel, best_xy, chi2_best,
                    table1a_xy=None, title=''):
    """Profile (minimum-projected) chi^2 over the THIRD axis, not a fixed slice
    through the global best-fit point -- a fixed slice can miss a genuine
    secondary branch entirely if that branch's own optimum sits at a different
    value of the axis being held fixed (verified: OMC S4's low-density local
    minimum sits at log_Ndv=14.73, while the global best fit's own log_Ndv is
    15.42 -- a slice through the latter does not pass through the former).
    Matches the profile-landscape approach already used for the legacy-grid
    retrieval (report's chi2_landscapes section), extended to log color
    scaling so basins spanning orders of magnitude in chi^2 both remain visible.
    """
    Zprof = np.nanmin(np.where(np.isfinite(Z3d), Z3d, np.nan), axis=axis_i)
    from matplotlib.colors import LogNorm
    vmin = max(chi2_best, 1e-2)
    vmax = max(np.nanpercentile(Zprof, 95), vmin * 10)
    pcm = ax.pcolormesh(X, Y, Zprof.T, shading='auto', cmap='viridis_r',
                        norm=LogNorm(vmin=vmin, vmax=vmax))
    levels = sorted(set([chi2_best + 1, chi2_best + 4, chi2_best + 9,
                         chi2_best * 5, chi2_best * 10, chi2_best * 25]))
    levels = [l for l in levels if vmin < l < vmax]
    if levels:
        ax.contour(X, Y, Zprof.T, levels=levels, colors='white', linewidths=0.5, alpha=0.6)
    ax.scatter(*best_xy, marker='*', s=200, color='red', edgecolor='k', linewidth=0.7, zorder=6,
              label=f'global best ($\chi^2$={chi2_best:.2f})')
    minima = find_local_minima(Zprof, max_chi2=max(50.0, chi2_best * 30))
    for k, (i, j, c) in enumerate(minima[1:5]):  # skip index 0 = the global best itself
        ax.scatter(X[i], Y[j], marker='X', s=110, color='orange', edgecolor='k', linewidth=0.6,
                  zorder=5, label=('other local minima' if k == 0 else None))
        ax.annotate(f'{c:.1f}', (X[i], Y[j]), textcoords='offset points', xytext=(5, 5),
                   fontsize=7, color='orange')
    if table1a_xy is not None and np.isfinite(table1a_xy[0]) and np.isfinite(table1a_xy[1]):
        ax.scatter(*table1a_xy, marker='D', s=60, color='cyan', edgecolor='k', linewidth=0.6, zorder=5,
                   label='Stutzki Table 1a')
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8)
    plt.colorbar(pcm, ax=ax, shrink=0.85, label=r'profile $\chi^2_{min}$ (log scale)')
    ax.legend(fontsize=6, loc='best')


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("building interpolator + dense query grid...", flush=True)
    interp = LutInterpolator()
    pts = interp.query_grid(**QGRID)
    ratio_grid = interp.predict_ratios(pts)
    log_n_ax = np.linspace(interp.axis_lo[0], interp.axis_hi[0], QGRID['n_log_n'])
    T_ax = np.linspace(interp.axis_lo[1], interp.axis_hi[1], QGRID['n_T'])
    ndv_ax = np.linspace(interp.axis_lo[2], interp.axis_hi[2], QGRID['n_log_Ndv'])
    shape3d = (QGRID['n_log_n'], QGRID['n_T'], QGRID['n_log_Ndv'])
    print(f"  {len(interp.df)} LUT rows, {len(pts)} query points, axis ranges "
          f"log_n=[{interp.axis_lo[0]:.2f},{interp.axis_hi[0]:.2f}] "
          f"T=[{interp.axis_lo[1]:.1f},{interp.axis_hi[1]:.1f}] "
          f"log_Ndv=[{interp.axis_lo[2]:.3f},{interp.axis_hi[2]:.3f}]", flush=True)

    summary = []
    for region, pos in sp.RETRIEVAL_TEST_POSITIONS:
        entry = sp.TABLE_2_1984.get((region, pos))
        a1a = sp.TABLE_1A.get(region, {}).get(pos)
        if entry is None:
            continue
        tag = f"{region}_{pos}".replace(',', '_').replace(' ', '')
        print(f"=== {region} {pos} ===", flush=True)

        obs = {'R_01_MAIN': entry['R_01'], 'R_10_MAIN': entry['R_10'],
               'R_21_MAIN': entry['R_21'], 'R_12_MAIN': entry['R_12'],
               'R_22_MAIN': entry.get('R_22_MAIN', np.nan)}
        err = {'R_01_MAIN': entry.get('R_01_err', np.nan), 'R_10_MAIN': entry.get('R_10_err', np.nan),
               'R_21_MAIN': entry.get('R_21_err', np.nan), 'R_12_MAIN': entry.get('R_12_err', np.nan),
               'R_22_MAIN': entry.get('R_22_MAIN_err', np.nan)}
        obs_vec = np.array([obs[k] for k in RATIO_KEYS])
        sigma_vec = np.array([err[k] for k in RATIO_KEYS])

        best_idx, chi2_best, chi2_all = chi2_retrieve(interp, pts, ratio_grid, obs_vec, sigma_vec)
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

        # --- spectrum reconstructions ---
        v = np.linspace(-32, 32, 800)
        obs_amps = [T_B_obs * entry['R_10'], T_B_obs * entry['R_12'], T_B_obs,
                    T_B_obs * entry['R_21'], T_B_obs * entry['R_01']]
        obs_spec = gaussian_spectrum(v, obs_amps, dv_obs)
        model_amps = [A_MAIN_fit * model_ratios['R_10_MAIN'], A_MAIN_fit * model_ratios['R_12_MAIN'], A_MAIN_fit,
                      A_MAIN_fit * model_ratios['R_21_MAIN'], A_MAIN_fit * model_ratios['R_01_MAIN']]
        model_spec = gaussian_spectrum(v, model_amps, DV_CLUMP_KMS)

        # --- (2,1) comparison, where available ---
        t321 = sp.TABLE_3_21.get((region, pos))

        # --- chi2 3D landscape, sliced through the best-fit point ---
        chi2_3d = chi2_all.reshape(shape3d)
        i_n = int(np.argmin(np.abs(log_n_ax - log_n_fit)))
        i_T = int(np.argmin(np.abs(T_ax - T_fit)))
        i_d = int(np.argmin(np.abs(ndv_ax - ndv_fit)))

        # --- figure ---
        fig = plt.figure(figsize=(21, 10))
        gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.9)

        ax_spec = fig.add_subplot(gs[0, 0:3])
        ax_spec.plot(v, obs_spec, color='k', lw=1.6, label=f'observed (FWHM={dv_obs:.2f} km/s)')
        ax_spec.plot(v, model_spec, color='tab:red', lw=1.4, ls='--',
                    label=f'best-fit model (FWHM={DV_CLUMP_KMS} km/s)')
        ax_spec.set_xlabel('v [km/s]', fontsize=9)
        ax_spec.set_ylabel(r'$T_B$ [K]', fontsize=9)
        ax_spec.set_title(f'{region} {pos}: (1,1) spectrum', fontsize=10)
        ax_spec.legend(fontsize=7)
        ax_spec.grid(alpha=0.25)

        # Same two spectra, each divided by its OWN main-line peak (T_B_obs for
        # the observed reconstruction, A_MAIN_fit for the model) -- isolates the
        # ratio shapes the chi^2 actually fits from the absolute-brightness
        # mismatch eta_f already reports separately.
        obs_spec_norm = gaussian_spectrum(v, [x / T_B_obs for x in obs_amps], dv_obs)
        model_spec_norm = gaussian_spectrum(v, [x / A_MAIN_fit for x in model_amps], DV_CLUMP_KMS)
        ax_spec_n = fig.add_subplot(gs[0, 3:6])
        ax_spec_n.plot(v, obs_spec_norm, color='k', lw=1.6, label='observed / $T_{B,obs}$')
        ax_spec_n.plot(v, model_spec_norm, color='tab:red', lw=1.4, ls='--', label='model / $T_{B,theor}$')
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
        ]
        ax_txt.text(0.02, 0.95, "\n".join(lines), va='top', ha='left', fontsize=9.5,
                    transform=ax_txt.transAxes, family='monospace')

        ax21 = fig.add_subplot(gs[0, 9:12])
        labels = ['predicted']
        vals = [R21_11_pred]
        colors = ['tab:red']
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
            pinned_log_n_lo=np.isclose(log_n_fit, interp.axis_lo[0], atol=0.05),
            pinned_log_n_hi=np.isclose(log_n_fit, interp.axis_hi[0], atol=0.05),
            pinned_T_lo=np.isclose(T_fit, interp.axis_lo[1], atol=0.3),
            pinned_T_hi=np.isclose(T_fit, interp.axis_hi[1], atol=0.3),
            pinned_ndv_lo=np.isclose(ndv_fit, interp.axis_lo[2], atol=0.02),
            pinned_ndv_hi=np.isclose(ndv_fit, interp.axis_hi[2], atol=0.02),
        ))

    df = pd.DataFrame(summary)
    csv_path = os.path.join(OUTDIR, 'summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nsaved {csv_path}  ({len(df)} positions)")
    print(f"median chi2 = {df.chi2.median():.3f}   median |delta log n| = {df.delta_log_n.abs().median():.3f}")
    print(f"physically_consistent: {df.physically_consistent.mean():.1%}")
    pinned = df[['pinned_log_n_lo', 'pinned_log_n_hi', 'pinned_T_lo', 'pinned_T_hi',
                'pinned_ndv_lo', 'pinned_ndv_hi']].sum()
    print("pinned-at-edge counts:\n", pinned.to_string())


if __name__ == '__main__':
    main()
