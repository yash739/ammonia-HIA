"""Compare the escape-probability model against the pre-fix Magritte 3D RT
run (output_test_1e-6_parallel_12rays_v2), with the outer-satellite label
swap corrected.

Rigorous derivation of the correction (see conversation, not repeated in
full here): the pre-fix pipeline used the WRONG-SIGN velocity convention
(v = c(nu-nu_rest)/nu_rest, the negative of the standard radio convention),
so its output spectrum was ordered by DESCENDING true velocity -- the mirror
image of the physical spectrum. Its label assignment was a FIXED array
position -> name mapping (amps11[0]->'A_10', [1]->'A_21', [2]->'A_MAIN',
[3]->'A_12', [4]->'A_01'), independent of which physical component actually
landed there. Composing the mirroring with that fixed mapping:
  position 0 (mirrored) = true rightmost = true A_01, but labelled 'A_10'  -> SWAPPED
  position 1 (mirrored) = true 2nd-from-right = true A_21, labelled 'A_21' -> correct
  position 2 = true A_MAIN, labelled 'A_MAIN'                             -> correct
  position 3 (mirrored) = true 2nd-from-left = true A_12, labelled 'A_12' -> correct
  position 4 (mirrored) = true leftmost = true A_10, but labelled 'A_01'  -> SWAPPED
i.e. ONLY the outer pair (A_01/A_10, hence R_01_MAIN/R_10_MAIN) is swapped;
the inner pair happens to survive correctly despite the sign bug, because
the fixed-position labelling for the inner components was already in
"reversed" order, cancelling the mirroring. This matches the fix commit's
own description exactly ("The outer NH3 (1,1) satellites were mislabelled").
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import nh3_escape_model as m
from make_figures import eq11_ratio, OUTER_LTE_RATIO, INNER_LTE_RATIO

MAGRITTE_CSV = '/home/yasho379/magritte_rebuilt/scratch/output/output_test_1e-6_parallel_12rays_v2/results/NLTE_nh3_1e-6_parallel_12rays_v2.csv'
OUT_DIR = '/home/yasho379/magritte_rebuilt/scratch/output/output_stutzki85'


def load_corrected_magritte(T_k=36.0):
    df = pd.read_csv(MAGRITTE_CSV)
    df = df[(df['Status'] == 'SUCCESS') & np.isclose(df['T_cloud'], T_k)].copy()
    # THE FIX: swap outer pair only.
    df['R_01_MAIN_fixed'] = df['R_10_MAIN']
    df['R_10_MAIN_fixed'] = df['R_01_MAIN']
    df['R_12_MAIN_fixed'] = df['R_12_MAIN']  # unchanged
    df['R_21_MAIN_fixed'] = df['R_21_MAIN']  # unchanged
    df['log_n_H2'] = np.log10(df['numberdensity'])
    df = df.rename(columns={'Main Hyperfine Optical Depth': 'tau_main'})
    return df


def compute_escape_model_grid(T_k=36.0, log_n_list=(3.5, 4.5, 5.5, 6.5)):
    model = m.NH3Model()
    Cmat = model.collision_matrix(T_k)
    rows = []
    for log_n in log_n_list:
        n_H2 = 10.0 ** log_n
        x0 = None
        for log_N in np.arange(14.2, 17.6, 0.05):  # extend to match Magritte's N range
            out = m.run_one(model, Cmat, T_k=T_k, n_H2=n_H2, log_N_dv=log_N, x0=x0)
            x0 = out['x']
            rows.append(dict(log_n_H2=log_n, log_N_dv=log_N,
                              **{k: v for k, v in out.items() if k != 'x'}))
    return pd.DataFrame(rows)


def plot_comparison(df_magritte, df_escape, T_k, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    panels = [('R_01', 'R_01_MAIN_fixed', OUTER_LTE_RATIO, r'$F_1=0\to1$'),
              ('R_10', 'R_10_MAIN_fixed', OUTER_LTE_RATIO, r'$F_1=1\to0$'),
              ('R_12', 'R_12_MAIN_fixed', INNER_LTE_RATIO, r'$F_1=1\to2$'),
              ('R_21', 'R_21_MAIN_fixed', INNER_LTE_RATIO, r'$F_1=2\to1$')]
    log_n_list = sorted(df_escape['log_n_H2'].unique())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(log_n_list)))

    for ax, (esc_col, mag_col, lte_r, title) in zip(axes.flat, panels):
        for log_n, color in zip(log_n_list, colors):
            sub_e = df_escape[np.isclose(df_escape['log_n_H2'], log_n)].sort_values('tau_main')
            sub_e = sub_e[sub_e[esc_col].apply(np.isfinite)]
            ax.plot(sub_e['tau_main'], sub_e[esc_col], color=color, lw=1.6,
                     label=f'escape model, $n=10^{{{log_n:.1f}}}$')

        mag_n_list = sorted(df_magritte['log_n_H2'].unique())
        mag_colors = plt.cm.autumn(np.linspace(0.1, 0.9, len(mag_n_list)))
        for log_n, color in zip(mag_n_list, mag_colors):
            sub_m = df_magritte[np.isclose(df_magritte['log_n_H2'], log_n)].sort_values('tau_main')
            ax.scatter(sub_m['tau_main'], sub_m[mag_col], color=color, s=10, marker='x',
                        label=f'Magritte (corrected), $n=10^{{{log_n:.2f}}}$')

        tau_ref = np.logspace(-2, 1, 200)
        ax.plot(tau_ref, eq11_ratio(tau_ref, lte_r), color='k', lw=1.5, ls='--', label='Eq.(11) no anomaly')
        ax.set_xscale('log')
        ax.set_xlabel(r'$\tau(\Delta F_1=0)$')
        ax.set_ylabel(f'{title} / main')
        ax.set_title(title)
        ax.set_xlim(0.01, 20)
        ax.set_ylim(0, 1.3)
        ax.grid(alpha=0.3)

    axes[0, 0].legend(fontsize=6, ncol=1, loc='upper left')
    plt.suptitle(f'Escape-probability model (Loreau rates) vs corrected Magritte 3D RT, T={T_k:.0f}K\n'
                  '(Magritte outer pair R_01/R_10 swap-corrected per the labelling fix; inner pair unaffected)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f'saved {out_path}')


if __name__ == '__main__':
    T_k = 36.0
    df_mag = load_corrected_magritte(T_k)
    print(f'Magritte points at T={T_k}K: {len(df_mag)}, n range 10^{df_mag["log_n_H2"].min():.2f}-10^{df_mag["log_n_H2"].max():.2f}, '
          f'tau_main range {df_mag["tau_main"].min():.3f}-{df_mag["tau_main"].max():.3f}')
    df_esc = compute_escape_model_grid(T_k)
    df_esc.to_csv(f'{OUT_DIR}/compare_escape_grid_T{T_k:.0f}.csv', index=False)
    plot_comparison(df_mag, df_esc, T_k, f'{OUT_DIR}/compare_magritte_T{T_k:.0f}.png')
