"""Grid completeness map for the gold LUT: for each T_cloud plane, which
(log_n_H2, log_N_dv) cells of the intended 14x14x9 cross product are filled
(converged SUCCESS) vs still pending, plus per-temperature and overall summary
stats. Reuses the axis definitions from build_lut_gold.py directly so this
never drifts from the actual target grid.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_lut_gold import LOG_N_AXIS, T_AXIS, LOG_NDV_AXIS

DEFAULT_LUT = "/home/yasho379/magritte_rebuilt/production/output_lut_gold/results/lut_dv0.30.csv"
PC_CM = 3.0857e18


def load(path):
    df = pd.read_csv(path)
    df = df[df['Status'] == 'SUCCESS']
    df = df[df['convergence_ok']]
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--lut', default=DEFAULT_LUT)
    ap.add_argument('--out', default='/home/yasho379/magritte_rebuilt/production/output_lut_gold/results/grid_completeness.png')
    ap.add_argument('--mask-radius', action='store_true',
                     help='also mark cells excluded by the 0.5pc radius mask (see important_notes)')
    a = ap.parse_args()

    df_all = load(a.lut)
    df = df_all[df_all.radius_sphere <= 0.5 * PC_CM] if a.mask_radius else df_all
    masked_out = set(zip(df_all.log_n_H2.round(4), df_all.T_cloud.round(4), df_all.log_N_dv.round(4))) \
        - set(zip(df.log_n_H2.round(4), df.T_cloud.round(4), df.log_N_dv.round(4)))

    have = set(zip(df.log_n_H2.round(4), df.T_cloud.round(4), df.log_N_dv.round(4)))
    n_axis, T_list, ndv_axis = [round(x, 4) for x in LOG_N_AXIS], T_AXIS, [round(x, 4) for x in LOG_NDV_AXIS]
    total_target = len(n_axis) * len(T_list) * len(ndv_axis)

    n_cols = 7
    n_rows = int(np.ceil(len(T_list) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.6 * n_rows))
    axes = np.atleast_2d(axes)

    done_total = 0
    for i, T in enumerate(T_list):
        ax = axes[i // n_cols, i % n_cols]
        grid = np.zeros((len(ndv_axis), len(n_axis)))  # 0=missing, 1=done, 2=radius-masked
        for xi, n in enumerate(n_axis):
            for yi, ndv in enumerate(ndv_axis):
                key = (n, round(T, 4), ndv)
                if key in have:
                    grid[yi, xi] = 1
                elif key in masked_out:
                    grid[yi, xi] = 2
        n_done = int((grid == 1).sum())
        done_total += n_done
        cmap = matplotlib.colors.ListedColormap(['#d9d9d9', '#2ca02c', '#ff7f0e'])
        ax.imshow(grid, origin='lower', cmap=cmap, vmin=0, vmax=2, aspect='auto',
                  extent=(0, len(n_axis), 0, len(ndv_axis)))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"T={T:.0f}K\n{n_done}/{len(n_axis)*len(ndv_axis)}", fontsize=9)
        if i % n_cols == 0:
            ax.set_ylabel('log(N/dv)\n$\\rightarrow$', fontsize=8)
        if i // n_cols == n_rows - 1:
            ax.set_xlabel('log(n)$\\rightarrow$', fontsize=8)

    for j in range(len(T_list), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis('off')

    from matplotlib.patches import Patch
    handles = [Patch(color='#2ca02c', label='converged'), Patch(color='#d9d9d9', label='pending')]
    if a.mask_radius:
        handles.append(Patch(color='#ff7f0e', label='radius-masked (>0.5pc)'))
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9)
    pct = 100 * done_total / total_target
    fig.suptitle(f"Gold LUT grid completeness: {done_total}/{total_target} cells ({pct:.1f}%) "
                 f"-- {len(n_axis)} density x {len(T_list)} T x {len(ndv_axis)} log(N/dv)", fontsize=12)
    plt.tight_layout(rect=(0, 0.04, 1, 0.94))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    plt.savefig(a.out, dpi=150)
    print(f"saved {a.out}")
    print(f"total: {done_total}/{total_target} ({pct:.1f}%)  raw_converged={len(df_all)} "
          f"radius_masked_out={len(masked_out)}")


if __name__ == '__main__':
    main()
