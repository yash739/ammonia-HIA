"""Progress and ETA of the radial gold rebuild (output_lut_gold_radial/), with a cost-model projection.

The remaining cost is projected from the gold-grid wall times of the nodes not yet done, scaled by the
measured ratio (rebuild wall / gold wall) on the nodes already done, divided by the measured number of
concurrent processes; the elapsed-time rate is shown alongside. Deadline Fri 2026-10-09 09:00 by default.

Usage: python3 -m analysis.grids.rebuild_progress [--deadline '2026-10-09 09:00'] [--processes 3]
"""
import argparse
import datetime as dt
import os

import pandas as pd

from nh3hia import paths

ODIR = os.path.join(paths.PRODUCTION, 'output_lut_gold_radial')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deadline', default='2026-10-09 09:00')
    ap.add_argument('--processes', type=int, default=3)
    a = ap.parse_args()
    f = os.path.join(ODIR, 'results', 'lut_dv0.30.csv')
    d = pd.read_csv(f)
    g = pd.read_csv(paths.LUT_GOLD_CSV)
    key = ['log_n_H2', 'T_cloud', 'log_N_dv']
    for x in (d, g):
        for k in key:
            x[k] = x[k].round(4)
    m = d.merge(g[key + ['wall_s']], on=key, suffixes=('', '_gold'))
    done_wall = d.wall_s.sum()
    ratio = done_wall / m.wall_s_gold.sum()
    todo = g.merge(d[key], on=key, how='left', indicator=True)
    todo = todo[todo._merge == 'left_only']
    rem_core = todo.wall_s.sum() * ratio
    start = dt.datetime.fromtimestamp(os.path.getctime(os.path.join(ODIR, 'results')))
    now = dt.datetime.now()
    el = (now - start).total_seconds()
    eta_model = now + dt.timedelta(seconds=rem_core / a.processes)
    rate = len(d) / el
    eta_rate = now + dt.timedelta(seconds=(len(g) - len(d)) / rate) if rate else None
    dl = dt.datetime.fromisoformat(a.deadline)
    print(f"{len(d)}/{len(g)} done ({d.Status.eq('SUCCESS').sum()} ok, {(d.convergence_ok == True).sum()} converged); "
          f"elapsed {el / 3600:.1f} h; rebuild/gold wall ratio so far {ratio:.3f}")
    print(f"ETA (cost model): {eta_model:%a %H:%M}   ETA (current rate): {eta_rate:%a %H:%M}   deadline {dl:%a %H:%M}   "
          f"margin (model) {(dl - eta_model).total_seconds() / 3600:+.1f} h")


if __name__ == '__main__':
    main()
