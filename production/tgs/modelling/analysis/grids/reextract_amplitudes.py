"""Re-extract the (2,2) and (2,1) amplitudes of a LUT from its stored spectra.

Why: until 2026-10-06 the grid builder took A_MAIN_22 / A_MAIN_21 from a
5-channel box-smoothed peak (`native_peak_tmb`), which lowered them by 25-43%
because the lines are only 2-4 channels wide (important_notes sec. 23). The
stored per-model spectra (spectra/<key>.npz, the (1,1), (2,2), (2,1) lines on
the 0.152 km/s channel grid) let every row be re-extracted with the
unbiased `line_peak_tmb` WITHOUT any Magritte compute.

Rewrites A_MAIN_22, R_22_MAIN, A_MAIN_21, R_21_11 (and adds the `extractor`
column) in results/lut_dv<dv>.csv, after saving the original as
results/lut_dv<dv>.pre_v2_extraction.csv. Per-row diagnostics (fit method,
fitted FWHM, Gaussian/spline agreement) go to results/extraction_v2_diagnostics.csv.
Idempotent: always starts from the saved original table.

Acceptance rules (accept_22 / accept_21): rows whose line is ripple-dominated
(undetected (2,1); the cold, low-density, high-column corner) keep the old
value, which is an upper limit there; `extractor` records 'v2_gaussfit' (both
new), 'v2_gaussfit_22only', or 'v1' (both old).

Usage: python3 -m analysis.grids.reextract_amplitudes --odir <lut dir> [--dv 0.30]
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd

from nh3hia.lut.build import HEADER
from nh3hia.spectrum_utils import line_peak_tmb, EXTRACTOR_VERSION
from nh3hia.w33 import params as p


def spectrum_file(spec_dir, row):
    key = f"{row.log_n_H2:.4f}_{row.T_cloud:.4f}_{row.log_N_dv:.4f}"
    f = os.path.join(spec_dir, key + '.npz')
    return f if os.path.exists(f) else None


def accept_22(info, new, old):
    """A (2,2) re-extraction is trusted only for a real line: Gaussian fit
    accepted, Gaussian and model-free spline peaks agree, fitted width
    plausible, and the new peak is not below the smoothed one (smoothing can
    only lower a real line's peak). Anything else is ripple-dominated (cold,
    low-density, high-column corner; unconverged rows)."""
    return (info['method'] == 'gauss' and 0.85 <= info['gauss_over_spline'] <= 1.25
            and 0.15 <= info['fwhm'] <= 2.0 and old > 0 and 0.95 <= new / old <= 4.0)


def accept_21(info, new, old, detected):
    """(2,1) is trusted only where the line is above the numerical floor
    (`detected_21`, from the build) and the fit is a real Gaussian one; the
    undetected lines (low density) keep the old smoothed value, which is a
    ripple-floor upper limit and was only ever used as such."""
    return bool(detected) and info['method'] == 'gauss' and 0.85 <= info['gauss_over_spline'] <= 1.25 \
        and old > 0 and 0.95 <= new / old <= 4.0


def reextract(odir, dv=0.30):
    res = os.path.join(odir, 'results')
    csv_path = os.path.join(res, f'lut_dv{dv:.2f}.csv')
    spec_dir = os.path.join(odir, 'spectra')
    backup = os.path.join(res, f'lut_dv{dv:.2f}.pre_v2_extraction.csv')
    if not os.path.exists(backup):
        shutil.copy(csv_path, backup)
    d = pd.read_csv(backup)      # always start from the original v1 table: idempotent
    d['extractor'] = 'v1'
    diag = []
    n_missing = 0
    for i, r in d.iterrows():
        if r.Status != 'SUCCESS':
            continue
        f = spectrum_file(spec_dir, r)
        if f is None:
            n_missing += 1
            continue
        s = np.load(f)
        v = s['velos'].astype(float)
        out = {k: line_peak_tmb(v, s[k].astype(float), freq, return_info=True)
               for k, freq in (('22', p.FREQ_HZ['2,2']), ('21', p.FREQ_HZ['2,1']))}
        (a22, i22), (a21, i21) = out['22'], out['21']
        old22, old21 = d.at[i, 'A_MAIN_22'], d.at[i, 'A_MAIN_21']
        ok22 = accept_22(i22, a22, old22)
        ok21 = accept_21(i21, a21, old21, r.detected_21)
        am = d.at[i, 'A_MAIN']
        if ok22:
            d.at[i, 'A_MAIN_22'] = a22
            d.at[i, 'R_22_MAIN'] = a22 / am if am else np.nan
        if ok21:
            d.at[i, 'A_MAIN_21'] = a21
            d.at[i, 'R_21_11'] = a21 / am if am else np.nan
        d.at[i, 'extractor'] = EXTRACTOR_VERSION if (ok22 and ok21) else (EXTRACTOR_VERSION + '_22only' if ok22 else 'v1')
        diag.append(dict(log_n_H2=r.log_n_H2, T_cloud=r.T_cloud, log_N_dv=r.log_N_dv,
                         method22=i22['method'], fwhm22=i22['fwhm'], gauss_over_spline22=i22['gauss_over_spline'],
                         method21=i21['method'], fwhm21=i21['fwhm'], gauss_over_spline21=i21['gauss_over_spline'],
                         new_over_old22=a22 / old22 if old22 else np.nan,
                         new_over_old21=a21 / old21 if old21 else np.nan,
                         accepted22=ok22, accepted21=ok21))
    assert set(HEADER) <= set(d.columns), set(HEADER) - set(d.columns)
    d = d[HEADER]        # same column order as the builder appends in (resumable builds stay aligned)
    tmp = csv_path + '.tmp'
    d.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)
    dg = pd.DataFrame(diag)
    dg.to_csv(os.path.join(res, 'extraction_v2_diagnostics.csv'), index=False)
    print(f"{odir}: {len(dg)} rows ({n_missing} without spectra) | (2,2) re-extracted {dg.accepted22.sum()}, "
          f"kept old {(~dg.accepted22).sum()} | (2,1) re-extracted {dg.accepted21.sum()}, kept old {(~dg.accepted21).sum()} "
          f"| extractor: {d.extractor.value_counts().to_dict()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--odir', required=True, nargs='+')
    ap.add_argument('--dv', type=float, default=0.30)
    a = ap.parse_args()
    for o in a.odir:
        reextract(o, a.dv)


if __name__ == '__main__':
    main()
