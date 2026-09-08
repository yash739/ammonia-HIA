"""
Post-hoc fix for rows already computed by the currently-running (2,1)/(3,2)
batches (stutzki_21_check / w33_ladder_screen), which forked from a parent
process holding the pre-fix ratio_screen module in memory -- editing
ratio_screen.py on disk doesn't reach an already-running multiprocessing.Pool,
since fork() copies already-imported modules rather than re-reading them.

Rather than kill those jobs and lose hours of NLTE-solve progress, this
re-reads each row's already-saved (2,1)/(3,2) spectrum FITS files (written
unconditionally by nh3_NLTE_sphere._image_and_save_line and never deleted for
these two lines -- only (1,1)/(2,2) get cleaned up, by analyse_spectra) and
recomputes T_mb/noise/detected with the fixed _native_peak_tmb. No Magritte
recompute needed.
"""

import os
import csv
import numpy as np
from astropy.io import fits

import params as p
import ratio_screen as rs


def _read_spectrum_fits(path):
    with fits.open(path) as h:
        Is = h[0].data
        hdr = h[0].header
        velos = hdr['CRVAL1'] + np.arange(hdr['NAXIS1']) * hdr['CDELT1']
        freq = hdr['RESTFREQ']
    return velos, Is, freq


def reprocess_stutzki_21_check(csv_path=None, out_csv=None):
    csv_path = csv_path or os.path.join(rs.STUTZKI_ODIR, "stutzki_21_check.csv")
    out_csv = out_csv or csv_path.replace(".csv", "_fixed.csv")
    fits_dir = os.path.join(rs.STUTZKI_ODIR, "fits")

    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))

    fixed_rows = []
    for row in rows:
        if row['status'] != 'SUCCESS':
            fixed_rows.append(row)
            continue
        tag = f"1e-08_{float(row['n_H2']):.2e}_{float(row['N_NH3']) / (2.0 * float(row['n_H2']) * 1e-8):.2e}_180.16836131796748_{row['T_k']}"
        spectrum_path = os.path.join(fits_dir, f"NLTE_nh3_spectrum_21_{tag}.fits")
        if not os.path.exists(spectrum_path):
            print(f"MISSING: {spectrum_path} -- leaving row as-is")
            fixed_rows.append(row)
            continue
        velos, Is, freq = _read_spectrum_fits(spectrum_path)
        peak, noise, detected = rs._native_peak_tmb(velos, Is, freq)
        T11 = float(row['T_mb_11_native'])
        row = dict(row)
        row['T_mb_21_native'] = peak
        row['noise_21'] = noise
        row['detected_21'] = detected
        row['ratio_model'] = peak / T11 if T11 != 0 else np.nan
        fixed_rows.append(row)
        flag = "" if detected else "  [BELOW NOISE FLOOR]"
        print(f"{row['field']} {row['position']}: ratio_model={row['ratio_model']:.4f} "
              f"vs theor={row['ratio_theor_stutzki']} obs={row['ratio_obs_stutzki']}{flag}")

    fieldnames = list(rows[0].keys())
    if 'noise_21' not in fieldnames:
        idx = fieldnames.index('T_mb_21_native') + 1
        fieldnames[idx:idx] = ['noise_21', 'detected_21']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in fixed_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Wrote corrected CSV to {out_csv}")
    return out_csv


def reprocess_w33_ladder(source_name, csv_path=None, out_csv=None):
    csv_path = csv_path or os.path.join(rs.W33_ODIR, "results", f"{source_name}_ladder_screen.csv")
    out_csv = out_csv or csv_path.replace(".csv", "_fixed.csv")
    fits_dir = os.path.join(rs.W33_ODIR, "fits")

    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))

    obs_vec = rs.observed_ladder_ratio(source_name)
    freq_by_label = {'21': p.FREQ_HZ['2,1'], '32': p.FREQ_HZ['3,2']}

    fixed_rows = []
    for row in rows:
        if row['status'] != 'SUCCESS':
            fixed_rows.append(row)
            continue
        tag = f"{float(row['XNH3']):.0e}_{float(row['numberdensity']):.2e}_{float(row['radius_req']):.2e}_{row['vturb']}_{row['T_cloud']}"
        row = dict(row)
        for lbl in ('21', '32'):
            spectrum_path = os.path.join(fits_dir, f"NLTE_nh3_spectrum_{lbl}_{tag}.fits")
            if not os.path.exists(spectrum_path):
                print(f"MISSING: {spectrum_path} -- leaving {lbl} as-is")
                continue
            velos, Is, freq = _read_spectrum_fits(spectrum_path)
            peak, noise, detected = rs._native_peak_tmb(velos, Is, freq_by_label[lbl])
            row[f'T_mb_{lbl}'] = peak
            row[f'noise_{lbl}'] = noise
            row[f'detected_{lbl}'] = detected

        T11 = float(row['T_mb_11'])
        model_vec = []
        for lbl in rs.LADDER_LINES:
            key = lbl.replace(',', '')
            if key in ('21', '32'):
                # Only these two were re-derived above (below-noise-floor
                # issue) -- absence of a value means genuinely undetected.
                detected_val = row.get(f'detected_{key}')
                is_detected = detected_val is True or str(detected_val) == 'True'
            else:
                # 22/44 are strong lines untouched by the noise-floor bug;
                # the original batch's CSV predates the detected_* columns
                # entirely, so their presence in the row (not a boolean
                # flag) is the only signal available -- treat as detected.
                is_detected = f'T_mb_{key}' in row and row[f'T_mb_{key}'] not in (None, '')
            model_vec.append(float(row[f'T_mb_{key}']) if is_detected else np.nan)
        model_vec = np.array(model_vec) / T11
        row['cosine_similarity'] = rs.cosine_similarity(model_vec, obs_vec)
        fixed_rows.append(row)
        flag = "" if row.get('detected_21') in (True, 'True') else "  [21 BELOW NOISE FLOOR]"
        print(f"n_H2={float(row['numberdensity']):.2e} X={float(row['XNH3']):.0e}: "
              f"cos_sim={row['cosine_similarity']:.4f} T_mb21={float(row['T_mb_21']):.6f}{flag}")

    fieldnames = list(rows[0].keys())
    for key in ('noise_21', 'detected_21', 'noise_32', 'detected_32'):
        if key not in fieldnames:
            fieldnames.append(key)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in fixed_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Wrote corrected CSV to {out_csv}")
    return out_csv


if __name__ == "__main__":
    reprocess_stutzki_21_check()
    for src in rs.W33_21_DETECTED_SOURCES:
        path = os.path.join(rs.W33_ODIR, "results", f"{src}_ladder_screen.csv")
        if os.path.exists(path):
            reprocess_w33_ladder(src)
