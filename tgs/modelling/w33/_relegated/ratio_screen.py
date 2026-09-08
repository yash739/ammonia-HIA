"""
Ratio-based screening: rank candidates by matching a *ladder* of line ratios
(relative to (1,1)) against genuinely published numbers, via cosine
similarity -- instead of screen.py's existing absolute T_mb/tau/N matching.

Why this exists (session finding): neither paper actually publishes the
(1,1)-internal hyperfine satellite ratio vector [R_01,R_10,R_21,R_12] as
numbers. Stutzki's Table 1a gives fitted (T_k, n_H2, N_NH3), not ratios --
those only exist as image contours in his Fig. 4a-c, which this project does
not digitize (screen.py's docstring already flags ratios as dilution-
invariant but never actually scores on them). Tursun et al. (2022) don't
report per-satellite amplitudes at all -- their hfs fit reports one bulk
main-line optical depth per transition, nothing finer. So there was no
external ratio number for score_row (screen.py) to match against, and it
never tried.

What IS genuinely published on both ends is per-transition T_mb across the
FULL LADDER of lines our LAMDA file covers: (1,1), (2,2), (4,4), (2,1), (3,2).
That forms a natural ratio vector [T_mb(2,2), T_mb(4,4), T_mb(2,1),
T_mb(3,2)] / T_mb(1,1) -- multi-dimensional (cosine similarity is meaningful,
unlike on a lone scalar), and, because these 5 lines span only 22.83-24.14 GHz
(<6% in frequency), their Effelsberg beam FWHM (40"*23GHz/nu) differ by only
~6% -- so this ratio vector is dilution-invariant to good approximation
(~10% level from the eta(theta) nonlinearity, not exact) and can be compared
directly against *native* (undiluted) model spectra, no Stage-A-style disk
correction needed.

Two independent pieces:

1. stutzki_21_check() -- Stage 0 extension. Stutzki's own Table 3 publishes a
   genuinely observed AND theoretical (2,1)/(1,1) ratio for 3 positions
   (S106 '0,0', OMC 'S3', OMC 'S4') -- his own density diagnostic test,
   confirmed (via matching T_B(1,1) values) to be the same positions as
   Table 1a's high-density-solution fits. This was never actually run in
   Stage 0 (validate_stutzki.py never images the (2,1) line) -- it is here.

2. w33_ladder_screen() -- only for W33_A and W33_B, the two sources with a
   real (Table 4) (2,1) detection, not an upper limit (Main1/A1/B1 lack a
   (2,1) row in params.LINES entirely and are deliberately excluded, per the
   "sources which detected (2,1) in the first place" scope). Images the
   4-line ladder for a shortlist of existing catalogue candidates and ranks
   by cosine similarity between predicted and observed ratio vectors.
"""

import os
import sys
import csv
import multiprocessing
import numpy as np

import params as p
import stutzki_params as sp
import screen as sc

WDIR = "/home/yasho379/magritte_rebuilt/tgs/"
CATALOGUE_CSV = "/home/yasho379/magritte_rebuilt/output_w33_grid_extension/results/NLTE_nh3_w33_extension.csv"
STUTZKI_ODIR = "/home/yasho379/magritte_rebuilt/output_stutzki_21check/"
W33_ODIR = "/home/yasho379/magritte_rebuilt/output_w33_ratio_screen/"

# Relative to (1,1); all four are in p-nh3@loreau.dat.txt.
LADDER_LINES = ('2,2', '4,4', '2,1', '3,2')

# Only sources with a real (2,1) detection (not an upper limit, not absent)
# in Tursun et al.'s Table 4.
W33_21_DETECTED_SOURCES = ('W33_A', 'W33_B')


def _native_peak_tmb(velos, Is, freq_rest, smooth_channels=5, detection_sigma=3.0):
    """Convert a run_model-returned (velos [km/s], Is [raw intensity, NOT
    brightness temperature]) pair to Tmb and return (peak, noise, detected).

    Two issues found by hand on real output before this existed:

    1. Units: run_model's extra_spectra tuples are raw model.images[-1].I
       (~1e-13 to 1e-19 W/m^2/Hz/sr, straight from Magritte, see
       nh3_NLTE_sphere._center_beam_spectrum) -- NOT Kelvin. Only the cube
       path (return_cubes=True -> nh3_NLTE_sphere._intensity_cube_to_Tmb) is
       pre-converted; the plain (velos, Is) tuple is not. Fixed by reusing
       nh3_NLTE_analysis.intensity_to_Tmb (the same conversion A_MAIN uses),
       not re-deriving it.

    2. Baseline: for lines this faint (most of the W33 low-density
       candidates), nh3_NLTE_analysis.subtract_baseline's linear fit to the
       first/last 10% of channels breaks badly if a numerical ripple (a
       real, reproducible artifact -- confirmed by comparing raw spectra
       across candidates; a strong, well-thermalized line like Stutzki's
       OMC S4 is smooth, but these faint ones show small, roughly periodic
       dips scattered across the whole window, almost certainly a mesh/ray
       discretization effect at this level of sub-thermalization, not a
       real 18-hyperfine-component (2,1) feature) happens to land in that
       edge region -- the fit gets pulled off and np.max() afterward reads
       off noise, not signal (found: 2.4e-7 K "peak" against a 0.16 K-scale
       ripple floor in the very same spectrum). Fixed with a median baseline
       (robust to a handful of dips, since real signal -- if present at all
       -- occupies only a few of 500 channels) plus light smoothing (a real
       line, even a narrow one at these clump widths, spans >1 channel;
       single-channel noise doesn't survive a 5-channel box average).

    Returns (peak_Tmb, noise_estimate, detected) -- `detected` is False when
    peak < detection_sigma * noise, meaning the line is not just faint but
    unresolvable at this simulation's own numerical precision; callers
    should report "below noise floor", not the raw peak number, in that case.
    """
    modelling_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if modelling_dir not in sys.path:
        sys.path.insert(0, modelling_dir)
    from nh3_NLTE_analysis import intensity_to_Tmb
    velos = np.asarray(velos)
    Tmb = intensity_to_Tmb(1000 * velos, np.asarray(Is), freq_rest)
    baseline = np.median(Tmb)
    resid = Tmb - baseline
    if smooth_channels > 1:
        kernel = np.ones(smooth_channels) / smooth_channels
        resid = np.convolve(resid, kernel, mode='same')
    peak = float(np.max(resid))
    noise = float(np.std(resid))
    detected = peak >= detection_sigma * noise
    return peak, noise, detected


def cosine_similarity(a, b):
    """NaN-safe cosine similarity; NaN in either vector at a given index drops
    that component from both (so a missing catalogue line doesn't zero it out
    and bias the comparison)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    if not mask.any():
        return np.nan
    a, b = a[mask], b[mask]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def observed_ladder_ratio(source_name):
    """[T_mb(2,2), T_mb(4,4), T_mb(2,1), T_mb(3,2)] / T_mb(1,1) from Table 4
    (params.LINES). A line absent from the source's entry (not published /
    not detected) is NaN, not zero -- e.g. Main1 has no (2,1) or (3,2) row at
    all, so those pull out of cosine_similarity rather than counting as a
    non-detection floor."""
    lines = p.LINES[source_name]
    T11 = abs(lines['1,1']['T_mb'])
    vec = []
    for label in LADDER_LINES:
        entry = lines.get(label)
        vec.append(entry['T_mb'] / T11 if entry is not None else np.nan)
    return np.array(vec)


# --------------------------------------------------------------------------- #
# 1. Stutzki (2,1) check
# --------------------------------------------------------------------------- #

def _stutzki_21_worker(task):
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    modelling_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if modelling_dir not in sys.path:
        sys.path.insert(0, modelling_dir)
    from nh3_NLTE_sphere import run_model
    from nh3_NLTE_analysis import analyse_spectra

    field, position, table3_entry, fit_entry, X, max_NLTE = task
    try:
        T_k = fit_entry['T_k']
        n_H2 = sp.n_H2_cm3(fit_entry)
        N_NH3 = sp.N_NH3_cm2(fit_entry)
        vturb = p.fwhm_kms_to_vturb_ms(sp.CLUMP_DV_KMS)
        radius_sphere = N_NH3 / (2.0 * n_H2 * X)  # same identity as validate_stutzki.radius_for_target_N

        os.makedirs(os.path.join(STUTZKI_ODIR, "fits"), exist_ok=True)
        os.makedirs(os.path.join(STUTZKI_ODIR, "images"), exist_ok=True)

        conv = run_model(wdir=WDIR, odir=STUTZKI_ODIR, XNH3=X, numberdensity=n_H2, vturb=vturb,
                          T_cloud=T_k, max_NLTE=max_NLTE, radius_sphere=radius_sphere,
                          image_lines={'21': p.FREQ_HZ['2,1']})
        tau_main = conv[2]
        extra_spectra = conv[3]
        velos21, Is21 = extra_spectra['21']
        T_mb_21_native, noise_21, detected_21 = _native_peak_tmb(velos21, Is21, p.FREQ_HZ['2,1'])

        hfs = analyse_spectra(odir=STUTZKI_ODIR, XNH3=X, numberdensity=n_H2, vturb=vturb,
                               T_cloud=T_k, radius_sphere=radius_sphere,
                               max_NLTE=max_NLTE, save_plots=False)
        T_mb_11_native = hfs['A_MAIN']
        ratio_model = T_mb_21_native / T_mb_11_native if T_mb_11_native != 0 else np.nan

        return dict(status="SUCCESS", field=field, position=position, X=X,
                    T_k=T_k, n_H2=n_H2, N_NH3=N_NH3, tau_main=tau_main,
                    T_mb_21_native=T_mb_21_native, noise_21=noise_21, detected_21=detected_21,
                    T_mb_11_native=T_mb_11_native,
                    ratio_model=ratio_model,
                    ratio_theor_stutzki=table3_entry['ratio_theor'],
                    ratio_obs_stutzki=table3_entry['ratio_obs'],
                    T_B_11_table3=table3_entry['T_B_11'], T_B_21_table3=table3_entry['T_B_21'])
    except Exception as e:
        return dict(status="FAILED", field=field, position=position, X=X, error=str(e))


def stutzki_21_check(X=1e-8, max_NLTE=300, processes=3, out_csv=None):
    """Image (2,1) at the 3 Table-3 benchmark positions' own Table 1a fitted
    parameters, and compare the model's native (2,1)/(1,1) ratio against
    Stutzki's own theoretical AND observed values. One X per position by
    default (X doesn't affect the ratio much since it only sets r via
    N=2rnX at fixed n,N -- the (T_k,n_H2) pair is what actually sets the
    ratio) to keep this cheap; pass a list to X to sweep more.
    """
    X_values = X if isinstance(X, (list, tuple)) else [X]
    tasks = [(field, position, entry, fit_entry, x, max_NLTE)
              for field, position, entry, fit_entry in sp.iter_table_3_21()
              for x in X_values]

    out_csv = out_csv or os.path.join(STUTZKI_ODIR, "stutzki_21_check.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    print(f"Stutzki (2,1) check: {len(tasks)} runs ({len(tasks)//len(X_values)} positions "
          f"x {len(X_values)} X values), max_NLTE={max_NLTE}, {processes} workers")

    fieldnames = ['status', 'field', 'position', 'X', 'T_k', 'n_H2', 'N_NH3', 'tau_main',
                  'T_mb_21_native', 'noise_21', 'detected_21', 'T_mb_11_native', 'ratio_model',
                  'ratio_theor_stutzki', 'ratio_obs_stutzki',
                  'T_B_11_table3', 'T_B_21_table3', 'error']

    n_done = 0
    with multiprocessing.Pool(processes=processes, maxtasksperchild=1) as pool:
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in pool.imap_unordered(_stutzki_21_worker, tasks):
                writer.writerow({k: result.get(k, "") for k in fieldnames})
                f.flush()
                n_done += 1
                tag = f"{result['field']} {result['position']} X={result['X']:.0e}"
                if result['status'] == 'SUCCESS':
                    flag = "" if result['detected_21'] else "  [BELOW NOISE FLOOR]"
                    print(f"[{n_done}/{len(tasks)}] OK   {tag}: ratio_model={result['ratio_model']:.4f} "
                          f"vs theor={result['ratio_theor_stutzki']:.3f} obs={result['ratio_obs_stutzki']:.3f}{flag}")
                else:
                    print(f"[{n_done}/{len(tasks)}] FAIL {tag}: {result['error']}")
    print(f"Wrote {n_done} rows to {out_csv}")
    return out_csv


# --------------------------------------------------------------------------- #
# 2. W33 ladder-ratio cosine-similarity screening
# --------------------------------------------------------------------------- #

def _ladder_worker(task):
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    modelling_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if modelling_dir not in sys.path:
        sys.path.insert(0, modelling_dir)
    from nh3_NLTE_sphere import run_model

    source_name, row, max_NLTE = task
    try:
        XNH3 = float(row['XNH3'])
        numberdensity = float(row['numberdensity'])
        vturb = float(row['vturb'])
        T_cloud = float(row['T_cloud'])
        radius_req = float(row['radius_req'])
        clump_dv = float(row['clump_dv'])

        image_lines = {lbl.replace(',', ''): p.FREQ_HZ[lbl] for lbl in LADDER_LINES}
        conv = run_model(wdir=WDIR, odir=W33_ODIR, XNH3=XNH3, numberdensity=numberdensity,
                          vturb=vturb, T_cloud=T_cloud, max_NLTE=max_NLTE,
                          radius_sphere=radius_req, image_lines=image_lines)
        tau_main, extra_spectra = conv[2], conv[3]

        freq_by_label = {'11': p.FREQ_HZ['1,1'], '22': p.FREQ_HZ['2,2'], '44': p.FREQ_HZ['4,4'],
                          '21': p.FREQ_HZ['2,1'], '32': p.FREQ_HZ['3,2']}
        native, noise, detected = {}, {}, {}
        for lbl in ('11', '22', '44', '21', '32'):
            velos, Is = extra_spectra[lbl]
            native[lbl], noise[lbl], detected[lbl] = _native_peak_tmb(velos, Is, freq_by_label[lbl])

        result = dict(status="SUCCESS", source=source_name, XNH3=XNH3, numberdensity=numberdensity,
                      vturb=vturb, T_cloud=T_cloud, radius_req=radius_req, clump_dv=clump_dv,
                      tau_main=tau_main)
        for lbl in ('11', '22', '44', '21', '32'):
            result[f'T_mb_{lbl}'] = native[lbl]
            result[f'noise_{lbl}'] = noise[lbl]
            result[f'detected_{lbl}'] = detected[lbl]
        return result
    except Exception as e:
        return dict(status="FAILED", source=source_name, error=str(e),
                    XNH3=row.get('XNH3'), numberdensity=row.get('numberdensity'))


def build_shortlist(source_name, csv_path=CATALOGUE_CSV, top_k=15):
    """Reuse screen.py's existing T_mb/tau/N-based ranking to bound the
    candidate pool -- imaging the full ladder for all ~200 grid points would
    re-run the NLTE solve from scratch for each (run_model has no
    resume-and-add-imaging path), which is not worth doing for candidates
    screen.py already ranks as poor."""
    return sc.screen_source(source_name, csv_path, top_k=top_k)


def w33_ladder_screen(source_name, top_k=15, out_csv=None, processes=3, max_NLTE=None):
    if source_name not in W33_21_DETECTED_SOURCES:
        raise ValueError(f"{source_name} has no real (2,1) detection in Table 4 "
                          f"(only {W33_21_DETECTED_SOURCES} do) -- ladder cosine "
                          f"comparison needs the (2,1) anchor.")

    # W33_A's grid ran at max_NLTE=300, all other sources (incl. W33_B) at 200
    # (see run_grid_w33.py) -- match whichever cap the shortlist candidate's
    # own row was generated with, since the CSV itself doesn't store this.
    if max_NLTE is None:
        max_NLTE = 300 if source_name == 'W33_A' else 200

    shortlist = build_shortlist(source_name, top_k=top_k)
    obs_vec = observed_ladder_ratio(source_name)
    print(f"{source_name}: observed ladder ratio [22,44,21,32]/(1,1) = "
          f"{np.round(obs_vec, 4).tolist()}")
    print(f"{source_name}: imaging ladder for top-{len(shortlist)} T_mb/tau/N shortlist "
          f"candidates ({processes} workers, max_NLTE={max_NLTE})")

    tasks = [(source_name, row, max_NLTE) for row in shortlist]
    out_csv = out_csv or os.path.join(W33_ODIR, "results", f"{source_name}_ladder_screen.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.join(W33_ODIR, "fits"), exist_ok=True)
    os.makedirs(os.path.join(W33_ODIR, "images"), exist_ok=True)

    fieldnames = ['status', 'source', 'XNH3', 'numberdensity', 'vturb', 'T_cloud', 'radius_req',
                  'clump_dv', 'tau_main',
                  'T_mb_11', 'T_mb_22', 'noise_22', 'detected_22',
                  'T_mb_44', 'noise_44', 'detected_44',
                  'T_mb_21', 'noise_21', 'detected_21',
                  'T_mb_32', 'noise_32', 'detected_32',
                  'cosine_similarity', 'error']

    results = []
    n_done = 0
    with multiprocessing.Pool(processes=processes, maxtasksperchild=1) as pool:
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in pool.imap_unordered(_ladder_worker, tasks):
                if result['status'] == 'SUCCESS':
                    # Undetected components (below the spectrum's own noise floor,
                    # see _native_peak_tmb) are excluded from the cosine comparison
                    # rather than trusted as a precise near-zero number.
                    model_vec = []
                    for lbl in LADDER_LINES:
                        key = lbl.replace(',', '')
                        model_vec.append(result[f'T_mb_{key}'] if result[f'detected_{key}'] else np.nan)
                    model_vec = np.array(model_vec) / result['T_mb_11']
                    result['cosine_similarity'] = cosine_similarity(model_vec, obs_vec)
                writer.writerow({k: result.get(k, "") for k in fieldnames})
                f.flush()
                n_done += 1
                results.append(result)
                tag = f"n_H2={float(result['numberdensity']):.2e}" if 'numberdensity' in result and result['numberdensity'] else "?"
                if result['status'] == 'SUCCESS':
                    flag = "" if result['detected_21'] else "  [21 BELOW NOISE FLOOR]"
                    print(f"[{n_done}/{len(tasks)}] OK   {tag}: cos_sim={result['cosine_similarity']:.4f} "
                          f"T_mb21={result['T_mb_21']:.4f} (obs {p.LINES[source_name]['2,1']['T_mb']}){flag}")
                else:
                    print(f"[{n_done}/{len(tasks)}] FAIL {tag}: {result['error']}")
    print(f"Wrote {n_done} rows to {out_csv}")
    return out_csv, results


if __name__ == "__main__":
    stutzki_21_check()
