"""
Stage A: cheap analytic screening of a Magritte sphere catalogue (as produced by
ModelGrid.py) against one W33 source's observed (1,1) line parameters.

For each catalogue row this applies the Effelsberg beam-dilution correction
(params.disk_beam_dilution) to the row's native (unconvolved, central-beam-pixel)
predictions -- main-hyperfine amplitude, peak column density -- and ranks rows by
how well the *diluted* predictions match the source's observed T_mb(1,1), tau(1,1),
and (as a secondary check) the paper's own para-NH3 column N(para) (Table 6).

Ratios (R_01_MAIN etc.) are dilution-invariant and are not rescaled here -- Stage 0
(validate_stutzki.py) is where those get compared against a real n(H2)-vs-ratio
grid; this module only handles the "how much does distance/beam size dilute this
row" half of the closure (Physical framework, step 3 in the plan).

This is a screening approximation only (uniform-disk beam dilution, not the real
dome-shaped projected sphere) -- see params.disk_beam_dilution's docstring. Stage B
(verify.py) replaces it with exact Magritte imaging + convolution for the shortlist
this module produces.
"""

import csv
import numpy as np

import params as p


def _read_catalogue(csv_path):
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get('Status') == 'SUCCESS']


def _row_floats(row, *keys):
    return {k: float(row[k]) for k in keys}


def dilute_row(row, freq_hz):
    """Compute this catalogue row's beam-diluted (1,1) predictions at `freq_hz`."""
    v = _row_floats(row, 'numberdensity', 'XNH3', 'radius_req', 'A_MAIN',
                     'Main Hyperfine Optical Depth')
    radius_m = p.catalogue_radius_cm_to_m(v['radius_req'])
    theta_source_arcsec = p.m_to_arcsec(radius_m)
    eta = p.disk_beam_dilution(theta_source_arcsec, freq_hz)

    N0_cm2 = p.catalogue_peak_column_cm2(v['numberdensity'], v['XNH3'], v['radius_req'])

    return dict(
        theta_source_arcsec=theta_source_arcsec,
        eta=eta,
        T_mb_native=v['A_MAIN'],
        T_mb_diluted=v['A_MAIN'] * eta,
        tau_main=v['Main Hyperfine Optical Depth'],
        N0_cm2=N0_cm2,
        N_diluted_cm2=N0_cm2 * eta,
    )


def score_row(diluted, target_T_mb, target_tau_main, target_N_para=None,
              w_T_mb=1.0, w_tau=1.0, w_N=0.5):
    """Combined relative-error score (lower is better); NaN-safe on missing targets."""
    terms, weights = [], []
    if target_T_mb is not None and target_T_mb != 0:
        terms.append(abs(diluted['T_mb_diluted'] - target_T_mb) / abs(target_T_mb))
        weights.append(w_T_mb)
    if target_tau_main is not None and target_tau_main != 0:
        terms.append(abs(diluted['tau_main'] - target_tau_main) / abs(target_tau_main))
        weights.append(w_tau)
    if target_N_para is not None and target_N_para != 0:
        terms.append(abs(diluted['N_diluted_cm2'] - target_N_para) / abs(target_N_para))
        weights.append(w_N)
    if not terms:
        return np.nan
    return float(np.average(terms, weights=weights))


def screen_source(source_name, csv_path, main_group_fraction=p.MAIN_GROUP_FRACTION,
                   top_k=10):
    """Rank a catalogue's rows against one W33 source's observed (1,1) parameters.

    Returns the top_k rows (as dicts merging the original row + diluted quantities
    + score), sorted best-first.
    """
    src = p.source(source_name)
    line = src['lines'].get('1,1')
    if line is None:
        raise KeyError(f"no (1,1) line parameters for {source_name!r}")

    target_T_mb = abs(line['T_mb'])  # W33 Main's (1,1) is in absorption (negative)
    target_tau_main = line['tau'] * main_group_fraction
    target_N_para = src.get('N_para')
    freq_hz = p.FREQ_HZ['1,1']

    rows = _read_catalogue(csv_path)
    scored = []
    for row in rows:
        diluted = dilute_row(row, freq_hz)
        s = score_row(diluted, target_T_mb, target_tau_main, target_N_para)
        if np.isnan(s):
            continue
        scored.append({**row, **diluted, 'score': s})

    scored.sort(key=lambda d: d['score'])
    return scored[:top_k]


def print_shortlist(source_name, csv_path, top_k=10):
    src = p.source(source_name)
    line = src['lines']['1,1']
    print(f"{source_name}: target T_mb(1,1)={abs(line['T_mb']):.2f} K, "
          f"tau_main~{line['tau']*p.MAIN_GROUP_FRACTION:.2f}, "
          f"N_para={src.get('N_para', float('nan')):.2e} cm^-2 "
          f"[T_kin={src['T_kin']} K, n_H2={src['n_H2']:.2e} cm^-3 from Table A.5]")
    shortlist = screen_source(source_name, csv_path, top_k=top_k)
    if not shortlist:
        print("  (no rows scored -- check catalogue path/targets)")
        return shortlist
    print(f"  {'T_cloud':>7} {'X_NH3':>8} {'n(cm-3)':>10} {'theta(as)':>9} "
          f"{'eta':>6} {'T_mb_dil':>9} {'tau_main':>9} {'N_dil':>10} {'score':>7}")
    for r in shortlist:
        print(f"  {float(r['T_cloud']):7.1f} {float(r['XNH3']):8.1e} "
              f"{float(r['numberdensity']):10.2e} {r['theta_source_arcsec']:9.3f} "
              f"{r['eta']:6.3f} {r['T_mb_diluted']:9.3f} {r['tau_main']:9.3f} "
              f"{r['N_diluted_cm2']:10.2e} {r['score']:7.3f}")
    return shortlist


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/yasho379/magritte_rebuilt/output_test_1e-6_parallel_12rays_v2/"
        "results/NLTE_nh3_1e-6_parallel_12rays_v2.csv"
    )
    for name in p.EMISSION_SOURCES:
        print_shortlist(name, csv_path, top_k=5)
        print()
