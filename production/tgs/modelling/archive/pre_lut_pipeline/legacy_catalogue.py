"""Recover what is usable from the pre-fix 878-row catalogue.

WHAT WAS WRONG WITH IT
----------------------
The catalogue in output_test_1e-6_parallel_12rays_v2/ was computed before the
hyperfine labelling fix, so its outer satellite columns are interchanged:
`R_01_MAIN` actually holds F1 = 1->0 and `R_10_MAIN` holds F1 = 0->1. The inner
pair was never affected.

That is not an assumption. Checked against the file itself, on 555
well-converged rows with 0.05 < tau < 20:

    correct outer sense R(0->1) > R(1->0), as labelled ...    0.0% of rows
    correct outer sense after swapping the outer pair ...   100.0% of rows
    correct inner sense R(1->2) > R(2->1), either way ...     98.9% of rows

The published anomaly sense (Stutzki & Winnewisser 1985; his own 1984 Table 2;
Camarata et al. 2015) is R(0->1) > R(1->0) and R(1->2) > R(2->1). A single swap
of the outer pair reproduces it exactly, and leaves the inner pair alone.

The catalogue also shows no collapsed fits: zero rows sit at the amplitude
floor, including all 225 rows above tau = 5. The old fitter seeded component
centres at the velocity-window midpoint rather than from the brightest channel,
so it never hit the self-absorption failure that affected the newer code.

WHAT THE SWAP DOES NOT FIX
--------------------------
These rows were computed with nrays=12 (the coarsest HEALPix quadrature),
resolution=5 (about 18 interior points, and NOT constant across the grid, since
the remesher's density contrast against a fixed 100 cm^-3 background varied by
four orders of magnitude), an NH3-bearing envelope outside the sphere, centre-
pixel rather than disc-averaged imaging, and peak rather than integrated ratios.

So the corrected rows ARE usable for:
  - cross-validating the new lookup table at overlapping parameters, where any
    disagreement usefully quantifies how much the numerics mattered;
  - qualitative trends, which should survive the numerical differences;
  - developing and exercising the interpolation, chi^2-landscape and inversion
    machinery before the new table exists.

And are NOT usable for quantitative parameter retrieval, published values, or
seeding search bounds -- the numerics differ from the new table, and the
inconsistent mesh means its interpolation-error behaviour will not represent
the real one. Every row is stamped `is_legacy=True` so this cannot be forgotten
downstream.
"""

import os
import csv
import numpy as np

LEGACY_CSV = ("/home/yasho379/magritte_rebuilt/scratch/output/output_test_1e-6_parallel_12rays_v2/"
              "results/NLTE_nh3_1e-6_parallel_12rays_v2.csv")

# The numerics these rows were produced with, recorded so a comparison against
# the new table is never made without them in view.
LEGACY_PROVENANCE = dict(
    is_legacy=True, nrays=12, resolution=5, spectrum='center', estimator='peak',
    bare_clump=False, outer_pair_swapped_on_read=True,
    note='pre-fix catalogue; see legacy_catalogue module docstring',
)

CONV_THRESHOLD = 90.0


def load_legacy_catalogue(path=None, min_convergence=CONV_THRESHOLD,
                           tau_range=None, require_success=True):
    """Load the pre-fix catalogue with the outer satellite pair swapped back.

    Returns a list of dicts with float-converted physical columns, corrected
    ratios, derived HIA values, and LEGACY_PROVENANCE merged in.
    """
    path = path or LEGACY_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    out = []
    with open(path, newline='') as f:
        for raw in csv.DictReader(f):
            if require_success and raw.get('Status') != 'SUCCESS':
                continue
            try:
                conv = float(raw['Final Convergence'])
                tau = float(raw['Main Hyperfine Optical Depth'])
            except (KeyError, ValueError):
                continue
            if min_convergence is not None and conv < min_convergence:
                continue
            if tau_range is not None and not (tau_range[0] <= tau <= tau_range[1]):
                continue

            # THE SWAP: the file's R_01_MAIN is really F1 = 1->0 and vice versa.
            r01_true = float(raw['R_10_MAIN'])
            r10_true = float(raw['R_01_MAIN'])
            r12 = float(raw['R_12_MAIN'])
            r21 = float(raw['R_21_MAIN'])

            row = dict(LEGACY_PROVENANCE)
            row.update(
                T_cloud=float(raw['T_cloud']), vturb=float(raw['vturb']),
                XNH3=float(raw['XNH3']), numberdensity=float(raw['numberdensity']),
                radius_req=float(raw['radius_req']),
                N_NH3_proxy=float(raw['N_NH3']),
                tau_main=tau, final_convergence=conv,
                R_01_MAIN=r01_true, R_10_MAIN=r10_true,
                R_12_MAIN=r12, R_21_MAIN=r21,
                # amplitudes swap identically
                A_01=float(raw['A_10']), A_10=float(raw['A_01']),
                A_12=float(raw['A_12']), A_21=float(raw['A_21']),
                A_MAIN=float(raw['A_MAIN']),
                # redshifted/blueshifted anomaly ratios (Zhou/Wu convention)
                HIA_OS=(r01_true / r10_true) if r10_true else np.nan,
                HIA_IS=(r21 / r12) if r12 else np.nan,
            )
            out.append(row)
    return out


def anomaly_sense_fraction(rows):
    """Fraction of rows showing the published anomaly sense, outer and inner.

    A corrected catalogue should give ~1.0 for outer; the inner fraction is
    ~0.99 in the data and its shortfall is not caused by the swap.
    """
    if not rows:
        return dict(outer=np.nan, inner=np.nan, n=0)
    o = np.mean([r['R_01_MAIN'] > r['R_10_MAIN'] for r in rows])
    i = np.mean([r['R_12_MAIN'] > r['R_21_MAIN'] for r in rows])
    return dict(outer=float(o), inner=float(i), n=len(rows))


def assert_not_mistaken_for_lut(rows):
    """Guard: refuse to proceed if legacy rows reach code expecting the new table."""
    if any(r.get('is_legacy') for r in rows):
        raise ValueError(
            "legacy catalogue rows passed to code expecting the new lookup table. "
            "They were computed with nrays=12, resolution=5 (non-uniform mesh), an "
            "NH3 envelope, centre-pixel imaging and peak ratios -- valid for trend "
            "checks and machinery development, not for retrieval or published values."
        )
