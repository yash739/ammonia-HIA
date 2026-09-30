"""LTE geometry probe for the sphere model.

With max_NLTE=0 the level populations are LTE, so there is no hyperfine anomaly
and the optical-depth profile across the projected disc is known exactly for a
uniform sphere: tau(b) = tau_centre * sqrt(1 - (b/R)^2). Any departure is a
mesh/imaging artefact, not physics. LTE models cost seconds, which is what makes
this usable as a mandatory gate before a production grid.

tau(b) is taken two independent ways and both are reported:
  * from brightness: T = T0 (1 - exp(-tau)) with T0 = T_RJ(T_k) - T_RJ(T_CMB)
    known analytically, inverted per pixel at the (1,1) main-line peak;
  * directly from Magritte's optical-depth image on the (1,1) spectral window.
Impact parameters come from the imager's own ImX/ImY, never an assumed extent.
"""
import os
import sys

import numpy as np

from astropy import constants as C

import nh3hia.w33.params as p
from nh3hia import paths

WDIR = paths.WDIR
XNH3 = 1e-8
DV_KMS = 0.3
T_CMB = 2.725
_H, _K, _C = C.h.si.value, C.k_B.si.value, C.c.si.value


def t_rj(T, freq):
    return (_H * freq / _K) / np.expm1(_H * freq / (_K * T))


def run_lte(log_n, T, log_ndv, mesh=None, npix=32, odir='/tmp/lte_probe/',
            fov_pad_factor=1.0, resolution=10, image_lines=None):
    """One LTE model; returns npoints, reported tau_main and per-line images."""
    from nh3hia.model3d.sphere import run_model
    for sub in ('fits', 'images'):
        os.makedirs(os.path.join(odir, sub), exist_ok=True)
    n = 10.0 ** log_n
    N = 10.0 ** log_ndv * DV_KMS
    radius = N / (2.0 * n * XNH3)
    hi, conv, tau_main, extra, npoints, nb = run_model(
        wdir=WDIR, odir=odir, XNH3=XNH3, numberdensity=n,
        vturb=p.fwhm_kms_to_vturb_ms(DV_KMS), T_cloud=T, max_NLTE=0,
        radius_sphere=radius, nrays=12, resolution=resolution, nx_pix=npix, ny_pix=npix,
        spectrum='integrated', fov_pad_factor=fov_pad_factor, mesh=mesh,
        image_lines=image_lines, return_image=True)
    return dict(npoints=int(npoints), tau_main_reported=float(tau_main), extra=extra, T=T)


def _b_over_R(img):
    return np.hypot(img['ImX'], img['ImY']) / img['r_out']


def _main_window(img, freq_rest, half_kms=1.2):
    v = (freq_rest - img['freqs']) / freq_rest * 2.99792458e5
    return np.abs(v) < half_kms


def tau_from_brightness(run, label='11'):
    img = run['extra'][f'{label}_image']
    freq = p.FREQ_HZ[f'{label[0]},{label[1]}']
    m = _main_window(img, freq)
    T_pk = ((_C ** 2 * img['I'][:, m]) / (2 * _K * freq ** 2)).max(axis=1) - t_rj(T_CMB, freq)
    T0 = t_rj(run['T'], freq) - t_rj(T_CMB, freq)
    frac = T_pk / T0
    tau = -np.log(np.clip(1.0 - frac, 1e-12, None))
    return _b_over_R(img), tau, float(np.max(frac))


def tau_from_image(run, label='11'):
    img = run['extra'][f'{label}_image']
    freq = p.FREQ_HZ[f'{label[0]},{label[1]}']
    m = _main_window(img, freq)
    return _b_over_R(img), img['tau'][:, m].max(axis=1)


BINS = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0])


def chord_metrics(b, tau, bins=BINS):
    """Fit A*sqrt(1-b^2) over b<0.9 and report shape diagnostics.

    rms: RMS fractional deviation of the binned profile from the fitted chord
    law (b<0.9); monotonic: binned profile non-increasing within 2%;
    limb: binned value / chord law in the outermost bin (0.9-1.0).
    """
    inside = (b < 0.9) & np.isfinite(tau)
    shape = np.sqrt(np.clip(1.0 - b ** 2, 0.0, None))
    A = float(np.sum(tau[inside] * shape[inside]) / np.sum(shape[inside] ** 2))
    bc, tb, law = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (b >= lo) & (b < hi) & np.isfinite(tau)
        if s.sum() < 2:
            continue
        bm = float(np.mean(b[s]))
        bc.append(bm)
        tb.append(float(np.mean(tau[s])))
        law.append(A * np.sqrt(max(1.0 - bm ** 2, 0.0)))
    bc, tb, law = map(np.array, (bc, tb, law))
    core = bc < 0.9
    rel = tb[core] / law[core]
    monotonic = bool(np.all(np.diff(tb[core]) <= 0.02 * tb[core][:-1]))
    limb = float(tb[~core][0] / law[~core][0]) if (~core).any() and law[~core][0] > 0 else np.nan
    return dict(A=A, rms=float(np.sqrt(np.mean((rel - 1.0) ** 2))), monotonic=monotonic,
                limb=limb, b_bins=bc.tolist(), tau_bins=tb.tolist(), law_bins=law.tolist())
