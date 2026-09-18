"""Re-run gold's models for one temperature slice, on a chosen mesh, to test
whether Stutzki & Winnewisser (1985) Figs. 5 and 6 can be faithfully reproduced.

Differences from the gold build, each fixing a verified quirk
(see important_notes/magritte-quirks-2026-09-16.md):
  * mesh selectable ('cube' reproduces gold's point cloud; 'radial' fixes tau(b));
  * fov_pad_factor 1.0 by default, so the field of view no longer reshapes the mesh;
  * tau of the (1,1) main line is read from the (1,1) optical-depth image on its
    own spectral window -- run_model's tau_main is the LAST-imaged line's tau;
  * disc and centre spectra are built from the imager's own ImX/ImY (b <= R and
    min b), with no assumed image-extent factor;
  * ratios use the production estimator (baseline subtraction, 5-Gaussian fit,
    component identification by fitted centre) plus native channel peaks.

Resumable by (mesh, mode, log_n, T, log_ndv) key. Per-model images are saved as
compressed float32 npz so everything can be re-derived without re-running.
"""
import argparse
import csv
import multiprocessing
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELLING = os.path.dirname(_HERE)

MESHES = {
    'cube': None,
    'radial': {'kind': 'radial', 'n_shell': 14, 'base': 260, 'seed': 7},
    'radial_s10b140': {'kind': 'radial', 'n_shell': 10, 'base': 140, 'seed': 7},
    'radial_s8b90': {'kind': 'radial', 'n_shell': 8, 'base': 90, 'seed': 7},
}
FIG56_LOG_N = (3.5, 5.0, 7.0)
LOG_NDV = (14.0, 14.225, 14.45, 14.675, 14.9, 15.125, 15.35, 15.575, 15.8)
KEYS = ('A_10', 'A_12', 'A_MAIN', 'A_21', 'A_01')
FIELDS = ['mesh', 'mode', 'log_n_H2', 'T_cloud', 'log_N_dv', 'radius_pc', 'npoints',
          'halting_iter', 'final_convergence', 'wall_s', 'tau_main_reported',
          'tau11_chord', 'tau22_chord', 'chord_rms', 'chord_monotonic', 'chord_limb',
          'bright_chord_rms', 'bright_monotonic', 'sat_frac',
          'disc_fit_R01', 'disc_fit_R10', 'disc_fit_R12', 'disc_fit_R21', 'disc_fit_ok',
          'disc_pk_R01', 'disc_pk_R10', 'disc_pk_R12', 'disc_pk_R21',
          'cen_fit_R01', 'cen_fit_R10', 'cen_fit_R12', 'cen_fit_R21',
          'cen_pk_R01', 'cen_pk_R10', 'cen_pk_R12', 'cen_pk_R21',
          'disc_A_MAIN', 'Status', 'Error_Msg']


def mesh_spec(name):
    """Named mesh, or 'radial_s<n_shell>b<base>' for an ad-hoc radial mesh."""
    if name in MESHES:
        return MESHES[name]
    import re
    if re.fullmatch(r'cube_r\d+', name):
        return None
    m = re.fullmatch(r'radial_s(\d+)b(\d+)', name)
    if not m:
        raise KeyError(name)
    return {'kind': 'radial', 'n_shell': int(m.group(1)), 'base': int(m.group(2)), 'seed': 7}


def key_of(mesh, mode, log_n, T, log_ndv):
    return f"{mesh}|{mode}|{log_n:.4f}|{T:.4f}|{log_ndv:.4f}"


def _worker(task):
    os.environ["OMP_NUM_THREADS"] = str(task['threads'])
    for pth in (_HERE, _MODELLING):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    import lte_probe as L
    import nh3_hyperfine as hf
    import params as p
    from nh3_NLTE_analysis import fit_five_gaussians, subtract_baseline, intensity_to_Tmb
    from nh3_NLTE_sphere import run_model

    row = dict(mesh=task['mesh'], mode=task['mode'], log_n_H2=task['log_n'],
               T_cloud=task['T'], log_N_dv=task['log_ndv'])
    t0 = time.time()
    try:
        n = 10.0 ** task['log_n']
        N = 10.0 ** task['log_ndv'] * L.DV_KMS
        radius = N / (2.0 * n * L.XNH3)
        row['radius_pc'] = radius / 3.0857e18
        odir = os.path.join(task['odir'], 'work', task['mesh'], '')
        for sub in ('fits', 'images'):
            os.makedirs(os.path.join(odir, sub), exist_ok=True)
        hi, conv, tau_main, extra, npoints, nb = run_model(
            wdir=L.WDIR, odir=odir, XNH3=L.XNH3, numberdensity=n,
            vturb=p.fwhm_kms_to_vturb_ms(L.DV_KMS), T_cloud=task['T'],
            max_NLTE=task['max_nlte'], radius_sphere=radius, nrays=12,
            resolution=int(task['mesh'].split('_r')[1]) if task['mesh'].startswith('cube_r') else 10,
            nx_pix=task['npix'], ny_pix=task['npix'], spectrum='integrated',
            fov_pad_factor=task['pad'], mesh=mesh_spec(task['mesh']), return_image=True)
        row.update(npoints=int(npoints), halting_iter=int(hi), final_convergence=float(conv),
                   tau_main_reported=float(tau_main))

        f11 = p.FREQ_HZ['1,1']
        img = extra['11_image']
        b = np.hypot(img['ImX'], img['ImY']) / img['r_out']
        run = dict(extra=extra, T=task['T'])
        _, t11 = L.tau_from_image(run, '11')
        _, t22 = L.tau_from_image(run, '22')
        ic = int(np.argmin(b))
        row['tau11_chord'] = float(t11[ic])
        row['tau22_chord'] = float(t22[int(np.argmin(np.hypot(extra['22_image']['ImX'], extra['22_image']['ImY'])))])
        cm = L.chord_metrics(b, t11)
        row.update(chord_rms=cm['rms'], chord_monotonic=cm['monotonic'], chord_limb=cm['limb'])
        bb, tb, sat = L.tau_from_brightness(run, '11')
        if sat < 0.97:
            bm = L.chord_metrics(bb, tb)
            row.update(bright_chord_rms=bm['rms'], bright_monotonic=bm['monotonic'])
        row['sat_frac'] = sat

        v = hf.freq_to_radio_velocity_kms(img['freqs'], f11)
        order = np.argsort(v)
        v = v[order]
        I = img['I'][:, order]
        disc = I[b <= 1.0, :].mean(axis=0)
        cen = I[ic, :]
        for tag, spec in (('disc', disc), ('cen', cen)):
            T = subtract_baseline(v, intensity_to_Tmb(1000 * v, spec, f11))
            off = dict(zip(KEYS, hf.NH3_11_OFFSETS_KMS))
            for k in ('01', '10', '12', '21'):
                row[f'{tag}_pk_R{k}'] = float(
                    np.max(T[np.abs(v - off[f'A_{k}']) < 1.0])
                    / np.max(T[np.abs(v) < 1.0]))
            try:
                pars = fit_five_gaussians(v, T, 'one')
                amps, cens = pars[0:15:3], pars[1:15:3]
                idx = hf.identify_components(cens - hf.estimate_v_sys_kms(v, T))
                A = {kk: float(amps[i]) for kk, i in idx.items()}
                for k in ('01', '10', '12', '21'):
                    row[f'{tag}_fit_R{k}'] = A[f'A_{k}'] / A['A_MAIN']
                if tag == 'disc':
                    row['disc_A_MAIN'] = A['A_MAIN']
                    row['disc_fit_ok'] = True
            except Exception as fe:
                if tag == 'disc':
                    row['disc_fit_ok'] = False
                    row['Error_Msg'] = f"fit: {fe}"[:200]

        np.savez_compressed(
            os.path.join(task['odir'], 'images', task['key'].replace('|', '_') + '.npz'),
            ImX=img['ImX'].astype(np.float32), ImY=img['ImY'].astype(np.float32),
            r_out=img['r_out'], freqs=img['freqs'], I11=img['I'].astype(np.float32),
            tau11=img['tau'].astype(np.float32))
        row['Status'] = 'SUCCESS'
    except Exception as e:
        row['Status'] = 'FAILED'
        row['Error_Msg'] = f"{type(e).__name__}: {e}"[:300]
    row['wall_s'] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('lte', 'nlte'), default='nlte')
    ap.add_argument('--meshes', default='radial,cube')
    ap.add_argument('--T', type=float, default=18.0)
    ap.add_argument('--log-n', default=','.join(str(x) for x in FIG56_LOG_N))
    ap.add_argument('--log-ndv', default=','.join(str(x) for x in LOG_NDV))
    ap.add_argument('--processes', type=int, default=5)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--max-nlte', type=int, default=300)
    ap.add_argument('--pad', type=float, default=1.0)
    ap.add_argument('--npix', type=int, default=32)
    ap.add_argument('--odir', default='/home/yasho379/magritte_rebuilt/scratch/output/fig56_T18/')
    a = ap.parse_args()

    os.makedirs(os.path.join(a.odir, 'images'), exist_ok=True)
    out_csv = os.path.join(a.odir, f'slice_{a.mode}.csv')
    done = set()
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                if r['Status'] == 'SUCCESS':
                    done.add(key_of(r['mesh'], r['mode'], float(r['log_n_H2']),
                                    float(r['T_cloud']), float(r['log_N_dv'])))
    max_nlte = 0 if a.mode == 'lte' else a.max_nlte
    log_ns = [float(x) for x in a.log_n.split(',')]
    log_ndvs = [float(x) for x in a.log_ndv.split(',')]
    tasks = []
    # meshes in the order given (radial first by default); within a mesh,
    # high density first -- those converge fastest, so curves fill early.
    for mesh in a.meshes.split(','):
        for log_n in sorted(log_ns, reverse=True):
            for log_ndv in log_ndvs:
                k = key_of(mesh, a.mode, log_n, a.T, log_ndv)
                if k in done:
                    continue
                tasks.append(dict(mesh=mesh, mode=a.mode, log_n=log_n, T=a.T, log_ndv=log_ndv,
                                  max_nlte=max_nlte, pad=a.pad, npix=a.npix, odir=a.odir,
                                  threads=a.threads, key=k))
    print(f"{len(tasks)} to run, {len(done)} done -> {out_csv}", flush=True)
    header = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=a.processes, maxtasksperchild=1) as pool, \
            open(out_csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        if header:
            w.writeheader()
            f.flush()
        for i, row in enumerate(pool.imap_unordered(_worker, tasks), 1):
            w.writerow(row)
            f.flush()
            print(f"[{i}/{len(tasks)}] {row['mesh']} n={row['log_n_H2']} N/dv={row['log_N_dv']} "
                  f"{row['Status']} {row.get('wall_s')}s conv={row.get('final_convergence')} "
                  f"tau11={row.get('tau11_chord')} {row.get('Error_Msg', '')}", flush=True)


if __name__ == '__main__':
    main()
