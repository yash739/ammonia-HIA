"""Validate build_lut's writer on a genuinely fast (high-density) point."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_lut as b

os.makedirs("/home/yasho379/magritte_rebuilt/scratch/output/output_lut_smoke/results", exist_ok=True)
os.makedirs("/home/yasho379/magritte_rebuilt/scratch/output/output_lut_smoke/fits", exist_ok=True)
os.makedirs("/home/yasho379/magritte_rebuilt/scratch/output/output_lut_smoke/images", exist_ok=True)
os.makedirs("/home/yasho379/magritte_rebuilt/scratch/output/output_lut_smoke/spectra", exist_ok=True)

log_n, T, log_ndv = 7.3, 27.8, 14.68   # the exact Phase 0 verification point -- 219s at resolution=10
n = 10.0 ** log_n
rs, N = b.radius_from_N_dv(10.0 ** log_ndv, b.REFERENCE_DV_KMS, n, b.XNH3_FIDUCIAL)
import params as p
task = dict(log_n=log_n, T_cloud=T, log_ndv=log_ndv, dv=b.REFERENCE_DV_KMS,
           XNH3=b.XNH3_FIDUCIAL, numberdensity=n, radius_sphere=rs, N_NH3_target=N,
           vturb=p.fwhm_kms_to_vturb_ms(b.REFERENCE_DV_KMS), wdir=b.WDIR,
           odir="/home/yasho379/magritte_rebuilt/scratch/output/output_lut_smoke/",
           key=b.point_key(log_n, T, log_ndv))

t0 = time.time()
row, spectra = b._worker(task)
print(f"wall time: {time.time()-t0:.1f}s")
print("row Status:", row.get('Status'))
if row.get('Status') == 'SUCCESS':
    for k in ('R_01_MAIN','R_10_MAIN','R_21_MAIN','R_12_MAIN','R_22_MAIN','R_21_11',
              'tau_main','final_convergence','npoints'):
        print(f"  {k} = {row.get(k)}")
    print("spectra keys:", list(spectra.keys()) if spectra else None)
    if spectra:
        for k,v in spectra.items():
            print(f"    {k}: shape={getattr(v,'shape',None)}")
else:
    print("Error:", row.get('Error_Msg'))
