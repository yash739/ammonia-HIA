"""Every filesystem location the code uses, in one place.

All paths are derived from this file's own location, so the repository can
be cloned anywhere. Layout assumed (see production/tgs/modelling/README.md):

    <REPO>/                       repository root
    <REPO>/production/            load-bearing data products (grids, results)
    <REPO>/production/tgs/        Magritte working directory (model_files/ cache,
                                  LAMDA file); passed to run_model as `wdir`
    <REPO>/production/tgs/modelling/   this code
    <REPO>/scratch/               disposable diagnostics (gitignored)
"""
import os

MODELLING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TGS = os.path.dirname(MODELLING)
PRODUCTION = os.path.dirname(TGS)
REPO = os.path.dirname(PRODUCTION)
SCRATCH = os.path.join(REPO, 'scratch')

# Magritte's working directory. run_model() expects a trailing separator.
WDIR = os.path.join(TGS, '')

# Para-NH3 LAMDA file with hyperfine-resolved NH3-H2 rates (Loreau et al. 2023).
LAMDA_NH3 = os.path.join(TGS, 'p-nh3@loreau.dat.txt')

# Digitized observations (small, curated, tracked in git).
OBSERVED_DATA = os.path.join(MODELLING, 'data', 'observed')

# Published papers and transcribed rate tables.
REFERENCES = os.path.join(PRODUCTION, 'references')

# Data products.
LUT_GOLD = os.path.join(PRODUCTION, 'output_lut_gold')          # main 3D grid
LUT_GOLD_CSV = os.path.join(LUT_GOLD, 'results', 'lut_dv0.30.csv')
LUT_FIG56 = os.path.join(PRODUCTION, 'output_lut_fig56')        # Figs 5/6 grid
LUT_FIG56_CSV = os.path.join(LUT_FIG56, 'results', 'lut_dv0.30.csv')
ESCAPE1D = os.path.join(PRODUCTION, 'output_escape1d')          # 1D model grids
