"""Grid axes of the Magritte lookup tables, shared by the grid builders and
by everything that needs to know the grid (the 1D escape-probability grid is
computed on the same axes, and the validation scripts check coverage
against them).

Gold grid (production/output_lut_gold/): the main 3D non-LTE table, a full
cross product of 14 x 14 x 9 = 1764 models at Delta v = 0.3 km/s.
T_AXIS lands on Stutzki & Winnewisser's (1985) Fig. 4 panel temperatures
(18, 24, 30, 36 K).

Fig. 5/6 grid (production/output_lut_fig56/): Stutzki's Fig. 5/6
temperatures (including 26 K, which the gold axis skips) at eight
densities; 3 x 8 x 9 = 216 models, computed after the tau_main fix.
"""
LOG_N_AXIS = [3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.5, 7.0, 7.5]
T_AXIS = [float(t) for t in range(9, 49, 3)]  # 9, 12, ..., 48 K
LOG_NDV_AXIS = [14.0, 14.225, 14.45, 14.675, 14.9, 15.125, 15.35, 15.575, 15.8]

FIG56_LOG_N_AXIS = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
FIG56_T_AXIS = [18.0, 26.0, 36.0]
FIG56_LOG_NDV_AXIS = LOG_NDV_AXIS
