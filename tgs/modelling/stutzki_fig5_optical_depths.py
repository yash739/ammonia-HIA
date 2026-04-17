import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_grouped_stutzki_plot(csv_filename):
    # 1. Load the data and filter successful runs
    df = pd.read_csv(csv_filename)
    df = df[df['Status'] == 'SUCCESS']

    # 2. Sort by Optical Depth so the line draws correctly from left to right
    df = df.sort_values(by='Main Hyperfine Optical Depth')

    # 3. Calculate the Ratio of two satellites (Inner vs Outer)
    # Stutzki (1985) investigates the inner/outer satellite anomaly. 
    # Here we use A_10 (inner) and A_01 (outer).
    df['Satellite_Ratio'] = df['A_10'] / df['A_01']

    # 4. Set up the plot
    plt.figure(figsize=(10, 7))

    # Dynamically extract all unique Temperatures and Number Densities available
    temps = sorted(df['T_cloud'].unique(), reverse=True) # E.g., 36, 24, 18
    densities = sorted(df['numberdensity'].unique())     # E.g., 1e4 to 1e8

    # Create mapping dictionaries for distinct Colors (Temperatures) and Linestyles (Densities)
    colors = plt.cm.Set1(np.linspace(0, 1, len(temps)))
    color_map = {t: colors[i] for i, t in enumerate(temps)}
    
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]
    linestyle_map = {d: linestyles[i % len(linestyles)] for i, d in enumerate(densities)}

    # 5. Loop through every combination and plot a line
    for t in temps:
        for d in densities:
            # Subset the dataframe for this specific combination
            subset = df[(df['T_cloud'] == t) & (df['numberdensity'] == d)]
            
            # If the subset has data, plot it
            if not subset.empty:
                plt.plot(subset['Main Hyperfine Optical Depth'], subset['Satellite_Ratio'], 
                         color=color_map[t], linestyle=linestyle_map[d], 
                         linewidth=2.5, alpha=0.85,
                         label=f'T={t} K, n={d:.1e} cm$^{{-3}}$')

    # 6. Add the Theoretical Optically Thin LTE Ratio (~0.278 / 0.222 = ~1.252)
    plt.axhline(y=(0.278/0.222), color='black', linestyle='-', alpha=0.8, label='LTE Ratio (~1.25)')

    # 7. Formatting
    plt.xscale('log')
    plt.xlabel('Main Hyperfine Optical Depth ($\\tau_{main}$)', fontsize=12)
    plt.ylabel('Inner / Outer Satellite Ratio ($A_{10} / A_{01}$)', fontsize=12)
    plt.title('NH$_3$ Hyperfine Satellite Ratio vs. Optical Depth (Stutzki Style)', fontsize=14)

    # Place the legend outside the plot box to avoid overlapping the data curves
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()

    # 8. Save and display
    plt.savefig('grouped_stutzki_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run the function
create_grouped_stutzki_plot('NLTE_nh3_1e-6_parallel_12rays_v2.csv')