import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_file = 'NLTE_nh3_results_stutzki.csv'

df=pd.read_csv(csv_file,comment="#")


x=df['numberdensity']
y=df['N_NH3']

# Ensure each plot has only one color bar and is saved separately
cmap = 'viridis'

# Scatter plot with R_OSL_MAIN as the color map
plt.figure(figsize=(10, 5))
scatter1 = plt.scatter(x, y, c=df['R_10_MAIN'], cmap='viridis', edgecolor='k')
plt.colorbar(scatter1, label='R_10_MAIN')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('numberdensity')
plt.ylabel('NNH3')
plt.title('Scatter Plot with R_10_MAIN as Color Map')
plt.grid(True)
plt.savefig('scatter_r_10_main.png', dpi=300)
plt.close()

# Scatter plot with R_OSH_MAIN as the color map
plt.figure(figsize=(10, 5))
scatter2 = plt.scatter(x, y, c=df['R_01_MAIN'], cmap='plasma', edgecolor='k')
plt.colorbar(scatter2, label='R_01_MAIN')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('numberdensity')
plt.ylabel('NNH3')
plt.title('Scatter Plot with R_01_MAIN as Color Map')
plt.grid(True)
plt.savefig('scatter_r_01_main.png', dpi=300)
plt.close()

# Scatter plot with A_OSH / A_OSL as the color map
z = df['A_01'] / df['A_10']
plt.figure(figsize=(10, 5))
scatter3 = plt.scatter(x, y, c=z, cmap='inferno', edgecolor='k')
plt.colorbar(scatter3, label='A_01 / A_10')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('numberdensity')
plt.ylabel('NNH3')
plt.title('Scatter Plot with A_01 / A_10 as Color Map')
plt.grid(True)
plt.savefig('scatter_a_01_over_a_10.png', dpi=300)
plt.close()

# Scatter plot with R_ISH_MAIN as the color map
plt.figure(figsize=(10, 5))
scatter4 = plt.scatter(x, y, c=df['R_12_MAIN'], cmap='cividis', edgecolor='k')
plt.colorbar(scatter4, label='R_12_MAIN')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('numberdensity')
plt.ylabel('NNH3')
plt.title('Scatter Plot with R_12_MAIN as Color Map')
plt.grid(True)
plt.savefig('scatter_r_12_main.png', dpi=300)
plt.close()

# Scatter plot with R_ISL_MAIN as the color map
plt.figure(figsize=(10, 5))
scatter5 = plt.scatter(x, y, c=df['R_21_MAIN'], cmap='magma', edgecolor='k')
plt.colorbar(scatter5, label='R_21_MAIN')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('numberdensity')
plt.ylabel('NNH3')
plt.title('Scatter Plot with R_21_MAIN as Color Map')
plt.grid(True)
plt.savefig('scatter_r_21_main.png', dpi=300)
plt.close()


