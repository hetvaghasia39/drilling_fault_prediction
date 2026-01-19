import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_deviation():
    project_dir = Path(__file__).resolve().parents[2]
    input_file = project_dir.joinpath('reports', 'rop_deviation_results.csv')
    output_dir = project_dir.joinpath('reports', 'figures')
    
    df = pd.read_csv(input_file)
    
    # 1. Depth vs ROP (Actual vs Predicted)
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(df['Depth'], df['ROP_AVG'], label='Actual ROP', color='black', alpha=0.7)
    ax.plot(df['Depth'], df['ROP_Pred'], label='Predicted ROP (Digital Twin)', color='green', linestyle='--')
    ax.set_ylabel('Rate of Penetration')
    ax.set_xlabel('Depth')
    ax.set_title('Drilling Digital Twin: Actual vs Predicted ROP')
    ax.legend()
    # Highlight the "Event" zone (roughly > 3950 based on EDA)
    ax.axvline(x=3950, color='red', linestyle=':', label='Potential Event Start')
    plt.savefig(output_dir / 'digital_twin_rop.png')
    plt.close()
    
    # 2. Deviation Track (Health Signal)
    fig, ax = plt.subplots(figsize=(20, 6))
    
    # Color code deviation: Green (Ok), Red (Problem)
    colors = np.where(df['ROP_Deviation'] > 0, 'blue', 'red') # Positive = Fast, Negative = Slow
    
    ax.bar(df['Depth'], df['ROP_Deviation'], width=2, color=colors)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel('ROP Deviation (Actual - Pred)')
    ax.set_xlabel('Depth')
    ax.set_title('Drilling Efficiency Deviation Log')
    plt.savefig(output_dir / 'deviation_log.png')
    plt.close()
    
    print("Deviation plots saved to reports/figures/")

if __name__ == '__main__':
    import numpy as np # Needed for plotting logic
    plot_deviation()
