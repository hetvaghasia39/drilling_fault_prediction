import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def generate_eda():
    project_dir = Path(__file__).resolve().parents[2]
    input_file = project_dir.joinpath('data', 'processed', 'drilling_log_processed.csv')
    output_dir = project_dir.joinpath('reports', 'figures')
    
    df = pd.read_csv(input_file)
    
    print("Generating EDA...")
    print(f"Data Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 1. Distribution of Target (ROP_AVG)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['ROP_AVG'], kde=True, bins=30)
    plt.title('Distribution of ROP_AVG')
    plt.savefig(output_dir / 'rop_distribution.png')
    plt.close()
    
    # 2. Correlation Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.savefig(output_dir / 'correlation_matrix.png')
    plt.close()
    
    # 3. Depth vs ROP (Time Series proxy)
    plt.figure(figsize=(15, 6))
    plt.plot(df['Depth'], df['ROP_AVG'], label='ROP_AVG')
    plt.xlabel('Depth')
    plt.ylabel('ROP_AVG')
    plt.title('ROP_AVG vs Depth')
    plt.legend()
    plt.savefig(output_dir / 'rop_vs_depth.png')
    plt.close()
    
    # 4. Pairplot of key features
    key_features = ['WOB', 'SURF_RPM', 'ROP_AVG', 'Depth']
    sns.pairplot(df[key_features])
    plt.savefig(output_dir / 'pairplot.png')
    plt.close()

    # 5. Descriptive Stats
    desc = df.describe()
    desc.to_csv(project_dir.joinpath('reports', 'descriptive_stats.csv'))
    print("EDA Visualizations saved to reports/figures/")
    print("Descriptive stats saved to reports/descriptive_stats.csv")
    
    # Analysis for Failure Prediction
    # Assuming low ROP might indicate issues, let's look at low quantiles
    low_rop_threshold = df['ROP_AVG'].quantile(0.10)
    print(f"10th percentile of ROP: {low_rop_threshold}")
    
    # Check for NaNs
    print("Missing Values:\n", df.isnull().sum())

if __name__ == "__main__":
    generate_eda()
