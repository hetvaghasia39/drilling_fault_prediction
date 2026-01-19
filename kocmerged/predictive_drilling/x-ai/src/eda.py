import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging
from data_loader import load_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def perform_eda():
    """Performs Exploratory Data Analysis and saves plots."""
    logging.info("Starting EDA process...")
    
    # Load data
    df = load_data()
    logging.info(f"Columns: {df.columns.tolist()}")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Target Distribution (Main Failure)
    # The user says "Main failure" is the label, but let's check the column name.
    # Based on previous output, columns might be different. 
    # Let's inspect columns first in correct run, but assuming 'Main failure' or similar exists.
    # If not, we'll find it.
    
    # Check for likely target columns
    potential_targets = [c for c in df.columns if 'failure' in c.lower() or 'label' in c.lower()]
    logging.info(f"Potential target columns: {potential_targets}")
    
    if not potential_targets:
        logging.warning("No obvious target column found. Plotting all categorical distributions.")
    
    # 2. Correlation Matrix
    logging.info("Generating correlation matrix...")
    plt.figure(figsize=(12, 10))
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()
    
    # 3. Distributions of Numerical Features
    logging.info("Generating numerical distributions...")
    for col in numeric_df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        filename = f"dist_{col.replace(' ', '_').replace('/', '_').replace('[', '').replace(']', '')}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        
    logging.info("EDA completed. Plots saved to reports/figures/.")

if __name__ == "__main__":
    perform_eda()
