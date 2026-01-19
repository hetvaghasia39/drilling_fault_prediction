import kagglehub
import pandas as pd
import os
import glob

def download_data():
    """Downloads the XAI Drilling Dataset from KaggleHub."""
    print("Downloading dataset...")
    path = kagglehub.dataset_download("raphaelwallsberger/xai-drilling-dataset")
    print("Path to dataset files:", path)
    return path

def load_data(data_path=None):
    """
    Loads the drilling dataset into a pandas DataFrame.
    If data_path is not provided, it downloads the data first.
    """
    if data_path is None:
        data_path = download_data()
    
    # Find the CSV file in the directory
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {data_path}")
    
    # Assuming the first CSV is the main dataset
    csv_file = csv_files[0]
    print(f"Loading data from: {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
