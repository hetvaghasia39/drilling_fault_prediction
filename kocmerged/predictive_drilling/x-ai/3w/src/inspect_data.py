import kagglehub
import pandas as pd
import os
import shutil

try:
    # Download latest version
    path = kagglehub.dataset_download("ahmedelbashir99/drilling-log-dataset")
    print(f"Dataset downloaded to: {path}")

    # List files in the downloaded directory
    files = os.listdir(path)
    print("Files found:", files)

    # Load the csv file
    csv_files = [f for f in files if f.endswith('.csv')]
    if csv_files:
        filename = csv_files[0]
        data_path = os.path.join(path, filename)
        print(f"Reading file: {data_path}")
        
        # Create data directory if not exists
        os.makedirs('data', exist_ok=True)
        
        # Safe copy
        target_path = os.path.join('data', 'drilling_log.csv') # Rename to safe name
        print(f"Copying to: {target_path}")
        shutil.copy(data_path, target_path)
        
        df = pd.read_csv(target_path)
        print("\nDataset Info:")
        print(df.info())
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nColumns:")
        print(df.columns.tolist())
    else:
        print("No CSV file found in the dataset directory.")

except Exception as e:
    print(f"An error occurred: {e}")
