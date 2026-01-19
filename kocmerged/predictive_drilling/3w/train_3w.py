import pandas as pd
import numpy as np
import os
import glob
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error

# Constants
DATA_DIR = "3w/data/raw/3w_official"
MODEL_DIR = "3w/models"
REQUIRED_COLS = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']
TARGET_COL = 'class'

def load_and_process_data(base_dir, sample_size_per_class=5):
    """
    Loads a sample of files from the 3W dataset.
    3W structure: /0 (Normal), /1 (Abrupt Increase of BSW), etc.
    We'll walk through directories and pick samples.
    """
    all_data = []
    
    print(f"Scanning {base_dir}...")
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        class_label = os.path.basename(root)
        
        # Check if folder name is a number (event code)
        if not class_label.isdigit():
            continue
            
        csv_files = [f for f in files if f.endswith('.parquet')]
        print(f"Found {len(csv_files)} files for Class {class_label}")
        
        # Take a sample to keep training fast/lightweight
        sampled_files = csv_files[:sample_size_per_class]
        
        for file in sampled_files:
            file_path = os.path.join(root, file)
            try:
                df = pd.read_parquet(file_path)
                
                # Ensure columns exist
                if not all(col in df.columns for col in REQUIRED_COLS):
                    continue
                
                # Fill NaNs (FFill then Fill with 0)
                df = df.ffill().fillna(0)
                
                # --- Feature Engineering: Time to Failure (TTF) ---
                # Find the first index where class != 0 (if any)
                # Assuming class 0 is normal.
                
                # Convert 'class' to numeric if it's not
                df['class'] = pd.to_numeric(df['class'], errors='coerce').fillna(0)
                
                failure_idx = df[df['class'] != 0].index.min()
                
                if pd.isna(failure_idx):
                    # No failure in this file (Normal run)
                    # TTF is chosen as a large number or standardized max
                    df['TTF'] = 360000 # Large value (e.g. 100 hours in seconds)
                else:
                    # Calculate seconds to failure
                    ttf_series = (failure_idx - df.index)
                    # Convert to seconds (TimedeltaIndex has .total_seconds())
                    df['TTF'] = ttf_series.total_seconds().values.clip(min=0) / 60 # In minutes
                
                all_data.append(df[REQUIRED_COLS + [TARGET_COL, 'TTF']])
                
            except Exception as e:
                print(f"Error reading {file}: {e}")
                
    if not all_data:
        raise ValueError("No data loaded! Check directory structure.")
        
    return pd.concat(all_data, ignore_index=True)

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Loading data...")
    # Adjust path if needed
    if not os.path.exists(DATA_DIR):
        print(f"Waiting for data at: {DATA_DIR}")
        return

    # REDUCED SAMPLE SIZE FOR SPEED (User request demo)
    df = load_and_process_data(DATA_DIR, sample_size_per_class=3)
    print(f"Data Loaded: {df.shape}")
    
    X = df[REQUIRED_COLS]
    y_class = df[TARGET_COL]
    y_ttf = df['TTF']
    
    # 1. Classification Model (Normal vs Fault)
    print("Training Classifier...")
    clf = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X, y_class)
    
    # 2. Regression Model (Time to Failure)
    print("Training Regressor (TTF)...")
    reg = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1)
    reg.fit(X, y_ttf)
    
    # Save
    joblib.dump(clf, os.path.join(MODEL_DIR, '3w_classifier.pkl'))
    joblib.dump(reg, os.path.join(MODEL_DIR, '3w_ttf.pkl'))
    
    print("Models saved successfully.")

if __name__ == "__main__":
    train()
