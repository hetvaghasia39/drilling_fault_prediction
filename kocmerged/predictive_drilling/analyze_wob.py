import pandas as pd
import numpy as np

def analyze(path, name):
    print(f"--- {name} ---")
    try:
        df = pd.read_csv(path)
        if 'WOB' in df.columns:
            print(df['WOB'].describe())
            print("95th percentile:", df['WOB'].quantile(0.95))
        else:
            print("WOB column not found")
            
        if 'ROP_AVG' in df.columns:
            print("\nROP Stats:")
            print(df['ROP_AVG'].describe())
            print("5th percentile:", df['ROP_AVG'].quantile(0.05))
    except Exception as e:
        print(e)
    print("\n")

analyze('data/drilling_log.csv', 'PrediDrill Data')
analyze('3w/data/processed/drilling_log_processed.csv', '3w Data')
