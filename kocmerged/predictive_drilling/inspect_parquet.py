import pandas as pd
import sys

# Load the file passed as argument
file_path = sys.argv[1]
print(f"Inspecting: {file_path}")

df = pd.read_parquet(file_path)
print("\nIndex:")
print(df.index)
print("\nColumns:")
print(df.columns)
print("\nFirst 5 rows:")
print(df.head())
