from preprocessing import load_data, clean_column_names

df = load_data()
df = clean_column_names(df)
print("\n--- COLUMNS ---")
for c in df.columns:
    print(c)
