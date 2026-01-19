import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from data_loader import load_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_column_names(df):
    """Cleans column names to snake_case."""
    # Mapping based on common names or just heuristic
    new_cols = []
    for col in df.columns:
        c = col.lower().strip()
        c = c.replace(' [m/min]', '').replace(' [1/min]', '').replace(' [mm/rev]', '').replace(' [mm/min]', '').replace(' [kw]', '').replace(' [%]', '').replace(' [s]', '').replace(' [sec]', '')
        c = c.replace(' ', '_').replace('-', '_')
        new_cols.append(c)
    df.columns = new_cols
    return df

def get_preprocessor():
    """Returns a ColumnTransformer for preprocessing."""
    
    # Define features
    numeric_features = ['cutting_speed_vc', 'spindle_speed_n', 'feed_f', 'feed_rate_vf', 'power_pc', 'process_time'] # Update names based on cleaning logic
    categorical_features = ['material', 'drill_bit_type']
    
    # Cooling might be numeric or categorical. User said 0, 25, 50..
    # If loaded as numbers, keep as numeric. If strings '25%', strip %.
    # We'll handle 'cooling' inside the load_and_preprocess to ensure it checks out.
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def load_and_preprocess_data(test_size=0.2, random_state=42):
    """
    Loads data, cleans it, and splits into train/test sets.
    Returns: X_train, X_test, y_train, y_test, preprocessor
    """
    logging.info("Loading and preprocessing data...")
    df = load_data()
    
    # Clean names
    df = clean_column_names(df)
    logging.info(f"Cleaned columns: {df.columns.tolist()}")

    # Handle Cooling: check if it needs parsing
    # Typically Kaggle datasets might have it as int directly if the column is 'Cooling [%]' or similar.
    # We'll treat 'cooling' as numeric (assuming 0, 25, 50, 75, 100).
    # If it's not in numeric features list, we should add it.
    if 'cooling' in df.columns:
        # Check if needs numeric conversion
        # The column clean logic produced 'cooling' from 'Cooling [%]'
        pass
    
    # Define features and target
    # Target: main_failure
    # Drop ID and subgroup failures
    drop_cols = ['id']
    subgroup_failures = ['bef', 'buef', 'ccf', 'fwf', 'wdf', 'build_up_edge_failure', 'compression_chips_failure', 'flank_wear_failure', 'wrong_drill_bit_failure'] # Add all possible variants
    
    # Identify actual columns to drop
    cols_to_drop = [c for c in drop_cols + subgroup_failures if c in df.columns]
    
    target_col = 'main_failure'
    if target_col not in df.columns:
        # Try to find it
        candidates = [c for c in df.columns if 'main_failure' in c or 'failure' in c and c not in cols_to_drop]
        if candidates:
            target_col = candidates[0]
        else:
            raise ValueError("Could not identify main target column.")

    logging.info(f"Target column: {target_col}")
    
    X = df.drop(columns=cols_to_drop + [target_col])
    y = df[target_col]
    
    # Update lists for preprocessor based on actual columns in X
    # Numeric: all except object/category
    # Categorical: object/category
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logging.info(f"Numeric features: {num_cols}")
    logging.info(f"Categorical features: {cat_cols}")
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()
    
    # Fit preprocessor to checking it works
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    print("Preprocessing successful.")
    print("Processed feature matrix shape:", X_train_processed.shape)
