import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def train_regression_model():
    """ 
    Train a Regression model to predict 'Ideal ROP'.
    Any significant deviation from this prediction is a 'Failure Signal'.
    """
    project_dir = Path(__file__).resolve().parents[2]
    input_file = project_dir.joinpath('data', 'processed', 'drilling_log_processed.csv')
    model_path = project_dir.joinpath('models', 'rop_regression_model.pkl')
    
    df = pd.read_csv(input_file)
    
    # Feature Selection
    # We want to predict ROP based on inputs (WOB, RPM) and Rock Properties (Gamma, Res, Neutron)
    features = ['WOB', 'SURF_RPM', 'PHIF', 'VSH', 'SW', 'KLOGH']
    target = 'ROP_AVG'
    
    X = df[features]
    y = df[target]
    
    # Split
    # Important: In a real "Deviation" scenario, we might train on "Known Good" offset well data.
    # Here, we will train on a random split, assuming the majority of data is 'Normal'.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model
    # Random Forest is robust for non-linear drilling relationships
    reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    reg.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = reg.predict(X_train)
    y_pred_test = reg.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    logger.info(f"Model Performance -- MSE: {mse:.4f}, R2: {r2:.4f}")
    
    # Generate Predictions for the ENTIRE dataset to visualize deviation
    df['ROP_Pred'] = reg.predict(X)
    df['ROP_Deviation'] = df['ROP_AVG'] - df['ROP_Pred']
    
    # Save results
    output_file = project_dir.joinpath('reports', 'rop_deviation_results.csv')
    df.to_csv(output_file, index=False)
    logger.info(f"Deviation results saved to {output_file}")
    
    # Save Model
    joblib.dump(reg, model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train_regression_model()
