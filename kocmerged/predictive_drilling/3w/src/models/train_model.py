import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def train_model():
    """ Train a model to detect anomalies in drilling data. """
    project_dir = Path(__file__).resolve().parents[2]
    input_file = project_dir.joinpath('data', 'processed', 'drilling_log_processed.csv')
    model_path = project_dir.joinpath('models', 'isolation_forest_model.pkl')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(input_file)
    
    # Feature Selection
    features = ['WOB', 'SURF_RPM', 'ROP_AVG', 'PHIF', 'VSH', 'SW', 'KLOGH']
    X = df[features]
    
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Unsupervised Anomaly Detection
    # Contamination: Estimate of the proportion of outliers in the data set.
    # Assuming the "event" at the end is the failure/anomaly.
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X_scaled)
    
    predictions = clf.predict(X_scaled)
    # -1 for outliers, 1 for inliers
    
    df['anomaly_score'] = clf.decision_function(X_scaled)
    df['anomaly_pred'] = predictions
    
    # Save predictions to analyze
    output_file = project_dir.joinpath('reports', 'anomaly_predictions.csv')
    df.to_csv(output_file, index=False)
    
    logger.info(f"Model trained. Anomaly counts: {df['anomaly_pred'].value_counts()}")
    logger.info(f"Predictions saved to {output_file}")
    
    # Save Model
    joblib.dump(clf, model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train_model()
