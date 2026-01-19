import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from preprocessing import load_and_preprocess_data, get_preprocessor
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():
    """Trains an XGBoost model and saves it."""
    logging.info("Starting model training...")
    
    # Load data
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()
    
    # Define Model
    # Using scale_pos_weight for imbalance (5% failure rate)
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    logging.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
    
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    # Create Pipeline (Preprocessor + Model)
    # Note: load_and_preprocess_data returns a preprocessor instance, but it's already fitted? 
    # Actually, in scikit-learn pipelines, we usually pass the transformer to the pipeline and let fit() handle it.
    # But preprocessing.py returns a fitted or unfitted one?
    # In my preprocessing.py, I return `preprocessor` which is a `ColumnTransformer`.
    # `load_and_preprocess_data` doesn't fit it. It just defines it.
    # Ah, wait. `load_and_preprocess_data` returns `preprocessor` which is just the object definition from `get_preprocessor`.
    # But I should verify `preprocessing.py`.
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    logging.info("Fitting model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    logging.info("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Save model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')
    joblib.dump(pipeline, model_path)
    logging.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
