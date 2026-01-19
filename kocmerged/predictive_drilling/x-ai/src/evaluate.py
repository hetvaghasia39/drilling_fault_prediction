import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import logging
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from preprocessing import load_and_preprocess_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model():
    """Evaluates the trained model and performs SHAP analysis."""
    logging.info("Starting evaluation...")
    
    # Load Model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')
    try:
        pipeline = joblib.load(model_path)
    except FileNotFoundError:
        logging.error("Model file not found. Please train the model first.")
        return

    # Load Data
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    
    # Predictions
    logging.info("Calculating metrics...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Metrics
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc:.4f}")
    
    # Plot Confusion Matrix
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    # --- SHAP Analysis ---
    logging.info("Starting SHAP analysis...")
    
    # Extract components
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    # Transform data
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names
    try:
        feature_names = preprocessor.get_feature_names_out()
        # Clean names for better plotting (remove 'num__', 'cat__' prefixes if present)
        feature_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]
    except AttributeError:
        logging.warning("Could not retrieve feature names. Using indices.")
        feature_names = None
        
    # Convert transformed data to DataFrame for SHAP
    if feature_names is not None:
        X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)
    
    # Explainer
    # Using TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(classifier)
    
    # Shap values (on a sample if dataset is huge, but 4000 is likely okay)
    # Using a subset for speed if needed, but 4000 is manageable.
    logging.info("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_test_transformed)
    
    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"))
    plt.close()
    
    logging.info(f"Evaluation completed. Plots saved to {output_dir}")

if __name__ == "__main__":
    evaluate_model()
