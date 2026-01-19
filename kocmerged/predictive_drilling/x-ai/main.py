import logging
import sys
import os

# Add src to python path if needed (though running as module is better)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train import train_model
from src.evaluate import evaluate_model
from src.eda import perform_eda

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting Drilling Fault Prediction Pipeline...")
    
    # 1. EDA
    logging.info("Step 1: Exploratory Data Analysis")
    perform_eda()
    
    # 2. Train
    logging.info("Step 2: Model Training")
    train_model()
    
    # 3. Evaluate
    logging.info("Step 3: Evaluation")
    evaluate_model()
    
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
