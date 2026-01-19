from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize API
app = FastAPI(title="Drilling Failure Prediction API", version="1.0")

# Load Model
PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIR.joinpath('models', 'rop_regression_model.pkl')
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError("Model loading failed.")

# Define Request Schema
class DrillingData(BaseModel):
    WOB: float
    SURF_RPM: float
    PHIF: float
    VSH: float
    SW: float
    KLOGH: float
    ROP_AVG_ACTUAL: float # For deviation calculation

class PredictionResponse(BaseModel):
    ROP_Pred: float
    ROP_Deviation: float
    Status: str

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Drilling Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_drilling_state(data: DrillingData):
    try:
        # Prepare feature vector
        features = pd.DataFrame([{
            'WOB': data.WOB,
            'SURF_RPM': data.SURF_RPM,
            'PHIF': data.PHIF,
            'VSH': data.VSH,
            'SW': data.SW,
            'KLOGH': data.KLOGH
        }])
        
        # Predict "Ideal ROP"
        rop_pred = float(model.predict(features)[0])
        
        # Calculate Deviation
        # Deviation = Actual - Predicted
        # Negative Deviation = Inefficiency/Failure (Actual < Ideal)
        # Positive Deviation = Drilling Break (Actual > Ideal)
        deviation = data.ROP_AVG_ACTUAL - rop_pred
        
        # Determine Status
        status = "NORMAL"
        threshold = 0.5 # Based on previous EDA, small dataset values are small 0.00x. Wait, check scaling..
        # Inspecting CSV: ROP is approx 0.008. So threshold should be much smaller.
        # Let's recalibrate threshold based on typical values. 
        # Mean ROP is ~0.008. 
        # So a 30% deviation is ~0.002.
        
        if deviation < -0.002: 
            status = "WARNING: HIGH INEFFICIENCY (Low ROP)"
        elif deviation > 0.005: # Big spike
            status = "WARNING: UNCONTROLLED BREAK (High ROP)"
            
        return {
            "ROP_Pred": rop_pred,
            "ROP_Deviation": deviation,
            "Status": status
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
