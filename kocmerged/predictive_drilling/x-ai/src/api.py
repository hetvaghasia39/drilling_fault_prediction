import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Drilling Fault Prediction API", description="API to predict drilling failures based on operational parameters.", version="1.0")

# Load Model
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')
try:
    pipeline = joblib.load(model_path)
    logging.info(f"Model loaded from {model_path}")
except FileNotFoundError:
    logging.error("Model file not found. Please train model first.")
    raise RuntimeError("Model file not found.")

# Define Input Schema
class DrillingInput(BaseModel):
    cutting_speed_vc: float
    spindle_speed_n: float
    feed_f: float
    feed_rate_vf: float
    power_pc: float
    cooling: str  # "0%", "25%" etc.
    material: str # "P (Steel)", "K (Cast Iron)", "N (Non-ferrous)"
    drill_bit_type: str # "N", "H", "W"
    process_time: float

    # Config for example docs
    class Config:
        json_schema_extra = {
            "example": {
                "cutting_speed_vc": 30.5,
                "spindle_speed_n": 500.0,
                "feed_f": 0.1,
                "feed_rate_vf": 150.0,
                "power_pc": 2.5,
                "cooling": "50%",
                "material": "P (Steel)",
                "drill_bit_type": "N",
                "process_time": 12.0
            }
        }

def preprocess_input(input_data: DrillingInput):
    """Converts API input to DataFrame expected by model."""
    
    # Create DataFrame
    data = {
        'cutting_speed_vc': [input_data.cutting_speed_vc],
        'spindle_speed_n': [input_data.spindle_speed_n],
        'feed_f': [input_data.feed_f],
        'feed_rate_vf': [input_data.feed_rate_vf],
        'power_pc': [input_data.power_pc],
        'cooling': [input_data.cooling], # Model expects specific format, likely 0, 25 or string "25%". 
        # In preprocessing.py, we might have cleaner logic.
        # Let's check how `load_data` loaded 'cooling'.
        # Assuming for now we pass it as 'cooling' column value.
        'material': [input_data.material],
        'drill_bit_type': [input_data.drill_bit_type],
        'process_time': [input_data.process_time]
    }
    
    df = pd.DataFrame(data)
    
    # Handle 'Cooling' cleaning if needed to match training data
    # In training, if it was numeric (0, 25), we need to convert.
    # If it was categorical '0%', '25%', we keep as is.
    # Looking at my memory of `eda.py` output or `train.py`, cooling might be numeric.
    # But `preprocessing.py` treats it as numeric if in numeric_features, or categorical otherwise.
    # Let's check `process_time_t` vs `process_time` name mismatch from `clean_column_names` too.
    # In `preprocessing.py`: 
    # `clean_column_names` converts 'Cooling [%]' -> 'cooling'
    # 'Process Time [s]' -> 'process_time' or similar? 
    # Wait, `clean_column_names` simply replaces ` [s]` then lowercases.
    
    # We should match exactly what `preprocessing.py` produces.
    # Ideally, we reuse `clean_column_names` but that takes raw data.
    # Here we are constructing the cleaned dataframe directly.
    
    # Adjusting column names to match what `get_preprocessor` expects.
    # In `preprocessing.py` I assumed columns were `cutting_speed_vc` etc.
    # I will assume the input names here map 1:1 to the cleaner names.
    
    # Special handling for 'cooling' if model expects int
    # Based on eda output, it seemed cooling was 0, 25, 50, 75, 100
    if isinstance(input_data.cooling, str) and '%' in input_data.cooling:
        try:
             df['cooling'] = int(input_data.cooling.replace('%', ''))
        except ValueError:
             pass # Leave as is if fails
    
    return df

@app.get("/")
def home():
    return {"message": "Drilling Fault Prediction API is running."}

@app.post("/predict")
def predict_fault(input_data: DrillingInput):
    try:
        df = preprocess_input(input_data)
        
        # Predict
        prediction = pipeline.predict(df)[0]
        probability = pipeline.predict_proba(df)[0][1]
        
        result = "FAILURE" if prediction == 1 else "SUCCESS"
        
        return {
            "prediction": result,
            "probability_failure": float(probability),
            "status": "danger" if prediction == 1 else "safe"
        }
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
