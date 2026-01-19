import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import streamlit.components.v1 as components

# Helper for SHAP in Streamlit (Must be defined before use)
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Page Config
st.set_page_config(page_title="Drilling Fault Prediction", page_icon="⚙️", layout="wide")

# Load Model
@st.cache_resource
def load_pipeline():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')
    return joblib.load(model_path)

pipeline = load_pipeline()

# Title
st.title("⚙️ Drilling Fault Prediction Dashboard")
st.markdown("Enter the drilling parameters below to check for potential failures.")

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Drilling Parameters")
    
    # Input Form
    with st.form("input_form"):
        cutting_speed = st.slider("Cutting speed vc [m/min]", 0.0, 200.0, 30.0)
        spindle_speed = st.number_input("Spindle speed n [1/min]", 0.0, 10000.0, 500.0)
        feed = st.slider("Feed f [mm/rev]", 0.0, 2.0, 0.1)
        feed_rate = st.number_input("Feed rate vf [mm/min]", 0.0, 5000.0, 150.0)
        power = st.number_input("Power Pc [kW]", 0.0, 10.0, 2.5)
        
        cooling = st.selectbox("Cooling", ["0%", "25%", "50%", "75%", "100%"], index=2)
        material = st.selectbox("Material", ["P (Steel)", "K (Cast Iron)", "N (Non-ferrous)"])
        drill_bit = st.selectbox("Drill Bit Type", ["N", "H", "W"])
        
        process_time = st.number_input("Process time t [s]", 0.0, 100.0, 10.0)
        
        submit_button = st.form_submit_button("Predict Fault Risk")

if submit_button:
    # Prepare Data
    # Convert inputs to match training data expectation
    cooling_val = int(cooling.replace('%', '')) # Assuming model was trained on numeric cooling 0, 25...
    
    input_data = {
        'cutting_speed_vc': [cutting_speed],
        'spindle_speed_n': [spindle_speed],
        'feed_f': [feed],
        'feed_rate_vf': [feed_rate],
        'power_pc': [power],
        'cooling': [cooling_val], 
        'material': [material],
        'drill_bit_type': [drill_bit],
        'process_time': [process_time]
    }
    
    df = pd.DataFrame(input_data)
    
    with col2:
        st.header("Prediction Result")
        
        # Predict
        try:
            pred = pipeline.predict(df)[0]
            prob = pipeline.predict_proba(df)[0][1]
            
            # Display Status
            if pred == 1:
                st.error(f"FAILURE PREDICTED")
                st.metric("Failure Probability", f"{prob*100:.2f}%")
            else:
                st.success(f"OPERATION SAFE")
                st.metric("Failure Probability", f"{prob*100:.2f}%")
            
            # Gauge / Progress Bar
            st.progress(float(prob))
            
            # Explainability (SHAP)
            st.subheader("Why this prediction?")
            with st.spinner("Calculating explanation..."):
                # Extract model components
                preprocessor = pipeline.named_steps['preprocessor']
                classifier = pipeline.named_steps['classifier']
                
                # Transform input
                X_transformed = preprocessor.transform(df)
                
                # Feature Names
                try:
                    feature_names = preprocessor.get_feature_names_out()
                    feature_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]
                except:
                    feature_names = None
                
                # SHAP
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(X_transformed)
                
                # Force Plot
                st_shap(shap.force_plot(explainer.expected_value, shap_values[0], X_transformed, feature_names=feature_names))
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
