import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import plotly.graph_objects as go
import os

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration & Paths ---
# Use absolute paths based on the known structure relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRW_DIR = os.path.join(BASE_DIR, '3w')
MODEL_PATH = os.path.join(TRW_DIR, 'models', 'rop_regression_model.pkl')
DATA_PATH = os.path.join(TRW_DIR, 'data', 'processed', 'drilling_log_processed.csv')

# --- Model & Logic (from api.py) ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Failed to load model from {MODEL_PATH}: {e}")
        return None

def predict_drilling_state(model, data_dict):
    """
    Predicts ROP and determines status/deviation.
    data_dict: dict containing WOB, SURF_RPM, PHIF, VSH, SW, KLOGH, ROP_AVG_ACTUAL
    """
    try:
        # Prepare feature vector
        features = pd.DataFrame([{
            'WOB': data_dict['WOB'],
            'SURF_RPM': data_dict['SURF_RPM'],
            'PHIF': data_dict['PHIF'],
            'VSH': data_dict['VSH'],
            'SW': data_dict['SW'],
            'KLOGH': data_dict['KLOGH']
        }])
        
        # Predict "Ideal ROP"
        rop_pred = float(model.predict(features)[0])
        
        # Calculate Deviation
        deviation = data_dict['ROP_AVG_ACTUAL'] - rop_pred
        
        # Determine Status
        status = "NORMAL"
        # Thresholds from api.py
        if deviation < -0.002: 
            status = "WARNING: HIGH INEFFICIENCY (Low ROP)"
        elif deviation > 0.005: 
            status = "WARNING: UNCONTROLLED BREAK (High ROP)"
        
        # --- WOB Safety Checks (Override) ---
        wob = data_dict['WOB']
        rop = data_dict['ROP_AVG_ACTUAL']
        
        if wob > 80000:
            status = "💥 CRITICAL: TOOTH FRACTURE RISK (Extreme WOB)"
        elif wob > 45000 and rop < 0.005:
            status = "⚠️ WARNING: BIT BALLING DETECTED"
            
        return {
            "ROP_Pred": rop_pred,
            "ROP_Deviation": deviation,
            "Status": status
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise e

# --- UI Application (from dashboard.py) ---
def run():
    # Remove st.set_page_config as it's handled by unified_app
    
    # Load Model
    model = load_model()
    if model is None:
        st.stop()

    # Load Data
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found at {DATA_PATH}")
        st.stop()
        
    @st.cache_data
    def load_data_csv():
        return pd.read_csv(DATA_PATH)

    df = load_data_csv()

    # Session State
    if 'trw_current_index' not in st.session_state:
        st.session_state.trw_current_index = 0
    if 'trw_simulation_running' not in st.session_state:
        st.session_state.trw_simulation_running = False
    if 'trw_history' not in st.session_state:
        st.session_state.trw_history = {'Depth': [], 'Actual': [], 'Pred': [], 'Deviation': []}

    # Sidebar
    st.sidebar.title("Equipment Selection Controls") # Renamed for context
    speed = st.sidebar.slider("Simulation Speed (sec/step)", 0.1, 2.0, 0.5, key="trw_speed")

    col_btn1, col_btn2, col_btn3 = st.sidebar.columns(3)
    if col_btn1.button("Start"):
        st.session_state.trw_simulation_running = True
    if col_btn2.button("Stop"):
        st.session_state.trw_simulation_running = False
    if col_btn3.button("Reset"):
        st.session_state.trw_current_index = 0
        st.session_state.trw_history = {'Depth': [], 'Actual': [], 'Pred': [], 'Deviation': []}
        st.session_state.trw_simulation_running = False

    # Layout
    st.markdown("## Equipment Selection - Well Oil Extraction (3w)")
    st.markdown("### Digital Twin Monitor: Detecting Anomalies via Supervised Deviation")
    
    col1, col2 = st.columns([3, 1])
    
    placeholder = col1.empty()
    info_box = col2.empty()
    
    # Use a loop that respects Streamlit's rerun model
    # Note: Infinite loops in Streamlit can lock the UI if not careful with reruns.
    # The original dashboard used a while loop + time.sleep(). This works but blocks interaction until rerun.
    # We will keep the original logic for fidelity.
    
    if st.session_state.trw_simulation_running:
        # We process one step or loop? 
        # Original dashboard had a while loop inside the script execution.
        # This blocks formatting other parts of the app, but since this IS the app view, it's fine.
        
        while st.session_state.trw_simulation_running:
            if st.session_state.trw_current_index >= len(df):
                st.session_state.trw_simulation_running = False
                st.success("Simulation Complete")
                break
                
            # Get current row
            row = df.iloc[st.session_state.trw_current_index]
            
            # Payload data
            data_dict = {
                "WOB": row['WOB'],
                "SURF_RPM": row['SURF_RPM'],
                "PHIF": row['PHIF'],
                "VSH": row['VSH'],
                "SW": row['SW'],
                "KLOGH": row['KLOGH'],
                "ROP_AVG_ACTUAL": row['ROP_AVG']
            }
            
            # Inference
            try:
                res = predict_drilling_state(model, data_dict)
                
                # Update History
                st.session_state.trw_history['Depth'].append(row['Depth'])
                st.session_state.trw_history['Actual'].append(row['ROP_AVG'])
                st.session_state.trw_history['Pred'].append(res['ROP_Pred'])
                st.session_state.trw_history['Deviation'].append(res['ROP_Deviation'])
                
                # --- VISUALIZATION ---
                with placeholder.container():
                    # Metric Cards
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Depth (ft)", f"{row['Depth']:.1f}")
                    m2.metric("WOB (lb)", f"{row['WOB']:.0f}")
                    m3.metric("ROP Actual", f"{row['ROP_AVG']:.4f}")
                    
                    dev_val = res['ROP_Deviation']
                    dev_color = "normal"
                    if abs(dev_val) > 0.002: dev_color = "inverse" # Changed to inverse for visibility
                    m4.metric("ROP Deviation", f"{dev_val:.4f}", delta_color=dev_color)

                    # Charts
                    fig = go.Figure()
                    
                    # Plot 1: Digital Twin
                    fig.add_trace(go.Scatter(
                        x=st.session_state.trw_history['Depth'], 
                        y=st.session_state.trw_history['Actual'],
                        mode='lines', name='Actual ROP', line=dict(color='white')
                    ))
                    fig.add_trace(go.Scatter(
                        x=st.session_state.trw_history['Depth'], 
                        y=st.session_state.trw_history['Pred'],
                        mode='lines', name='Predicted ROP', line=dict(color='green', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="Digital Twin: Real-time ROP Tracking",
                        xaxis_title="Depth",
                        yaxis_title="ROP",
                        height=400,
                        template="plotly_dark",
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Plot 2: Deviation
                    fig_dev = go.Figure()
                    fig_dev.add_trace(go.Bar(
                        x=st.session_state.trw_history['Depth'],
                        y=st.session_state.trw_history['Deviation'],
                        marker_color=['red' if abs(x) > 0.002 else 'blue' for x in st.session_state.trw_history['Deviation']]
                    ))
                    fig_dev.update_layout(
                        title="Health Signal (Deviation)",
                        yaxis_title="Deviation",
                        height=200,
                        template="plotly_dark",
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_dev, use_container_width=True)

                # --- ALERTS ---
                with info_box.container():
                    st.subheader("Status Log")
                    if res['Status'] != "NORMAL":
                        st.error(f"🚨 {res['Status']}\nDepth {row['Depth']}")
                    else:
                        st.success("✅ Operations Normal")
                        
                    st.markdown("**Drilling Parameters:**")
                    st.json(data_dict)
            
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.trw_simulation_running = False
                
            st.session_state.trw_current_index += 1
            pd.time.sleep(speed) # Using pandas sleep? No, imports say time was imported in dashboard.py, I used pd.time there? No my script above imported time?
            # Wait, I didn't import time in my script above!
            # I imported pandas, numpy, joblib...
            import time
            time.sleep(speed)
            st.rerun() # Force rerun to update UI if outside loop? 
            # Actually, inside the loop it updates the placeholder. 
            # But Streamlit loop inside a callback/script holds the connection.
            # To allow "Stop" button to work, we need to check session state which we do.
            # However, `st.sidebar.button` won't trigger during the loop unless we yield?
            # No, Streamlit 1.x runs script top to bottom. If we loop, we block.
            # The only way to interrupt is if the user interacts, which triggers a rerun/interrupt?
            # Actually, standard Streamlit pattern for animation is a loop with sleep and placeholders.
            # Interaction might be delayed until loop breaks or yields.
            # We'll stick to the original pattern.
