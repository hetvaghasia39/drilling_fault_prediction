import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging
import os
import datetime
import time
import glob
import plotly.graph_objects as go

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration & Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '3w', 'models')
DATA_ROOT = os.path.join(BASE_DIR, '3w', 'data', 'raw', '3w_official', '2.0.0')

# Labales Mapping
LABELS = {
    0: "Normal Operation",
    1: "Abrupt Increase of BSW",
    2: "Spurious Closure of DHSV",
    3: "Severe Slugging",
    4: "Flow Instability",
    5: "Rapid Productivity Loss",
    6: "Quick Restriction in PCK",
    7: "Scaling in PCK",
    8: "Hydrate in Production Line"
}

# --- Model Loading ---
@st.cache_resource
def load_models():
    try:
        clf = joblib.load(os.path.join(MODELS_DIR, '3w_classifier.pkl'))
        reg = joblib.load(os.path.join(MODELS_DIR, '3w_ttf.pkl'))
        return clf, reg
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return None, None

# @st.cache_data - Removed to allow random file selection and fix UI threading text
def load_scenario_file(label_code):
    """Loads a random parquet file for the given class label."""
    search_path = os.path.join(DATA_ROOT, str(label_code), "*.parquet")
    files = glob.glob(search_path)
    
    if not files:
        st.error(f"No files found for Class {label_code} in {search_path}")
        return None
        
    # Pick a random file
    import random
    file_path = random.choice(files)
    st.toast(f"Loaded: {os.path.basename(file_path)}")
    
    try:
        df = pd.read_parquet(file_path)
        # Ensure correct columns order
        req_cols = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']
        # Also need class if we want to cheat/verify? But let's use model.
        return df.fillna(0)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# --- UI Application ---
def run():
    st.markdown("## Equipment Selection - 3W Dataset (Petrobras)")
    st.markdown("### 🛠️ Predictive Maintenance & Fault Detection")

    clf, reg = load_models()
    
    if clf is None:
        st.warning("⚠️ Models are training or missing. Please wait for training to complete.")
        return

    # Session State
    if 'trw_idx' not in st.session_state:
        st.session_state.trw_idx = 0
    if 'trw_active' not in st.session_state:
        st.session_state.trw_active = False
    if 'trw_log' not in st.session_state:
        st.session_state.trw_log = []
    if 'trw_data' not in st.session_state:
        st.session_state.trw_data = None
    if 'trw_chart_data' not in st.session_state:
        st.session_state.trw_chart_data = {'Time': [], 'P-PDG': [], 'P-TPT': []}
    if 'trw_total_steps' not in st.session_state:
        st.session_state.trw_total_steps = 0
    if 'trw_current_phase' not in st.session_state:
        st.session_state.trw_current_phase = "Initializing"
    if 'trw_time_offset' not in st.session_state:
        st.session_state.trw_time_offset = None

    def load_next_phase():
        """Auto-selects the next drilling phase (70% Normal, 30% Fault)."""
        import random
        if random.random() > 0.3:
            code = 0 # Normal
            name = "Normal Operation"
        else:
            code = random.choice([1, 2, 3, 4, 5, 6, 7, 8])
            name = LABELS[code]
        
        st.session_state.trw_data = load_scenario_file(code)
        st.session_state.trw_idx = 0
        st.session_state.trw_current_phase = name
        
        # Ensure data loaded
        if st.session_state.trw_data is None:
            # Retry once if failed
            st.session_state.trw_data = load_scenario_file(0) 

        # Calculate Time Offset (Shift file start to NOW)
        if st.session_state.trw_data is not None:
            file_start = st.session_state.trw_data.index[0]
            st.session_state.trw_time_offset = datetime.datetime.now() - file_start

    # Sidebar
    st.sidebar.title("Simulation Controls")
    speed = st.sidebar.slider("Speed", 0.1, 2.0, 0.2) 
    
    col_b1, col_b2 = st.sidebar.columns(2)
    if col_b1.button("Start / Reset"):
        load_next_phase()
        st.session_state.trw_active = True
        st.session_state.trw_log = []
        st.session_state.trw_chart_data = {'Time': [], 'P-PDG': [], 'P-TPT': []}
        st.session_state.trw_total_steps = 0
        
    if col_b2.button("Stop"):
        st.session_state.trw_active = False

    # Main Layout
    st.info(f"📍 Current Phase: **{st.session_state.trw_current_phase}**")
    
    ph_metrics = st.empty()
    ph_chart = st.empty()
    st.markdown("### 📝 Live Diagnostic Log")
    ph_table = st.empty()

    if st.session_state.trw_active:
        # Load data if missing
        if st.session_state.trw_data is None:
             load_next_phase()
             
        while st.session_state.trw_active:
            df = st.session_state.trw_data
            
            # Check for End of File -> Transition
            if st.session_state.trw_idx >= len(df):
                st.toast("Phase Complete. Transitioning...", icon="🔄")
                load_next_phase()
                continue # Loop back to start with new data
                
            row = df.iloc[st.session_state.trw_idx]
            
            # Timestamp Calculation
            current_timestamp = datetime.datetime.now().strftime("%H:%M:%S") # Fallback
            if st.session_state.trw_time_offset:
                # Use index + offset
                row_time = df.index[st.session_state.trw_idx]
                sim_time = row_time + st.session_state.trw_time_offset
                current_timestamp = sim_time.strftime("%H:%M:%S")
            
            # Predict
            features = row[['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']].values.reshape(1, -1)
            pred_class = clf.predict(features)[0]
            pred_ttf = reg.predict(features)[0]
            
            # Status Logic
            status = "NORMAL"
            reason = "-"
            if pred_class != 0:
                class_label = LABELS.get(pred_class, "Unknown Fault")
                # However, our binary classifier might just predict 0 or 1?
                # Wait, train_3w.py used y_class which has 0,1,2... so it is MULTI-CLASS.
                # So pred_class is the specific fault code.
                status = "CRITICAL: FAULT DETECTED"
                reason = LABELS.get(pred_class, f"Fault Code {pred_class}")
            elif pred_ttf < 60: # Less than 60 mins predicted
                status = "WARNING: FAILURE IMMINENT"
                reason = f"TTF < {pred_ttf:.0f} min"
                
            # Update Metrics
            with ph_metrics.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Downhole Pressure (P-PDG)", f"{row['P-PDG']:.1f}")
                m2.metric("Temp (T-TPT)", f"{row['T-TPT']:.1f}")
                
                # TTF Color
                ttf_str = f"{pred_ttf:.0f} min" if pred_ttf < 10000 else "> 100 hrs"
                m3.metric("Time to Failure", ttf_str, delta_color="inverse" if pred_ttf < 60 else "normal")
                
                m4.metric("System Status", "FAULT" if pred_class != 0 else "OK")

            # Update Chart Data (Keep last 100 points)
            st.session_state.trw_total_steps += 1
            st.session_state.trw_chart_data['Time'].append(st.session_state.trw_total_steps)
            st.session_state.trw_chart_data['P-PDG'].append(row['P-PDG'])
            st.session_state.trw_chart_data['P-TPT'].append(row['P-TPT'])
            
            if len(st.session_state.trw_chart_data['Time']) > 100:
                for k in st.session_state.trw_chart_data:
                    st.session_state.trw_chart_data[k].pop(0)

            # Draw Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=st.session_state.trw_chart_data['P-PDG'], name='P-PDG'))
            fig.add_trace(go.Scatter(y=st.session_state.trw_chart_data['P-TPT'], name='P-TPT'))
            fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark")
            ph_chart.plotly_chart(fig, use_container_width=True)

            # Update Log
            # timestamp = datetime.datetime.now().strftime("%H:%M:%S") - REPLACED by realistic calculation above
            log_entry = {
                "Timestamp": current_timestamp,
                "P-PDG": f"{row['P-PDG']:.1f}",
                "T-TPT": f"{row['T-TPT']:.1f}",
                "Status": status,
                "Reason": reason,
                "TTF (min)": f"{pred_ttf:.1f}"
            }
            st.session_state.trw_log.insert(0, log_entry)
            ph_table.dataframe(pd.DataFrame(st.session_state.trw_log), height=250, use_container_width=True)

            st.session_state.trw_idx += 1
            time.sleep(1.0/speed)
