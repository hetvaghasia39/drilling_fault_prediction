import streamlit as st
import pandas as pd
import requests
import time
import plotly.graph_objects as go
from pathlib import Path

# Config
st.set_page_config(layout="wide", page_title="Drilling Failure Predictor")
API_URL = "http://localhost:8000/predict"
DATA_PATH = Path(__file__).resolve().parents[2].joinpath('data', 'processed', 'drilling_log_processed.csv')

# Load Data for Simulation
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# Session State for Simulation
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'history' not in st.session_state:
    st.session_state.history = {'Depth': [], 'Actual': [], 'Pred': [], 'Deviation': []}

# Sidebar
st.sidebar.title("Simulation Controls")
speed = st.sidebar.slider("Simulation Speed (sec/step)", 0.1, 2.0, 0.5)

if st.sidebar.button("Start/Resume Simulation"):
    st.session_state.simulation_running = True

if st.sidebar.button("Stop Simulation"):
    st.session_state.simulation_running = False

if st.sidebar.button("Reset"):
    st.session_state.current_index = 0
    st.session_state.history = {'Depth': [], 'Actual': [], 'Pred': [], 'Deviation': []}
    st.session_state.simulation_running = False

# Layout
st.title("🛢️ Real-Time Drilling Failure Prediction Analysis")
st.markdown("### Digital Twin Monitor: Detecting Anomalies via Supervised Deviation")

col1, col2 = st.columns([3, 1])

# Simulation Loop
placeholder = col1.empty()
info_box = col2.empty()

while st.session_state.simulation_running:
    if st.session_state.current_index >= len(df):
        st.session_state.simulation_running = False
        break
        
    # Get current row
    row = df.iloc[st.session_state.current_index]
    
    # Payload
    payload = {
        "WOB": row['WOB'],
        "SURF_RPM": row['SURF_RPM'],
        "PHIF": row['PHIF'],
        "VSH": row['VSH'],
        "SW": row['SW'],
        "KLOGH": row['KLOGH'],
        "ROP_AVG_ACTUAL": row['ROP_AVG']
    }
    
    try:
        # Call API
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            res = response.json()
            
            # Update History
            st.session_state.history['Depth'].append(row['Depth'])
            st.session_state.history['Actual'].append(row['ROP_AVG'])
            st.session_state.history['Pred'].append(res['ROP_Pred'])
            st.session_state.history['Deviation'].append(res['ROP_Deviation'])
            
            # --- VISUALIZATION ---
            with placeholder.container():
                # Metric Cards
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Depth (ft)", f"{row['Depth']:.1f}")
                m2.metric("WOB (lb)", f"{row['WOB']:.0f}")
                m3.metric("ROP Actual", f"{row['ROP_AVG']:.4f}")
                
                # Dynamic Color for Devaiation
                dev_val = res['ROP_Deviation']
                dev_color = "normal"
                if abs(dev_val) > 0.002: dev_color = "off" # Streamlit metric delta color logic
                m4.metric("ROP Deviation", f"{dev_val:.4f}", delta_color=dev_color)

                # Charts
                fig = go.Figure()
                
                # Plot 1: Digital Twin
                fig.add_trace(go.Scatter(
                    x=st.session_state.history['Depth'], 
                    y=st.session_state.history['Actual'],
                    mode='lines', name='Actual ROP', line=dict(color='black')
                ))
                fig.add_trace(go.Scatter(
                    x=st.session_state.history['Depth'], 
                    y=st.session_state.history['Pred'],
                    mode='lines', name='Predicted ROP', line=dict(color='green', dash='dash')
                ))
                
                fig.update_layout(
                    title="Digital Twin: Real-time ROP Tracking",
                    xaxis_title="Depth",
                    yaxis_title="ROP",
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot 2: Deviation
                fig_dev = go.Figure()
                fig_dev.add_trace(go.Bar(
                    x=st.session_state.history['Depth'],
                    y=st.session_state.history['Deviation'],
                    marker_color=['red' if abs(x) > 0.002 else 'blue' for x in st.session_state.history['Deviation']]
                ))
                fig_dev.update_layout(
                    title="Health Signal (Deviation)",
                    yaxis_title="Deviation",
                    height=200,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_dev, use_container_width=True)

            # --- ALERTS ---
            with info_box.container():
                st.subheader("Status Log")
                if res['Status'] != "NORMAL":
                    st.error(f"🚨 {res['Status']} at Depth {row['Depth']}")
                else:
                    st.success("✅ Operations Normal")
                    
                st.markdown("**Drilling Parameters:**")
                st.json(payload)

    except Exception as e:
        st.error(f"API Error: {e}")
        st.stop()
        
    st.session_state.current_index += 1
    time.sleep(speed)
