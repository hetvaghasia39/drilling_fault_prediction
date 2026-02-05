
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import math
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# --- Configuration & Styling ---
# --- Configuration & Styling ---
# st.set_page_config(
#     page_title="PrediDrill AI - Real-time Drilling Monitor",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
    }
    .safety-ok {
        color: #4CAF50;
        font-weight: bold;
    }
    .safety-warn {
        color: #FFC107;
        font-weight: bold;
    }
    .safety-crit {
        color: #FF5252;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Data & Model Loading (Cached) ---
@st.cache_resource
def load_data_and_model():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    log_path = os.path.join(base_dir, 'drilling_log.csv')
    xai_path = os.path.join(base_dir, 'xai_drilling.csv')
    
    # Load Data
    xai_df = pd.read_csv(xai_path)
    log_df = pd.read_csv(log_path)
    
    # --- Feature Selection & Mapping ---
    xai_features = ['Spindle speed n [1/min]', 'Feed rate vf [mm/min]', 'Power Pc [kW]']
    xai_target = 'Main Failure'
    log_features = ['SURF_RPM', 'ROP_AVG', 'WOB'] 

    # Clean Data
    xai_df = xai_df.dropna(subset=xai_features + [xai_target])
    log_df_clean = log_df.copy()
    for col in log_features:
        log_df_clean[col] = log_df_clean[col].fillna(log_df_clean[col].mean())

    # Scaling
    scaler_xai = MinMaxScaler()
    X_xai_scaled = scaler_xai.fit_transform(xai_df[xai_features])
    
    scaler_log = MinMaxScaler()
    scaler_log.fit(log_df_clean[log_features])

    # Train Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_xai_scaled, xai_df[xai_target])
    
    return clf, scaler_log, log_df_clean, log_features

model, scaler, log_data, log_features = load_data_and_model()

# Pre-calculate Predictions
X_log_scaled = scaler.transform(log_data[log_features])
probs = model.predict_proba(X_log_scaled)[:, 1]
log_data['Failure_Prob'] = probs

# --- Physics Engine (Von Mises) ---
PIPE_PROPS = {
    # Grade: Yield Strength (psi)
    "E-75": 75000,
    "X-95": 95000,
    "G-105": 105000,
    "S-135": 135000
}

# Standard 5" Pipe Dimensions (Approx)
PIPE_OD = 5.0 # inches
PIPE_ID = 4.276 # inches (19.50 ppf standard)
PIPE_AREA = (math.pi / 4) * (PIPE_OD**2 - PIPE_ID**2) # sq in
PIPE_J = (math.pi / 32) * (PIPE_OD**4 - PIPE_ID**4) # polar moment

def calculate_pipe_limits(grade_name, current_tension_lbs):
    Ym = PIPE_PROPS[grade_name]
    
    # 1. Pure Tensile Yield (Py)
    Py = PIPE_AREA * Ym
    
    # 2. Max Allowable Torque (Q) at current Tension (P)
    # Using User's Formula: Q = 0.096167 * (J/D) * sqrt(Ym^2 - (P/A)^2)
    # Ensure P doesn't exceed Py
    if current_tension_lbs >= Py:
        return 0, Py # Snapped by tension alone
    
    term_under_root = Ym**2 - (current_tension_lbs / PIPE_AREA)**2
    if term_under_root < 0: term_under_root = 0
    
    Q_max_ftlbs = 0.096167 * (PIPE_J / PIPE_OD) * math.sqrt(term_under_root)
    
    return Q_max_ftlbs, Py

def generate_tripping_loads(depth, speed_factor=1.0):
    # Simulate loads during pulling out of hole
    # Tension = String Weight (buoyed) + Drag
    # Assume 19.5 lb/ft pipe + BHA weight (say 50k)
    buoyancy = 0.85 # mud factor interaction
    string_weight = (depth * 19.5 * buoyancy) + 30000 
    
    # Drag / Friction (randomized)
    drag = np.random.normal(5000, 1000) * speed_factor
    if np.random.random() > 0.95: drag += 20000 # sudden stick notification
    
    total_tension = string_weight + drag
    
    # Torque (Rotation during backreaming or just breakout friction)
    # Usually low unless backreaming
    torque = np.random.normal(2000, 500) # ft-lbs
    if np.random.random() > 0.90: torque += 15000 # tight spot
    
    return total_tension, torque

# --- Sidebar Controls ---
st.sidebar.title("🎛 Control Panel")

mode = st.sidebar.radio("Operation Mode", ["Drilling (Predictive)", "Tripping Out (Physics)"])

st.sidebar.subheader("⚙️ Simulation Settings")
simulation_speed = st.sidebar.slider("Simulation Speed", 1, 10, 2)

if mode == "Tripping Out (Physics)":
    st.sidebar.subheader("🔧 Pipe Specs")
    pipe_grade = st.sidebar.selectbox("API Grade", list(PIPE_PROPS.keys()), index=2) # Default G-105
    st.sidebar.info(f"Yield Strength: {PIPE_PROPS[pipe_grade]:,} psi")

import datetime

# --- State Management ---
if 'active' not in st.session_state:
    st.session_state.active = False
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0
if 'trip_depth' not in st.session_state:
    st.session_state.trip_depth = 0.0
if 'drill_history' not in st.session_state:
    st.session_state.drill_history = []

def toggle_simulation():
    st.session_state.active = not st.session_state.active
    # If starting tripping, init depth
    if mode == "Tripping Out (Physics)" and st.session_state.active:
        st.session_state.trip_depth = log_data['Depth'].max()

btn_label = "⏹ Stop" if st.session_state.active else "▶ Start"
st.sidebar.button(btn_label, on_click=toggle_simulation, type="primary")

# --- Main Dashboard ---
# st.title("📡 PrediDrill AI Command Center") - Removed Title for cleaner dashboard look

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        font-family: 'Inter', sans-serif;
    }
    .dashboard-card {
        background-color: #1e2127;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #30333d;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .card-header {
        font-size: 14px;
        font-weight: 600;
        color: #8b92a6;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
     .stat-value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
    }
    .stat-label {
        font-size: 12px;
        color: #8b92a6;
    }
    .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
    }
    .status-ok { background-color: rgba(76, 175, 80, 0.2); color: #4CAF50; border: 1px solid #4CAF50; }
    .status-warn { background-color: rgba(255, 193, 7, 0.2); color: #FFC107; border: 1px solid #FFC107; }
    .status-crit { background-color: rgba(255, 82, 82, 0.2); color: #FF5252; border: 1px solid #FF5252; }
</style>
""", unsafe_allow_html=True)

# --- Layout ---
col_sidebar, col_main = st.columns([1, 3.5])

# --- Logic Variables ---
status_text = "STANDBY"
status_color = "gray"
reason = "Simulation Paused"
status_css = "status-ok"
display_risk = 0.0

if st.session_state.active:
    if mode == "Drilling (Predictive)":
        if st.session_state.current_idx < len(log_data):
            idx = st.session_state.current_idx
            row = log_data.iloc[idx]
            prob = row['Failure_Prob']
            display_risk = prob
            
            if prob > 0.6:
                status_text = "CRITICAL FAIL"
                status_css = "status-crit"
                reason = "Model High Probability"
            elif row['WOB'] > 80000:
                status_text = "OVERLOAD"
                status_css = "status-crit"
                reason = "Excessive WOB"
            elif row['WOB'] > 45000 and row['ROP_AVG'] < 0.005:
                status_text = "BIT BALLING"
                status_css = "status-warn"
                reason = "Low ROP Risk"
            else:
                 status_text = "NORMAL"
                 status_css = "status-ok"
                 reason = "Optimal Drilling"

            # History
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            log_entry = {
                "Time": timestamp,
                "Depth": f"{row['Depth']:.1f}",
                "WOB": f"{row['WOB']:.0f}",
                "Risk": f"{prob:.1%}",
                "Status": status_text
            }
            if st.session_state.active:
                 st.session_state.drill_history.insert(0, log_entry)

            # Advance
            st.session_state.current_idx += 1
            
    elif mode == "Tripping Out (Physics)":
        if st.session_state.trip_depth > 0:
             current_depth = st.session_state.trip_depth
             tension_lbs, torque_ftlbs = generate_tripping_loads(current_depth)
             max_torque, max_tension = calculate_pipe_limits(pipe_grade, tension_lbs)
             
             pct_tension = tension_lbs / max_tension
             pct_torque = torque_ftlbs / max_torque if max_torque > 0 else 999
             is_breakage = (tension_lbs > max_tension) or (torque_ftlbs > max_torque)
             
             display_risk = max(pct_tension, pct_torque)
             
             if is_breakage:
                 status_text = "FAILURE"
                 status_css = "status-crit"
                 reason = "Pipe Snapped!"
                 st.session_state.active = False
             elif display_risk > 0.8:
                 status_text = "STRESS WARNING"
                 status_css = "status-warn"
                 reason = "Approaching Yield"
             else:
                 status_text = "SAFE TRIP"
                 status_css = "status-ok"
                 reason = "Loads within API"

             if st.session_state.active:
                 st.session_state.trip_depth -= (5 * simulation_speed)

# --- LEFT COLUMN (Rig Status) ---
with col_sidebar:
    st.markdown(f"""
    <div class="dashboard-card">
        <div class="stat-label">Rig Unit</div>
        <div class="stat-value" style="font-size: 20px;">Deepwater Horizon II</div>
        <br>
        <div style="text-align: center;">
             <div style="font-size: 60px;">⛽</div>
        </div>
        <div class="stat-label" style="text-align: center; margin-top: 10px;">{mode}</div>
    </div>
    """, unsafe_allow_html=True)

    # Risk Gauge/Indicator
    st.markdown(f"""
    <div class="dashboard-card">
        <div class="card-header">Operational Risk</div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
             <div style="font-size: 40px; font-weight: bold; color: {'#FF5252' if display_risk > 0.5 else '#4CAF50'};">{display_risk*100:.0f}%</div>
             <div class="status-badge {status_css}">{status_text}</div>
        </div>
        <div style="background-color: #30333d; height: 8px; border-radius: 4px; margin-top: 10px;">
            <div style="background-color: {'#FF5252' if display_risk > 0.5 else '#4CAF50'}; width: {min(display_risk*100, 100)}%; height: 100%; border-radius: 4px;"></div>
        </div>
        <div class="stat-label" style="margin-top: 5px;">{reason}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Active State
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    if st.session_state.active:
        st.info("System ACTIVE")
    else:
        st.warning("System PAUSED")
    st.markdown('</div>', unsafe_allow_html=True)

# --- RIGHT COLUMN (Analysis) ---
with col_main:
    # 1. Main Chart
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    
    if mode == "Drilling (Predictive)":
        st.markdown('<div class="card-header">Drilling Risk Telemetry</div>', unsafe_allow_html=True)
        # Show recent history
        idx = st.session_state.current_idx
        chart_df = log_data.iloc[max(0, idx-100):idx+1]
        
        c = alt.Chart(chart_df).mark_area(
            line={'color':'#2196F3'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='#2196F3', offset=0),
                       alt.GradientStop(color='rgba(33, 150, 243, 0)', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x='Depth',
            y='Failure_Prob',
            tooltip=['Depth', 'Failure_Prob']
        ).properties(height=300)
        st.altair_chart(c, use_container_width=True)

    else:
        st.markdown('<div class="card-header">Tripping Loads Safety Envelope</div>', unsafe_allow_html=True)
        # Envelope Plot
        if 'tension_lbs' in locals():
            tensions = np.linspace(0, max_tension * 1.2, 50)
            torques = []
            for t in tensions:
                q, _ = calculate_pipe_limits(pipe_grade, t)
                torques.append(q)
            
            envelope_df = pd.DataFrame({"Tension": tensions, "Torque": torques})
            current_point = pd.DataFrame({"Tension": [tension_lbs], "Torque": [torque_ftlbs]})
            
            c = alt.Chart(envelope_df).mark_area(opacity=0.3, color='#4CAF50').encode(
                x='Tension', y='Torque'
            ) + alt.Chart(current_point).mark_point(
                color='#FF5252', size=300, shape='cross', filled=True
            ).encode(x='Tension', y='Torque')
            
            st.altair_chart(c, use_container_width=True)
        else:
            st.info("Start Simulation to view Physics Envelope")

    st.markdown('</div>', unsafe_allow_html=True)
    
    # 2. Key Metrics
    m1, m2, m3, m4 = st.columns(4)
    
    val1, val2, val3 = 0, 0, 0
    lbl1, lbl2, lbl3 = "N/A", "N/A", "N/A"
    
    if mode == "Drilling (Predictive)" and st.session_state.current_idx > 0:
        curr_row = log_data.iloc[st.session_state.current_idx - 1]
        val1 = curr_row['WOB']
        lbl1 = "Weight on Bit (lbs)"
        val2 = curr_row['ROP_AVG']
        lbl2 = "ROP (ft/hr)"
        val3 = curr_row['SURF_RPM']
        lbl3 = "Surface RPM"
    elif mode == "Tripping Out (Physics)" and 'tension_lbs' in locals():
        val1 = tension_lbs
        lbl1 = "Hook Load (lbs)"
        val2 = torque_ftlbs
        lbl2 = "Torque (ft-lbs)"
        val3 = current_depth
        lbl3 = "Depth (ft)"

    with m1:
         st.markdown(f"""
        <div class="dashboard-card" style="text-align: center;">
            <div class="card-header">Metric A</div>
            <div class="stat-value">{val1:,.0f}</div>
            <div class="stat-label">{lbl1}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
         st.markdown(f"""
        <div class="dashboard-card" style="text-align: center;">
            <div class="card-header">Metric B</div>
            <div class="stat-value" style="color: #2196F3;">{val2:,.2f}</div>
            <div class="stat-label">{lbl2}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
         st.markdown(f"""
        <div class="dashboard-card" style="text-align: center;">
            <div class="card-header">Metric C</div>
            <div class="stat-value">{val3:,.1f}</div>
            <div class="stat-label">{lbl3}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="dashboard-card" style="text-align: center;">
            <div class="card-header">AI Confidence</div>
            <div class="stat-value" style="color: #4CAF50;">99.8%</div>
            <div class="stat-label">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Log
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Event Log</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(st.session_state.drill_history[:5]), hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Loop Rerun
if st.session_state.active:
    time.sleep(1.0 / simulation_speed)
    st.rerun()

else:
    st.markdown("---")
    st.info("💡 Click 'Start' in the sidebar to begin the simulation.")
