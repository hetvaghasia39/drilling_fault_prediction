
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
    base_dir = '/home/het/Downloads/koc/kocmerged/predictive_drilling/data'
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
st.title("📡 PrediDrill AI Command Center")

if mode == "Drilling (Predictive)":
    # Reuse previous logic
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Real-Time Drilling Metrics")
        kpi = st.columns(4)
        m_depth = kpi[0].empty()
        m_rpm = kpi[1].empty()
        m_rop = kpi[2].empty()
        m_prob = kpi[3].empty()
        
        chart_viz = st.empty()
        
    with col2:
        st.subheader("AI Agent Alerts")
        alert_box = st.empty()

    # Placeholders for History
    st.markdown("### 📝 Live Sensor & Fault Log")
    history_table = st.empty()

    if st.session_state.active:
        while st.session_state.current_idx < len(log_data) and st.session_state.active:
            idx = st.session_state.current_idx
            row = log_data.iloc[idx]
            
            m_depth.metric("Depth", f"{row['Depth']:.0f} ft")
            m_rpm.metric("RPM", f"{row['SURF_RPM']:.1f}")
            m_rop.metric("ROP", f"{row['ROP_AVG']:.3f} ft/hr")
            st.sidebar.metric("WOB (Weight on Bit)", f"{row['WOB']:.0f} lbs")
            
            prob = row['Failure_Prob']
            delta_color = "inverse" if prob > 0.5 else "normal"
            m_prob.metric("Failure Risk", f"{prob:.1%}", delta=None, delta_color=delta_color)
            
            # Alerts & Logging
            current_status = "NORMAL"
            failure_reason = "None"
            
            if prob > 0.6:
                alert_box.error(f"🚨 FAIL PREDICTED\nDepth: {row['Depth']}")
                current_status = "CRITICAL"
                failure_reason = "Model Prediction (High Prob)"
            elif row['WOB'] > 80000:
                 alert_box.error(f"💥 CRITICAL: TOOTH FRACTURE RISK\nWOB: {row['WOB']:.0f} lbs")
                 current_status = "CRITICAL"
                 failure_reason = "Excessive WOB (Fracture)"
            elif row['WOB'] > 45000 and row['ROP_AVG'] < 0.005:
                 alert_box.warning(f"⚠️ BIT BALLING DETECTED\nHigh WOB / Low ROP")
                 current_status = "WARNING"
                 failure_reason = "Bit Balling Risk"
            elif row['WOB'] > 65000:
                 alert_box.warning(f"⚠️ HIGH WOB: FRACTURE WARNING\nWOB: {row['WOB']:.0f} lbs")
                 current_status = "WARNING"
                 failure_reason = "High WOB Warning"
            else:
                alert_box.success("✅ System Normal")
            
            # Record History
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            log_entry = {
                "Timestamp": timestamp,
                "Depth (ft)": f"{row['Depth']:.1f}",
                "WOB (lbs)": f"{row['WOB']:.0f}",
                "RPM": f"{row['SURF_RPM']:.1f}",
                "Risk (%)": f"{prob:.1%}",
                "Status": current_status,
                "Reason": failure_reason
            }
            st.session_state.drill_history.insert(0, log_entry) # Prepend for latest top
            
            # Chart
            hist = log_data.iloc[max(0, idx-50):idx+1]
            chart_data = pd.DataFrame({
                'Depth': hist['Depth'],
                'Risk': hist['Failure_Prob']
            })
            chart_viz.line_chart(chart_data, x='Depth', y='Risk', height=300)
            
            # History Table
            history_table.dataframe(pd.DataFrame(st.session_state.drill_history), height=250, use_container_width=True)
            
            st.session_state.current_idx += 1
            time.sleep(1.0/simulation_speed)
            
elif mode == "Tripping Out (Physics)":
    # New Physics Logic
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Tripping Out Monitor - Grade {pipe_grade}") 
        
        # Envelope Chart Placeholder
        chart_ph = st.empty()
        
    with col2:
        st.subheader("Pipe Loads")
        m_tens = st.empty()
        m_torque = st.empty()
        m_depth_trip = st.empty()
        st.divider()
        m_limit_tens = st.empty()
        m_limit_torq = st.empty()
        alert_phys = st.empty()

    if st.session_state.active and st.session_state.trip_depth > 0:
        while st.session_state.trip_depth > 0 and st.session_state.active:
            # 1. Simulate Physics
            current_depth = st.session_state.trip_depth
            tension_lbs, torque_ftlbs = generate_tripping_loads(current_depth)
            
            # 2. Calculate Limits
            max_torque, max_tension = calculate_pipe_limits(pipe_grade, tension_lbs)
            
            # 3. Safety Check
            pct_tension = tension_lbs / max_tension
            pct_torque = torque_ftlbs / max_torque if max_torque > 0 else 999
            
            is_breakage = (tension_lbs > max_tension) or (torque_ftlbs > max_torque)
            
            # 4. Display
            m_depth_trip.metric("Current Depth", f"{current_depth:.0f} ft")
            m_tens.metric("Tension (Hook Load)", f"{tension_lbs/1000:.1f} kips")
            m_torque.metric("Torque", f"{torque_ftlbs:.0f} ft-lbs")
            
            m_limit_tens.markdown(f"**Max Tension:** {max_tension/1000:.1f} kips")
            m_limit_torq.markdown(f"**Max Torque:** {max_torque:.0f} ft-lbs")
            
            if is_breakage:
                alert_phys.error("💥 SYSTEM FAILURE DETECTED! PIPE BROKEN!")
                st.session_state.active = False
            elif pct_tension > 0.8 or pct_torque > 0.8:
                alert_phys.warning("⚠️ CRITICAL STRESS! REDUCE SPEED!")
            else:
                alert_phys.success("✅ Loads within Safety Envelope")
            
            # 5. Envelope Visualization
            # Draw the curve for current grade
            tensions = np.linspace(0, max_tension, 50)
            torques = []
            for t in tensions:
                q, _ = calculate_pipe_limits(pipe_grade, t)
                torques.append(q)
            
            envelope_df = pd.DataFrame({"Tension": tensions, "Torque": torques})
            current_point = pd.DataFrame({"Tension": [tension_lbs], "Torque": [torque_ftlbs]})
            
            c = alt.Chart(envelope_df).mark_area(opacity=0.3, color='green').encode(
                x='Tension', y='Torque'
            ) + alt.Chart(envelope_df).mark_line(color='green').encode(
                x='Tension', y='Torque'
            ) + alt.Chart(current_point).mark_point(
                color='red', size=200, shape='cross'
            ).encode(x='Tension', y='Torque')
            
            chart_ph.altair_chart(c, use_container_width=True)
            
            # Decrement Depth
            st.session_state.trip_depth -= (5 * simulation_speed) # Pulling speed
            time.sleep(0.5)

    if st.session_state.trip_depth <= 0:
         st.success("Tripping Complete. Pipe retrieved safely.")
