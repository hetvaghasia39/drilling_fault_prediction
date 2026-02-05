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
    # --- Custom CSS for Dashboard Look ---
    st.markdown("""
        <style>
        /* Main background */
        .stApp {
            background-color: #0e1117;
            font-family: 'Inter', sans-serif;
        }
        
        /* Card Style */
        .dashboard-card {
            background-color: #1e2127;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #30333d;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        /* Text Styles */
        .card-header {
            font-size: 14px;
            font-weight: 600;
            color: #8b92a6;
            margin-bottom: 10px;
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

        /* Profile Section */
        .profile-name {
            font-size: 24px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 5px;
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

    clf, reg = load_models()
    
    if clf is None:
        st.warning("⚠️ Models are training or missing. Please wait for training to complete.")
        return

    # Session State Init
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
             st.session_state.trw_data = load_scenario_file(0) 

    # --- SIMULATION LOGIC STEP ---
    # We run the logic *before* UI rendering to ensure immediate update
    row = None
    pred_class = 0
    pred_ttf = 9999
    
    if st.session_state.trw_active:
        if st.session_state.trw_data is None:
             load_next_phase()
             
        df = st.session_state.trw_data
        
        # Check for End of File
        if st.session_state.trw_idx >= len(df):
            st.toast("Phase Complete. Transitioning...", icon="🔄")
            load_next_phase()
            df = st.session_state.trw_data # Reload new data

        if df is not None:
            row = df.iloc[st.session_state.trw_idx]
            
            # Predict
            features = row[['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']].values.reshape(1, -1)
            pred_class = clf.predict(features)[0]
            pred_ttf = reg.predict(features)[0]
            
            # Update history
            st.session_state.trw_total_steps += 1
            st.session_state.trw_chart_data['Time'].append(st.session_state.trw_total_steps)
            st.session_state.trw_chart_data['P-PDG'].append(row['P-PDG'])
            st.session_state.trw_chart_data['P-TPT'].append(row['P-TPT'])
            
            # Trim history
            if len(st.session_state.trw_chart_data['Time']) > 100:
                for k in st.session_state.trw_chart_data:
                    st.session_state.trw_chart_data[k].pop(0)

            # Advance
            st.session_state.trw_idx += 1
            
            # Loop delay
            # time.sleep(1.0/speed) - HANDLED BY RERUN in UI

    # --- MAIN LAYOUT ---
    
    # Header
    st.markdown("### Equipment Analysis Dashboard")
    
    col_nav, col_main = st.columns([1, 3.5])
    
    # --- LEFT COLUMN (Profile & Status) ---
    with col_nav:
        # Profile Card
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="profile-name">Wellsite EQ-7</div>
            <div class="stat-label">Sector: 3W-Offshore</div>
            <br>
            <div style="text-align: center;">
                <div style="font-size: 60px;">🏗️</div>
            </div>
            <br>
            <div class="stat-label">Model Confidence</div>
            <div style="background-color: #30333d; height: 8px; border-radius: 4px; margin-top: 5px;">
                <div style="background-color: #4CAF50; width: 92%; height: 100%; border-radius: 4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Health Score
        health_color = "#4CAF50"
        health_score = 98
        status_text = "OPTIMAL"
        status_css = "status-ok"
        
        if row is not None:
            if pred_class != 0:
                health_score = 35
                status_text = "CRITICAL"
                health_color = "#FF5252" 
                status_css = "status-crit"
            elif pred_ttf < 60:
                # Warning
                health_score = 65
                status_text = "WARNING"
                health_color = "#FFC107"
                status_css = "status-warn"
        
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="card-header">System Health</div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                 <div style="font-size: 48px; font-weight: bold; color: {health_color};">{health_score}%</div>
                 <div class="status-badge {status_css}">{status_text}</div>
            </div>
            <br>
            <div class="stat-label">Pred. Time to Failure</div>
            <div class="stat-value" style="font-size: 20px;">{f'{pred_ttf:.0f} min' if pred_ttf < 10000 else '> 48h'}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Controls
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Controls</div>', unsafe_allow_html=True)
        
        col_c1, col_c2 = st.columns(2)
        if col_c1.button("▶ START", use_container_width=True, type="primary"):
            st.session_state.trw_active = True
            if st.session_state.trw_data is None: load_next_phase()
            st.rerun()
            
        if col_c2.button("⏹ STOP", use_container_width=True):
            st.session_state.trw_active = False
            st.rerun()
            
        speed = st.slider("Speed", 0.1, 5.0, 1.0)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- RIGHT COLUMN (Analysis) ---
    with col_main:
        
        # 1. Top Section: Charts
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Real-Time Pressure / Temp Analysis</div>', unsafe_allow_html=True)
        
        chart_ph = st.empty()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=st.session_state.trw_chart_data['P-PDG'], x=st.session_state.trw_chart_data['Time'], 
                                 name='Downhole Pres. (PDG)', line=dict(color='#4CAF50', width=2), fill='tozeroy', fillcolor='rgba(76, 175, 80, 0.1)'))
        fig.add_trace(go.Scatter(y=st.session_state.trw_chart_data['P-TPT'], x=st.session_state.trw_chart_data['Time'], 
                                 name='Temp Pres. (TPT)', line=dict(color='#2196F3', width=2)))
        
        fig.update_layout(
            height=350, 
            margin=dict(l=10,r=10,t=10,b=10), 
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#30333d')
        )
        chart_ph.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 2. Key Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        
        p_pdg = row['P-PDG'] if row is not None else 0
        t_tpt = row['T-TPT'] if row is not None else 0
        p_mon = row['P-MON-CKP'] if row is not None else 0
        t_jus = row['T-JUS-CKP'] if row is not None else 0
        
        with m1:
            st.markdown(f"""
            <div class="dashboard-card" style="text-align: center;">
                <div class="card-header">P-PDG</div>
                <div class="stat-value">{p_pdg:.1f}</div>
                <div class="stat-label">Pascal</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="dashboard-card" style="text-align: center;">
                <div class="card-header">T-TPT</div>
                <div class="stat-value" style="color: #2196F3;">{t_tpt:.1f}</div>
                <div class="stat-label">Celsius</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="dashboard-card" style="text-align: center;">
                <div class="card-header">Ck-Point P</div>
                <div class="stat-value">{p_mon:.1f}</div>
                <div class="stat-label">Pascal</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
             st.markdown(f"""
            <div class="dashboard-card" style="text-align: center;">
                <div class="card-header">Ck-Point T</div>
                <div class="stat-value">{t_jus:.1f}</div>
                 <div class="stat-label">Celsius</div>
            </div>
            """, unsafe_allow_html=True)
            
        # 3. Diagnostics / Logs
        st.markdown(f"""
        <div class="dashboard-card">
            <div style="display: flex; justify-content: space-between;">
                <div class="card-header">Diagnostic Log</div>
                <div class="stat-label">Current Phase: {st.session_state.trw_current_phase}</div>
            </div>
        """, unsafe_allow_html=True)
        
        if row is not None:
            # Add to local log for display
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            log_entry = {
                "Time": timestamp,
                "Phase": st.session_state.trw_current_phase,
                "Status": status_text,
                "Pressure": f"{p_pdg:.1f}"
            }
            # Only add if moving
            if st.session_state.trw_active:
                st.session_state.trw_log.insert(0, log_entry)
        
        st.dataframe(
            pd.DataFrame(st.session_state.trw_log[:5]), 
            hide_index=True, 
            use_container_width=True,
            column_config={
                "Status": st.column_config.TextColumn("Status", help="System Status"),
            }
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Rerun loop if active
    if st.session_state.trw_active:
        time.sleep(1.0 / speed)
        st.rerun()

