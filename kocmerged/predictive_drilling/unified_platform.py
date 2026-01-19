import streamlit as st
import importlib
import sys
import os
import runpy

# --- Page Configuration ---
st.set_page_config(
    page_title="KOC Unified Drilling Platform", 
    layout="wide", 
    page_icon="🛢️",
    initial_sidebar_state="expanded"
)

# --- Aesthetic Styling (Premium Dark Mode) ---
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background: radial-gradient(circle at center, #1a1c24 0%, #0e1117 100%);
        color: #f0f2f6;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161920;
        border-right: 1px solid #2d3035;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #4CAF50;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #2e3440;
        color: white;
        border: 1px solid #434c5e;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #3b4252;
        border-color: #88c0d0;
    }
</style>
""", unsafe_allow_html=True)

# --- Navigation ---
with st.sidebar:
    st.markdown("## 🧭 Navigator")
    selection = st.radio(
        "", 
        ["Home", "Equipment Selection - Well Oil (3w)", "Sea Oil Extraction (PrediDrill)"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Status")
    st.markdown("🟢 System Online")
    st.markdown("🔗 Nodes Connected: 2")

# --- Routing Logic ---
if selection == "Home":
    st.title("🛢️ KOC Unified Drilling Intelligence")
    st.markdown("### Advanced analytics for drilling operations")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: #21252b; padding: 20px; border-radius: 10px; border: 1px solid #333;">
            <h3 style="color: #61afef;">🛠️ Equipment Selection (3w)</h3>
            <p>Access the 3W dataset prediction engine. Monitor drilling logs and detect equipment failures.</p>
            <p style="color: #98c379;">Features: ROP Prediction, Deviation Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch 3w Platform"):
            # This button just resets state, user must click sidebar (Streamlit limitation for programmatic nav without session state hacks)
            st.info("Please select 'Equipment Selection' from the sidebar to launch.")

    with col2:
        st.markdown("""
        <div style="background-color: #21252b; padding: 20px; border-radius: 10px; border: 1px solid #333;">
            <h3 style="color: #e06c75;">🌊 Sea Oil Extraction (PrediDrill)</h3>
            <p>Real-time drilling monitor with physics-based tripping simulation.</p>
            <p style="color: #98c379;">Features: Failure Probabilities, Pipe Stress Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch PrediDrill"):
             st.info("Please select 'Sea Oil Extraction' from the sidebar to launch.")

elif selection == "Equipment Selection - Well Oil (3w)":
    try:
        import equipment_app
        importlib.reload(equipment_app) # Ensure freshness on switch
        equipment_app.run()
    except Exception as e:
        st.error(f"Failed to load Equipment Selection module: {e}")

elif selection == "Sea Oil Extraction (PrediDrill)":
    try:
        # We need to run the script. Ideally, we would refactor it to a function too, 
        # but running it via runpy works for single-page scripts.
        target_file = os.path.join(os.path.dirname(__file__), 'predidrill_app.py')
        runpy.run_path(target_file)
    except Exception as e:
        st.error(f"Failed to load PrediDrill module: {e}")
