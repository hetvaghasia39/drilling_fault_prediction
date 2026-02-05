import pandas as pd
import numpy as np
import os
import glob
import pickle
import random
import streamlit as st

# Configuration - Update paths to be relative or absolute based on where this is run
# Assuming run from kocmerged/predictive_drilling, and dataset is in ../../drill_log/dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../drill_log/dataset"))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../drill_log/model_output/rf_model.pkl"))

FEATURES = ['P-TPT', 'P-PDG', 'T-TPT', 'QGL', 'P-MON-CKP', 'T-JUS-CKP']
ROLLING_WINDOW = 60

class StreamlitDrillSimulator:
    def __init__(self):
        self.history = pd.DataFrame(columns=FEATURES)
        self.load_model()
        self.reset()

    def load_model(self):
        try:
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading model from {MODEL_PATH}: {e}")
            self.model = None

    def reset(self):
        self.current_idx = 0
        self.history = pd.DataFrame(columns=FEATURES)
        self.is_fault_injected = False
        self.df_stream = None
        self.status = "IDLE"
        self.pick_normal_file()

    def pick_normal_file(self):
        files = glob.glob(os.path.join(DATASET_DIR, "**", "WELL-*.parquet"), recursive=True)
        if files:
            self.current_file = random.choice(files)
            self.df_stream = pd.read_parquet(self.current_file)[FEATURES].fillna(0)
            self.status = "NORMAL"
        else:
            st.error(f"No WELL files found in {DATASET_DIR}")

    def inject_fault(self):
        files = glob.glob(os.path.join(DATASET_DIR, "**", "SIMULATED_*.parquet"), recursive=True)
        if files:
            fault_file = random.choice(files)
            self.df_stream = pd.read_parquet(fault_file)[FEATURES].fillna(0)
            self.current_idx = 0
            self.is_fault_injected = True
            self.status = "FAULT_INJECTED"
            return True
        return False

    def next_step(self):
        if self.df_stream is None or self.current_idx >= len(self.df_stream):
            self.reset()
            return None

        # Get row
        row = self.df_stream.iloc[self.current_idx].to_frame().T
        self.current_idx += 1
        
        # Update history
        self.history = pd.concat([self.history, row], ignore_index=True)
        if len(self.history) > ROLLING_WINDOW:
            self.history = self.history.iloc[-ROLLING_WINDOW:]

        # Predict
        prob = 0.0
        pred = 0
        if len(self.history) == ROLLING_WINDOW and self.model:
            mean_vals = self.history.mean().to_frame().T
            std_vals = self.history.std().to_frame().T
            mean_vals.columns = [f"{c}_mean" for c in FEATURES]
            std_vals.columns = [f"{c}_std" for c in FEATURES]
            X_input = pd.concat([mean_vals, std_vals], axis=1)
            
            try:
                prob = self.model.predict_proba(X_input)[0][1]
                pred = self.model.predict(X_input)[0]
            except:
                pass

        return {
            'data': row.iloc[0].to_dict(),
            'risk_prob': prob,
            'prediction': pred,
            'status': self.status,
            'file': os.path.basename(self.current_file) if hasattr(self, 'current_file') else "Unknown"
        }
