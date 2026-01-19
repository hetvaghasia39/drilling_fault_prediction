# KOC Unified Drilling Fault Prediction Platform

This repository hosts the **Unified Drilling Intelligence Platform**, a comprehensive solution merging failure prediction models and equipment selection tools into a single interface.

## 🚀 Features

*   **Unified Command Center**: A premium, dark-themed Streamlit dashboard to manage all operations.
*   **Sea Oil Extraction (PrediDrill)**: Real-time drilling monitoring with physics-based tripping simulation and failure prediction.
*   **Equipment Selection (3w)**: An equipment selection engine powered by the 3W dataset, predicting ROP and detecting anomalies.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/hetvaghasia39/drilling_fault_prediction.git
    cd drilling_fault_prediction
    ```

2.  **Set up the environment**:
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install plotly streamlit pandas scikit-learn altair
    ```
    *(Note: Ensure `plotly` is installed as it is required for the 3w module)*

## ▶️ How to Run

You can launch the platform using the provided helper script:

```bash
./run.sh
```

Or manually via Streamlit:

```bash
streamlit run unified_platform.py
```

## 📂 Project Structure

*   `unified_platform.py`: Main application entry point.
*   `equipment_app.py`: Logic for the Equipment Selection module (formerly 3w).
*   `predidrill_app.py`: Logic for the Sea Oil Extraction module.
*   `3w/`: Original source code and data for the 3w dataset.
*   `models/`: Trained machine learning models (.pkl).
*   `data/`: CSV datasets used for simulation.

## 🔗 Public Access
If you need to share the application, you can use `ngrok` to expose port 8501:
```bash
ngrok http 8501
```
