# EDA and Anomaly Detection Report

## Dataset Overview
- **Total Records**: 151 rows
- **Columns**: `Depth`, `WOB`, `SURF_RPM`, `ROP_AVG`, `PHIF`, `VSH`, `SW`, `KLOGH`
- **Missing Values**: None found in the processed dataset.

## Exploratory Analysis Findings
1.  **Stable Operations**: For the majority of the log (up to Depth ~3950), parameters are relatively stable.
2.  **Significant Event**: Around Depth 3955-4085, there is a drastic change:
    - **KLOGH (Permeability?)**: Spikes from ~0.001 to ~600.
    - **SW (Saturation)**: Fluctuates/Drops.
    - **ROP_AVG**: Shows variance.
    This suggests a change in formation (hitting a reservoir?) or a drilling issue (msg: water influx?).

## Unsupervised Learning Approach (Anomaly Detection)
Given the unlabelled nature of the data, we applied an **Isolation Forest** algorithm to detect anomalies.
- **Goal**: Identify depth intervals where drilling parameters deviate significantly from the "normal" baseline.
- **Results**: The model identified the high variability region at the bottom of the well as anomalous (Score < 0).

## Enterprise Project Structure
The project has been organized into a scalable structure:
- `data/`: Managed data lifecycle (raw -> processed).
- `src/`: Modular code for data gen, visualization, and modeling.
- `notebooks/`: Sandbox for experiments.
- `tests/`: Placeholder for unit tests.
- `config/`: Configuration management.

## Next Steps
1.  **Labeling**: expert review to confirm if the anomalies at Depth > 3955 are "Failures" or just "Formation Changes".
2.  **Forecasting**: If "Failure" is confirmed, train a sequence model (LSTM/RNN) to predict this state change `n` steps ahead.
