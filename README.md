# Industrial Predictive Maintenance Failure Prevention

## Overview

This repository implements an end-to-end predictive maintenance solution for industrial machinery using the AI4I 2020 Predictive Maintenance dataset. The project combines data engineering, classical machine learning, deep learning anomaly detection, explainability, decision-threshold optimization, deployment, and an operational dashboard for maintenance decision support.

Repository: https://github.com/rawan-2112/Industrial-Predictive-Maintenance-Failure-Prevention

## Problem statement

Industrial equipment failures create costly downtime, unsafe operating conditions, and unnecessary maintenance overhead. In manufacturing environments, the key challenge is not only to predict failure, but to do so with high recall while controlling false alarms and translating the prediction into practical maintenance actions.

This project addresses that challenge by:
- predicting machine failure probability from sensor and operating parameters
- identifying abnormal operating behavior using anomaly detection
- estimating business-risk decisions with cost-aware thresholds
- generating maintenance recommendations and operational priorities
- exposing the solution through a Streamlit application for real-time monitoring and decision support

## Dataset

The project uses the AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository.

- Dataset: AI4I 2020 Predictive Maintenance
- Source: UCI Machine Learning Repository
- Reference: S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications"
- Data pattern: synthetic but realistic milling-machine telemetry with operational, thermal, torque, speed, and wear variables
- Target: machine failure detection

Key dataset characteristics:
- 10,000 rows
- severe class imbalance (~3.39% failure rate)
- sensor-driven operating conditions and engineered physical features
- multiple machine failure modes such as tool wear, heat dissipation, power anomalies, and overstrain conditions

## Project architecture

The repository is structured as a full production-style pipeline:

```text
Raw sensor + machine data
        |
        v
Data preprocessing / EDA / feature engineering
        |
        +--> Classical ML model (GradientBoostingClassifier)
        |
        +--> Deep learning anomaly detector / MLP model
        |
        v
Evaluation + explainability + threshold optimization
        |
        +--> Risk tiering (Low / Medium / High / Critical)
        |
        +--> Maintenance knowledge / recommendation layer
        |
        v
Streamlit dashboard + batch scoring + drift monitoring + audit log
        |
        v
Docker deployment / maintainable production prototype
```

### Major components

1. Data and preprocessing
   - `data/ai4i2020_raw.csv`
   - `data/ai4i2020_cleaned.csv`
   - `data/ai4i2020_feature_ready.csv`
   - `notebooks/01_eda_preprocessing.ipynb`

2. Classical ML and evaluation
   - `notebooks/02_classical_ml_predictive_maintenance_.ipynb`
   - `models/gradient_boosting_final_model.pkl`
   - `models/feature_scaler.pkl`
   - `models/evaluation_summary.json`

3. Deep learning and anomaly detection
   - `notebooks/04_deep_learning_anomaly_detection.ipynb`
   - `models/mlp_failure_classifier.keras`
   - `models/autoencoder_anomaly_detector.keras`
   - `models/deep_learning_thresholds.json`

4. Explainability and business logic
   - `notebooks/03_model_evaluation_explainability_risk.ipynb`
   - `models/risk_thresholds.json`
   - `docs/evaluation_and_explainability.md`

5. Application and monitoring
   - `app.py` — Streamlit application
   - `maintenance_agent.py` — grounded maintenance recommendation layer
   - `maintenance_knowledge.json` — recommendation knowledge base
   - `data/prediction_history.db` — SQLite audit log

6. Deployment and documentation
   - `Dockerfile`
   - `docker-compose.yml`
   - `docs/` project documentation and testing report

## Repository structure

```text
Industrial-Predictive-Maintenance-Failure-Prevention/
├── app.py
├── maintenance_agent.py
├── maintenance_knowledge.json
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── data/
│   ├── ai4i2020_raw.csv
│   ├── ai4i2020_cleaned.csv
│   ├── ai4i2020_feature_ready.csv
│   └── prediction_history.db
├── docs/
│   ├── data_dictionary.md
│   ├── deployment_and_mlops.md
│   ├── evaluation_and_explainability.md
│   ├── pipeline_documentation.md
│   └── test_report.md
├── figures/
│   ├── 01_outlier_boxplots.png
│   ├── 02_univariate_distributions.png
│   ├── 03_class_imbalance.png
│   ├── 04_bivariate_target_focused.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_engineered_features_vs_failure.png
│   ├── 07_feature_selection_mutual_info.png
│   ├── 08_roc_pr_curves.png
│   ├── 09_confusion_matrices.png
│   ├── 10_threshold_cost_optimization.png
│   ├── 11_feature_importance.png
│   ├── 12_shap_global_summary.png
│   ├── 13_shap_local_waterfall.png
│   ├── 14_maintenance_risk_tiers.png
│   ├── 15_MLP Training and Validation Loss.png
│   ├── 16_MLP Training and Validation PR-AUC.png
│   ├── 17_MLP Training and Validation Recall.png
│   ├── 18_MLP Training and Validation Precision.png
│   ├── 19_MLP Test Confusion Matrix — Default Threshold 0.50.png
│   ├── 20_MLP Cost-Sensitive Threshold Optimization.png
│   ├── 21_MLP Test Confusion Matrix — Optimized Threshold 0.36.png
│   ├── 22_Autoencoder Training and Validation Reconstruction Loss.png
│   ├── 23_Autoencoder Reconstruction Error Distribution — Healthy vs Failure.png
│   ├── 24_Autoencoder Reconstruction Error by Machine Status.png
│   ├── 25_Autoencoder Validation Reconstruction Error with Anomaly Threshold.png
│   ├── 26_Autoencoder Test Confusion Matrix.png
│   ├── 27_Deep Learning Model Business-Cost Comparison.png
│   ├── 28_Classical ML vs Deep Learning Business-Cost Comparison.png
│   ├── 29_member5_shap_local_waterfall.png
│   ├── 30_member5_sensor_stream_simulation.png
│   ├── 31_member5_batch_risk_triage.png
│   └── 32_member5_feature_drift_comparison.png
├── models/
│   ├── autoencoder_anomaly_detector.keras
│   ├── best_model_config.json
│   ├── deep_learning_metrics.json
│   ├── deep_learning_model_comparison.csv
│   ├── deep_learning_test_outputs.csv
│   ├── deep_learning_thresholds.json
│   ├── evaluation_summary.json
│   ├── final_metrics.json
│   ├── final_ml_dl_model_comparison.csv
│   ├── gradient_boosting_final_model.pkl
│   ├── feature_scaler.pkl
│   ├── mlp_failure_classifier.keras
│   ├── risk_thresholds.json
│   └── ...
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_classical_ml_predictive_maintenance_.ipynb
│   ├── 03_model_evaluation_explainability_risk.ipynb
│   ├── 04_deep_learning_anomaly_detection.ipynb
│   └── 05_deployment_mlops_monitoring_testing.ipynb
├── SQL/
│   └── queries.sql
└── tests/
    └── test_maintenance_agent.py
```

## Installation and setup

### 1. Clone the repository

```bash
git clone https://github.com/rawan-2112/Industrial-Predictive-Maintenance-Failure-Prevention.git
cd Industrial-Predictive-Maintenance-Failure-Prevention
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the environment

```bash
python --version
pip freeze
```

## Usage instructions

### Run the Streamlit application locally

```bash
streamlit run app.py
```

Then open the local dashboard in your browser, typically at:

```text
http://localhost:8501
```

### Dashboard features

The application includes:
- live machine health scoring
- risk probability and anomaly detection
- threshold-based risk tiers
- prescriptive maintenance recommendations
- batch scoring for multiple machines
- sensor stream simulation
- SQLite prediction history logging
- data drift monitoring

### Run with Docker

Build and start the app with Docker Compose:

```bash
docker-compose up --build
```

Or build the image manually:

```bash
docker build -t industrial-predictive-maintenance:latest .
docker run -p 8501:8501 industrial-predictive-maintenance:latest
```

## Model and evaluation summary

The model stack includes a classical machine learning component and a deep learning / anomaly layer. The final business-oriented risk model is based on a cost-sensitive threshold tuned for industrial decision making.

### Model summary

- Primary classification model: GradientBoostingClassifier
- Deep learning alternatives: MLP classifier and autoencoder anomaly detector
- Feature engineering: physical operating features such as temperature delta, power draw, wear-stage indicators, and torque interactions
- Decision logic: risk probability converted into Low / Medium / High / Critical tiers

### Evaluation metrics

From the project artifacts:

```text
GradientBoostingClassifier
- ROC-AUC: 0.9766
- PR-AUC: 0.9461
- Precision: 0.9556 at optimal threshold
- Recall: 0.8431
- F1-score: 0.8958 at optimized threshold
- Optimal threshold: 0.26
```

### Cost-sensitive thresholding

The project explicitly optimizes the decision threshold to reduce expensive false negatives:

- Cost of false negative: $5,000 per missed failure
- Cost of false positive: $500 per unnecessary inspection
- Cost-optimal threshold: 0.26

This reduces missed failures and captures the real cost trade-off in maintenance operations.

### Business impact summary

The model supports strong preventive action by trading a small number of additional false alarms for a significant reduction in missed failures.

```text
Reactive strategy cost: $255,000
Default model cost: $40,000
Cost-optimized model cost: $16,500
Estimated savings vs reactive: ~$238,500
```

## Application instructions

### Live monitoring workflow

1. Launch the Streamlit app.
2. Select a machine scenario or upload a CSV for batch scoring.
3. Observe the failure probability and risk tier.
4. Review anomaly score, root-cause indicators, and failure mode.
5. Read the maintenance recommendation generated by the advanced AI advisor.
6. Save records to the database for auditing and historical review.

### Drift monitoring and audit trail

The app includes:
- PSI and KS-based feature drift checks
- monitoring plots across features
- SQLite prediction log entries
- download/export of scored batch results and audit history

## Member contributions and results

This repository consolidates work from multiple members across the full project lifecycle.

### Member 1 — Data engineering and preprocessing
- `notebooks/01_eda_preprocessing.ipynb`
- `data/ai4i2020_cleaned.csv`
- `data/ai4i2020_feature_ready.csv`
- `figures/01_` through `figures/07_`
- `docs/pipeline_documentation.md`

### Member 2 — Classical machine learning and model building
- `notebooks/02_classical_ml_predictive_maintenance_.ipynb`
- model artifacts in `models/`
- classical ML comparison and optimized thresholding outputs

### Member 3 — Evaluation, explainability, and risk logic
- `notebooks/03_model_evaluation_explainability_risk.ipynb`
- `figures/08_` through `figures/14_`
- `models/evaluation_summary.json`
- `models/risk_thresholds.json`
- `docs/evaluation_and_explainability.md`

### Member 4 — Deep learning anomaly detection
- `notebooks/04_deep_learning_anomaly_detection.ipynb`
- `figures/15_` through `figures/28_`
- `models/deep_learning_thresholds.json`
- `models/mlp_failure_classifier.keras`
- `models/autoencoder_anomaly_detector.keras`

### Member 5 — Deployment, monitoring, dashboard, and AI maintenance layer
- `notebooks/05_deployment_mlops_monitoring_testing.ipynb`
- `app.py`
- `maintenance_agent.py`
- `maintenance_knowledge.json`
- `figures/29_` through `figures/32_`
- `docs/deployment_and_mlops.md`
- `docs/test_report.md`

## Screenshots and result highlights

The project includes a complete visual result set in `figures/`, covering:
- EDA and outlier analysis
- class imbalance and feature relationships
- ROC / PR curves and confusion matrices
- feature importance and SHAP explanations
- threshold optimization and cost trade-offs
- deep learning anomaly diagnostics
- sensor simulation and drift monitoring
- maintenance risk triage dashboard outputs

Representative outputs:
- `figures/08_roc_pr_curves.png`
- `figures/10_threshold_cost_optimization.png`
- `figures/12_shap_global_summary.png`
- `figures/14_maintenance_risk_tiers.png`
- `figures/29_member5_shap_local_waterfall.png`
- `figures/30_member5_sensor_stream_simulation.png`
- `figures/31_member5_batch_risk_triage.png`
- `figures/32_member5_feature_drift_comparison.png`

## Documentation

Additional project documentation is available in `docs/`:
- `docs/data_dictionary.md`
- `docs/pipeline_documentation.md`
- `docs/evaluation_and_explainability.md`
- `docs/deployment_and_mlops.md`
- `docs/test_report.md`

## Final note

This project is designed as an operational prototype for predicting machine failure risk and converting that risk into practical maintenance action. It connects the full chain from raw industrial data to machine learning, explainability, deployment, and grounded operational recommendations in a single reproducible workflow.
