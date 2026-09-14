# Section 5: Streamlit Application, Deployment, MLOps, Monitoring and Testing

**Author / Responsible Teammate**: Member 5  
**Target Artifacts**: `notebooks/05_deployment_mlops_monitoring_testing.ipynb`, `app.py`, `Dockerfile`, `docker-compose.yml`, `docs/deployment_and_mlops.md`, `docs/test_report.md`, `data/prediction_history.db`, `figures/29_` through `figures/32_`

***

## 1. Overview and Architecture

This deliverable provides the production deployment, interactive Streamlit Machine Health Dashboard, real-time IoT sensor telemetry streaming simulator, batch scoring and triage pipeline, statistical feature drift monitoring, automated testing suite, and containerization.

### Production Pipeline Flow:
```
Raw IoT Sensor Telemetry (Air Temp, Process Temp, RPM, Torque, Tool Wear, Type)
       │
       ▼
Feature Engineering Pipeline (16 features: temp_diff_k, power_w, wear stages, etc.)
       │
       ├──────────────────────────────────────────────┐
       ▼                                              ▼
Calibrated ML Model (GradientBoostingClassifier)    Statistical Anomaly Engine (Healthy Distance)
       │                                              │
       ▼                                              ▼
Failure Probability & Risk Tier                    Anomaly Score & Out-of-Bounds Flag
       │                                              │
       ├──────────────────────────────────────────────┘
       ▼
Prescriptive AI Engine (Physics Failure SOPs, ETA, Work Orders, Cost Savings vs $5,000 FN)
       │
       ├──────────────────────────────────────────────┐
       ▼                                              ▼
Streamlit Interactive Dashboard UI          SQLite Prediction Audit Log (prediction_history.db)
```

***

## 2. Streamlit Machine Health Dashboard Modules

The application is organized into six core functional modules:

### 2.1 Live Machine Health and Prescriptive Diagnosis
- Interactive telemetry sliders with physical bounds validation.
- Five industrial presets: Nominal Healthy Machine, Heat Dissipation (HDF) Risk, Tool Wear (TWF) Risk, Overstrain (OSF) Risk, and Power Anomaly (PWF) Risk.
- Real-time gauge visualizing failure probability against operational risk boundaries.
- Model explainability via SHAP waterfall feature contribution analysis.
- Prescriptive AI Maintenance Work Orders detailing root cause, priority, execution ETA, technician checklists, and required spare parts.

### 2.2 Real-Time Sensor Telemetry Stream Simulation
- Continuous synthetic sensor stream generator modeling live milling machine operations.
- Four dynamic fault injection scenarios: Thermal Runaway, Rapid Tool Wear, Overstrain Spike, and Power Surge.
- Multi-channel streaming telemetry chart tracking probability and anomaly progression over time.

### 2.3 Batch CSV Scoring and Fleet Risk Triage
- High-throughput CSV scoring engine for plant-wide machinery fleets.
- Risk tier segmentation (Low, Medium, High, Critical).
- Downloadable scored results CSV with financial cost avoidance calculations.

### 2.4 MLOps Feature Drift and Statistical Monitoring
- Baseline reference distribution established from training data (`ai4i2020_feature_ready.csv`).
- Two-sample Kolmogorov-Smirnov (KS) test for distribution shift detection ($p < 0.05$).
- Population Stability Index (PSI) tracking:
  * PSI < 0.10: Stable / No drift
  * 0.10 <= PSI < 0.25: Moderate drift
  * PSI >= 0.25: Significant drift / Retraining trigger
- Distribution overlay histograms comparing reference vs live production features.

### 2.5 Prediction History Database and Audit Trail
- Embedded SQLite storage (`data/prediction_history.db`) recording all single and batch scoring events.
- Audit attributes: timestamp, machine ID, raw telemetry, failure probability, risk tier, anomaly score, predicted failure mode, recommended action, and cost savings.
- Searchable history table with CSV export capability.

### 2.6 Model Registry and Governance Metadata
- Tracks model architecture, model ID (`GB-PRED-MAINT-V1.0`), version (`1.0.0`), optimal decision threshold ($t^* = 0.26$), PR-AUC ($0.9461$), ROC-AUC ($0.9766$), and cost parameters ($C_{FN}=\$5,000$, $C_{FP}=\$500$).

***

## 3. Operational Risk Tier Framework

| Risk Tier | Probability Range | Action Required | Cost Penalty / Benefit |
| :- | :-: | :- | :- |
| **Low Risk** | P < 0.15 | Routine continuous operation; standard inspection cycle | Normal operating cost |
| **Medium Risk** | 0.15 <= P < 0.26 | Monitor closely; inspect secondary sensor telemetry | Pre-emptive check |
| **High Risk** | 0.26 <= P < 0.75 | Schedule preventive maintenance within 24 hours | $4,500 net cost saved vs breakdown |
| **Critical Risk** | P >= 0.75 | Immediate emergency shutdown; replace worn tooling/parts | Prevents $5,000 catastrophic failure |

***

## 4. Docker Containerization and Deployment

### 4.1 Build Docker Image
```bash
docker build -t industrial-predictive-maintenance:latest .
```

### 4.2 Run with Docker
```bash
docker run -d -p 8501:8501 -name predictive_maint_dashboard industrial-predictive-maintenance:latest
```

### 4.3 Run with Docker Compose
```bash
docker-compose up -d
```

Access the live dashboard at `http://localhost:8501`.