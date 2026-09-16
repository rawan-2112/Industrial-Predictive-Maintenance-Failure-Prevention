# Project 5 — Industrial Predictive Maintenance & Failure Prevention

[![Dataset](https://img.shields.io/badge/Dataset-UCI%20AI4I%202020-blue)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Gradient%20Boosting-orange)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)](https://shap.readthedocs.io/)

**Graduation project** — TechTrek Advanced Data Science & AI Track

An end-to-end **industrial predictive maintenance** pipeline that predicts machine failures before they happen, using the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (UCI). The system covers data engineering, classical ML modeling, model evaluation, explainability (SHAP), and business cost optimization.

---

## Key Results

| Metric | Value |
|:---|:---:|
| **ROC-AUC** | 0.9766 |
| **PR-AUC** | 0.9273 |
| **Precision** (default threshold) | 100% |
| **Recall** (default threshold) | 84.31% |
| **F1-Score** | 0.9149 |
| **Cost savings vs. reactive maintenance** | **$214,000 (83.9%)** |
| **Best Model** | Gradient Boosting Classifier |

---

## Repository Structure

```
Industrial-Predictive-Maintenance-Failure-Prevention/
├── data/
│   ├── ai4i2020_raw.csv                # Original dataset, unmodified
│   ├── ai4i2020_cleaned.csv            # Audited & validated (Raw → Cleaned)
│   └── ai4i2020_feature_ready.csv      # Encoded + engineered, leakage-safe (hand-off to ML)
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb      # [Member 1] Full EDA & preprocessing pipeline
│   ├── 02_classical_ml_predictive_maintenance_.ipynb  # [Member 2] ML model training & selection
│   └── 03_model_evaluation_explainability_risk.ipynb  # [Member 3] Evaluation, SHAP, cost analysis
├── figures/
│   ├── 01_outlier_boxplots.png         # Outlier analysis (IQR)
│   ├── 02_univariate_distributions.png # Feature distributions
│   ├── 03_class_imbalance.png          # Target class balance
│   ├── 04_bivariate_target_focused.png # Feature vs. target analysis
│   ├── 05_correlation_heatmap.png      # Feature correlation matrix
│   ├── 06_engineered_features_vs_failure.png  # Engineered features validation
│   ├── 07_feature_selection_mutual_info.png   # Mutual information ranking
│   ├── 08_roc_pr_curves.png            # ROC & Precision-Recall curves
│   ├── 09_confusion_matrices.png       # Confusion matrices (default vs. optimal threshold)
│   ├── 10_threshold_cost_optimization.png # Business cost minimization
│   ├── 11_feature_importance.png       # Gini vs. Permutation importance
│   ├── 12_shap_global_summary.png      # SHAP global feature impact
│   ├── 13_shap_local_waterfall.png     # SHAP local waterfall (true positive case)
│   └── 14_maintenance_risk_tiers.png   # Risk tier distribution
├── models/
│   ├── gradient_boosting_final_model.pkl  # Trained model artifact
│   ├── feature_scaler.pkl              # StandardScaler for feature normalization
│   ├── best_model_config.json          # Hyperparameter configuration
│   ├── final_metrics.json              # Validation & test set metrics
│   ├── evaluation_summary.json         # Complete evaluation (confusion matrices, costs)
│   └── risk_thresholds.json            # Risk tier definitions & cost parameters
├── SQL/
│   ├── queries.sql                     # SQL analysis queries
│   └── ER Diagram.png                  # Entity-Relationship diagram
├── docs/
│   ├── data_dictionary.md              # Complete column-by-column data dictionary
│   ├── pipeline_documentation.md       # Data engineering pipeline documentation
│   └── evaluation_and_explainability.md # Model evaluation, SHAP & risk tier documentation
├── requirements.txt
└── README.md                           # (this file)
```

---

## Pipeline Overview

```
     ┌──────────────┐      ┌──────────────┐      ┌──────────────────────┐
     │   Member 1   │      │   Member 2   │      │      Member 3        │
     │ Data Eng &   │─────▶│ Classical ML │─────▶│ Evaluation, SHAP     │
     │ Preprocessing│      │ Training     │      │ & Risk Intelligence  │
     └──────────────┘      └──────────────┘      └──────────────────────┘
           │                      │                         │
     ai4i2020_raw.csv      GradientBoosting        evaluation_summary.json
           │               _final_model.pkl         risk_thresholds.json
     ai4i2020_cleaned.csv        │                  SHAP analysis
           │               final_metrics.json       figures/08–14
     ai4i2020_feature_          │
     ready.csv            best_model_config.json
           │
     figures/01–07
```

### Member 1 — Data Engineering, Preprocessing & EDA
- Dataset acquisition, quality audit (zero missing values, zero duplicates)
- Outlier detection via IQR — retained as part of the failure signal
- Feature engineering: `temp_diff_k`, `power_w`, `tool_wear_torque_product`, `speed_torque_ratio`, wear-stage buckets
- **Leakage control**: Removed failure sub-flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) from feature set — they are components of the target itself
- Exported clean, leakage-safe `ai4i2020_feature_ready.csv` as the hand-off artifact

### Member 2 — Classical ML Model Training & Selection
- Stratified 70/15/15 train/validation/test split
- Trained and compared: Logistic Regression, Random Forest, Gradient Boosting (XGBoost-style)
- **Best model**: Gradient Boosting Classifier (ROC-AUC = 0.9766, Precision = 100%, Recall = 84.31%)
- Saved trained model, scaler, and configuration artifacts

### Member 3 — Model Evaluation, Explainability & Risk Intelligence
- Comprehensive evaluation: PR-AUC, ROC-AUC, Precision, Recall, F1, FNR, confusion matrices
- **Threshold optimization**: Cost-based analysis with asymmetric loss (FN costs 10x FP)
- **SHAP explainability**: Global feature impact + local waterfall explanations for individual predictions
- **Maintenance risk tiers**: Low / Medium / High / Critical operational categories
- Full documentation in [`docs/evaluation_and_explainability.md`](docs/evaluation_and_explainability.md)

---

## How to Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/rawan-2112/Industrial-Predictive-Maintenance-Failure-Prevention.git
   cd Industrial-Predictive-Maintenance-Failure-Prevention
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # macOS/Linux:
   source .venv/bin/activate
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Register the kernel and launch Jupyter**
   ```bash
   python -m ipykernel install --user --name project5-venv
   jupyter notebook
   ```
   All notebooks have outputs saved — you can read them directly. To re-run from scratch: `Kernel → Restart & Run All`.

5. **Important**: All ML/DL work must build on `data/ai4i2020_feature_ready.csv` — the single hand-off artifact. Do **not** re-introduce `twf, hdf, pwf, osf, rnf` as model inputs (see [`docs/pipeline_documentation.md`](docs/pipeline_documentation.md), Section 8).

---

## Dataset Citation

S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," *3rd IEEE International Conference on Artificial Intelligence for Industries (AI4I)*, 2020.

**Source**: [UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
