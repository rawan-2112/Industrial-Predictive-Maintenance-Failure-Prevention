# Section 3: Model Evaluation, Explainability & Maintenance Risk Intelligence

**Author / Responsible Teammate**: Member 3  
**Target Artifacts**: `notebooks/03_model_evaluation_explainability_risk.ipynb`, `figures/08_` through `figures/14_`, `models/evaluation_summary.json`, `models/risk_thresholds.json`

---

## 1. Overview & Evaluation Philosophy

In industrial predictive maintenance applications (UCI AI4I 2020 Predictive Maintenance Dataset), model evaluation cannot rely on standard accuracy due to severe class imbalance (~3.39% failures). Furthermore, evaluating predictions with symmetric loss metrics fails to capture industrial reality:

> **Industrial Reality**: A **False Negative (FN)**—failing to detect an impending machine breakdown—results in catastrophic tool destruction, secondary component damage, unscheduled line shutdown, and thousands of dollars in repairs ($C_{FN} \approx \$5,000$). Conversely, a **False Positive (FP)**—an unnecessary preventive maintenance inspection—costs only a minor technician check fee ($C_{FP} \approx \$500$).

Therefore, Member 3's task centers around:
1. Building a **comprehensive evaluation framework** (PR-AUC, ROC-AUC, Precision, Recall, F1, False Negative Rate).
2. Implementing **Business Cost Simulation & Threshold Optimization** ($C_{total} = C_{FN} \cdot FN + C_{FP} \cdot FP$).
3. Conducting **Error Analysis** on missed failures vs false alarms.
4. Extracting **Global & Local Explainability** via Feature Importance and SHAP (Shapley Additive exPlanations).
5. Establishing operational **Maintenance Risk Tiers** (Low, Medium, High, Critical) and domain-grounded failure classification rules.

---

## 2. Comprehensive Evaluation Framework

### 2.1 Performance Metrics Summary
The hand-off model (`GradientBoostingClassifier`) was evaluated on the unseen 15% stratified test set ($N=1,500$, failure count $= 51$).

| Metric | Default Threshold ($t = 0.50$) | Cost-Optimal Threshold ($t^* = 0.22$) | Industrial Significance |
| :--- | :---: | :---: | :--- |
| **Precision** | **100.00%** | **94.12%** | High precision prevents unnecessary maintenance checks |
| **Recall (Sensitivity)** | **84.31%** | **94.12%** | High recall minimizes missed machine failures |
| **F1-Score** | **0.9149** | **0.9412** | Overall harmonic balance |
| **PR-AUC (Avg Precision)** | **0.9461** | **0.9461** | Primary metric for imbalanced failure detection |
| **ROC-AUC** | **0.9766** | **0.9766** | Global discrimination capacity |
| **False Negative Rate (FNR)** | **15.69%** (8 missed) | **5.88%** (3 missed) | **62.5% reduction in missed failures** |
| **False Positives (FP)** | **0** | **3** | Minimal false alarm penalty |
| **False Negatives (FN)** | **8** | **3** | **5 catastrophic breakdowns prevented** |

---

## 3. Threshold Optimization & Cost-Based Alerting Analysis

### 3.1 Business Cost Formulation
Total operational cost function:
$$C_{total}(t) = C_{FN} \times FN(t) + C_{FP} \times FP(t)$$

Given an asymmetric cost ratio of $10:1$ ($C_{FN} = \$5,000$, $C_{FP} = \$500$):
* **Reactive Baseline (No Model)**: Missing all 51 test failures $\rightarrow \$5,000 \times 51 = \mathbf{\$255,000}$
* **Default Model ($t = 0.50$)**: 8 Missed Failures ($FN=8$), 0 False Alarms ($FP=0$) $\rightarrow \$5,000 \times 8 + \$500 \times 0 = \mathbf{\$40,000}$
* **Cost-Optimal Model ($t^* = 0.22$)**: 3 Missed Failures ($FN=3$), 3 False Alarms ($FP=3$) $\rightarrow \$5,000 \times 3 + \$500 \times 3 = \mathbf{\$16,500}$

```
                          BUSINESS COST COMPARISON (TEST SET)
┌──────────────────────────────────────┬─────────────────┬─────────────────────┐
│ Maintenance Strategy                 │ Total Cost ($)  │ Savings vs Reactive │
├──────────────────────────────────────┼─────────────────┼─────────────────────┤
│ Reactive (Run to Failure)            │ $255,000        │ $0 (0%)             │
│ ML Default Threshold (t = 0.50)      │ $40,000         │ $215,000 (84.3%)    │
│ ML Cost-Optimal Threshold (t = 0.22) │ $16,500         │ $238,500 (93.5%)    │
└──────────────────────────────────────┴─────────────────┴─────────────────────┘
```

> **Key Financial Impact**: Threshold optimization from $t=0.50 \rightarrow t^*=0.22$ saves an additional **$23,500** (a **58.75% cost reduction** compared to default ML predictions).

---

## 4. Explainability & Feature Importance (SHAP)

### 4.1 Global Feature Driver Analysis
1. **Permutation & Gini Importance**:
   - `torque_nm` (Torque in Nm) and `tool_wear_min` (Tool wear duration in minutes) represent over **65%** of predictive power.
   - Domain engineered interactions (`tool_wear_torque_product` and `power_w`) rank as high-yield secondary predictors.
2. **SHAP Summary Breakdown**:
   - **High Torque** combined with **High Tool Wear** strongly pushes Shapley values positive ($\rightarrow$ failure prediction).
   - Low temperature difference (`temp_diff_k < 8.6 K`) triggers heat dissipation failure signals.

### 4.2 Local Prediction-Level Explanations (Waterfall Analysis)
* **True Positive Failure Case**: `torque_nm = 62.5 Nm` (+0.42 SHAP), `tool_wear_min = 210 min` (+0.38 SHAP), and `power_w = 9.8 kW` (+0.18 SHAP) drive the failure probability to **98.4%**.
* **Normal Operating Case**: Moderate torque (`38 Nm`) and fresh tool (`15 min`) yield SHAP contributions below base value $\rightarrow$ failure probability **0.02%**.

---

## 5. Maintenance Risk Tiers & Failure Mode Rules

To communicate predictions effectively to plant engineers and downstream dashboard components (Streamlit & Advanced AI Agent), operational probabilities are mapped to 4 actionable risk tiers:

```
┌─────────────────┬──────────────────────┬────────────────────────────────────────────────────────┐
│ Risk Tier       │ Probability Range    │ Recommended Industrial Action                          │
├─────────────────┼──────────────────────┼────────────────────────────────────────────────────────┤
│ Low Risk        │ P < 0.15             │ Normal operation; standard routine check schedule      │
│ Medium Risk     │ 0.15 <= P < 0.22     │ Monitor closely; inspect secondary sensor telemetry    │
│ High Risk       │ 0.22 <= P < 0.75     │ Schedule preventive maintenance within 24–48 hours     │
│ Critical Risk   │ P >= 0.75            │ Immediate machine shutdown; replace worn components    │
└─────────────────┴──────────────────────┴────────────────────────────────────────────────────────┘
```

### Failure Mode Heuristic Rules
Based on sensor physical bounds:
* **Tool Wear Failure (TWF)**: `tool_wear_min >= 200 min`.
* **Heat Dissipation Failure (HDF)**: `temp_diff_k < 8.6 K` AND `rotational_speed_rpm < 1380 RPM`.
* **Power Failure (PWF)**: `power_w < 3500 W` OR `power_w > 9000 W`.
* **Overstrain Failure (OSF)**: `tool_wear_torque_product > 11,000 min·Nm`.

---

## 6. Output Files & Artifact Summary

1. **`figures/08_roc_pr_curves.png`**: ROC Curve ($AUC=0.9766$) and Precision-Recall Curve ($PR-AUC=0.9461$).
2. **`figures/09_confusion_matrices.png`**: Confusion matrices comparing default vs cost-optimal thresholds.
3. **`figures/10_threshold_cost_optimization.png`**: Business loss minimization curve across threshold sweep.
4. **`figures/11_feature_importance.png`**: Gini vs Permutation Feature Importance.
5. **`figures/12_shap_global_summary.png`**: SHAP beeswarm global feature impact plot.
6. **`figures/13_shap_local_waterfall.png`**: SHAP local waterfall prediction breakdown.
7. **`figures/14_maintenance_risk_tiers.png`**: Machinery distribution bar chart across risk tiers.
8. **`models/evaluation_summary.json`**: Complete metrics, confusion matrices, and business cost savings.
9. **`models/risk_thresholds.json`**: Threshold configuration and risk tier operational definitions.
