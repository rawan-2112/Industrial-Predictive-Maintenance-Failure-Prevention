# Pipeline Documentation — Member 1: Data Engineering, Preprocessing & EDA
## Project 5: Industrial Predictive Maintenance & Failure Prevention

## 1. Scope of this deliverable
This covers everything upstream of modeling: dataset acquisition, quality auditing, cleaning,
exploratory analysis, preprocessing, feature engineering, feature selection, and leakage control.
Output is a single **feature-ready CSV** handed off to the ML-track and DL-track members.

## 2. Dataset
- **Name:** AI4I 2020 Predictive Maintenance Dataset
- **Source:** UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- **Citation:** S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," 2020 Third International Conference on Artificial Intelligence for Industries (AI4I), 2020.
- **Size:** 10,000 rows × 14 raw columns, synthetic data modeled on a real milling machine.
- **License note:** cite the original authors in the final report; do not redistribute without attribution.

## 3. Pipeline stages (Raw → Cleaned → Feature-Ready)

```
data/ai4i2020_raw.csv            (10000, 14)   <- as downloaded, unmodified
        │
        │  Section 2-4 of the notebook: dtype/duplicate/missing/invalid-value audit,
        │  internal consistency check (Machine failure vs OR of 5 sub-flags),
        │  outlier detection (IQR) — inspected and retained (no rows dropped)
        ▼
data/ai4i2020_cleaned.csv        (10000, 14)   <- audited, confirmed clean
        │
        │  Section 8-10 of the notebook: drop identifiers (UDI, Product ID),
        │  rename to snake_case, one-hot encode `type`,
        │  engineer temp_diff_k / power_w / tool_wear_torque_product /
        │  speed_torque_ratio / wear-stage buckets,
        │  DROP the 5 failure sub-flags (TWF/HDF/PWF/OSF/RNF) — leakage risk
        ▼
data/ai4i2020_feature_ready.csv  (10000, 16)   <- final hand-off artifact
```

## 4. Data-quality audit results
| Check | Result |
|---|---|
| Missing values | 0 across all 14 columns |
| Duplicate rows | 0 |
| Duplicate UDI (primary key) | 0 |
| Invalid/out-of-range values (negative temps, speeds, torque, invalid `Type`/label) | 0 |
| Label consistency (`Machine failure` vs OR of 5 sub-flags) | see notebook Section 2 for exact mismatch count/rate — flagged as minor labeling noise, not corrected (original label is ground truth) |

No imputation or de-duplication was necessary. This was **verified**, not assumed.

## 5. Outlier handling policy
IQR-based detection was run on all 5 continuous variables. Outliers (chiefly high-RPM /
high-torque combinations) coincide with a higher machine-failure rate than the dataset average,
so they were **retained** rather than removed or capped — they are part of the real failure
signal. This is documented explicitly so the ML/DL members don't accidentally strip predictive
extremes during their own preprocessing.

## 6. Class imbalance
`machine_failure`: 9,661 negative vs 339 positive (**3.39% positive rate**, ≈29:1 imbalance).
Downstream implication (per the project's ML/DL requirements): use stratified train/val/test
splits, class weighting or resampling, and PR-AUC/recall/F1 as primary metrics — **never accuracy
alone**.

## 7. Feature engineering rationale
All four engineered continuous features and the wear-stage buckets are derived directly from the
**documented physical failure-trigger equations** of the AI4I dataset (see `data_dictionary.md`),
not arbitrary transforms:
- `temp_diff_k` mirrors the Heat-Dissipation-Failure (HDF) trigger (temp diff < 8.6K & RPM < 1380)
- `power_w` mirrors the Power-Failure (PWF) trigger (power outside 3500–9000 W)
- `tool_wear_torque_product` mirrors the Overstrain-Failure (OSF) trigger (wear × torque > quality-dependent threshold)
- `wear_*` buckets flag the 200–240 min tool-wear window where Tool-Wear-Failure (TWF) occurs

Boxplots in the notebook (Section 9) confirm each engineered feature visibly separates failed vs.
healthy machines around its documented physical threshold.

## 8. Leakage control (critical — applies to ALL team members)
`TWF, HDF, PWF, OSF, RNF` are **components of the target itself**
(`machine_failure = OR(TWF, HDF, PWF, OSF, RNF)`). They are:
- kept in `ai4i2020_cleaned.csv` only for target-validation / optional failure-mode classification stretch goals
- **removed** from `ai4i2020_feature_ready.csv` — the file every other track must train on
- never to be re-joined into the ML/DL feature matrix

## 9. Feature selection
Mutual-information screening (notebook Section 11) confirms `power_w`, `torque_nm`,
`tool_wear_torque_product`, `tool_wear_min`, and `temp_diff_k` carry the strongest signal.
All engineered + raw numeric/categorical features are still delivered in the feature-ready file
(no columns pre-emptively dropped) — pruning decisions are left to the ML-track member during
model tuning, informed by this ranking.

## 10. Hand-off artifact
**File:** `data/ai4i2020_feature_ready.csv` (10,000 rows × 16 columns)

**Columns:** `air_temperature_k, process_temperature_k, rotational_speed_rpm, torque_nm,
tool_wear_min, type_H, type_L, type_M, temp_diff_k, power_w, tool_wear_torque_product,
speed_torque_ratio, wear_early_life, wear_mid_life, wear_wear_window, wear_beyond_240,
machine_failure (target)`

## 11. Repository placement
Per the team's suggested repository structure:
```
project-5-predictive-maintenance/
├── data/
│   ├── ai4i2020_raw.csv
│   ├── ai4i2020_cleaned.csv
│   └── ai4i2020_feature_ready.csv
├── notebooks/
│   └── 01_eda_preprocessing.ipynb
├── figures/
│   └── 01_outlier_boxplots.png ... 07_feature_selection_mutual_info.png
├── docs/
│   ├── data_dictionary.md
│   └── pipeline_documentation.md   (this file)
```
