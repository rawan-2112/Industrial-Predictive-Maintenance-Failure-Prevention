# Data Dictionary — AI4I 2020 Predictive Maintenance Dataset

**Source:** UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset
**Citation:** S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," 3rd IEEE Int'l Conference on Artificial Intelligence for Industries (AI4I), 2020.
**Rows:** 10,000 | **Raw columns:** 14 | **Type:** Synthetic, modeled after a real milling machine

| # | Column (raw name) | Renamed (feature-ready) | Type | Description | Role |
|---|---|---|---|---|---|
| 1 | `UDI` | `udi` | int | Unique data point index (1–10000). No predictive value. | Identifier — dropped before modeling |
| 2 | `Product ID` | `product_id` | string | Product serial code, e.g. `M14860`. Prefix letter duplicates `Type`. | Identifier — dropped before modeling |
| 3 | `Type` | `type` | categorical (L/M/H) | Product quality variant: Low (50%), Medium (30%), High (20%) of the population. | Predictor (categorical, one-hot encoded) |
| 4 | `Air temperature [K]` | `air_temperature_k` | float | Ambient air temperature, generated as a random walk normalized to σ≈2 K around 300 K. | Predictor (numeric) |
| 5 | `Process temperature [K]` | `process_temperature_k` | float | Process/machine temperature, random walk normalized to σ≈1 K, then air temperature + 10 K. Strongly correlated with air temperature by construction. | Predictor (numeric) |
| 6 | `Rotational speed [rpm]` | `rotational_speed_rpm` | int | Spindle rotational speed, derived from ~2860 W power with normally distributed noise. | Predictor (numeric) |
| 7 | `Torque [Nm]` | `torque_nm` | float | Torque values, normally distributed around 40 Nm with σ=10 Nm (no negative values). | Predictor (numeric) |
| 8 | `Tool wear [min]` | `tool_wear_min` | int | Cumulative tool wear in minutes. Quality variant (H/M/L) adds 5/3/2 extra minutes of wear per process due to the different product grades. | Predictor (numeric) |
| 9 | `Machine failure` | `machine_failure` | binary (0/1) | **Primary target.** 1 if the machine failed for *any* of the 5 independent failure modes below (TWF, HDF, PWF, OSF, RNF). Class balance: 9,661 no-failure vs 339 failure (3.39%) — strongly imbalanced. | **Target label** |
| 10 | `TWF` | `twf` | binary (0/1) | Tool Wear Failure flag — tool fails/replaced between 200–240 min of wear at a random point. | Failure-mode sub-flag — **leakage risk, excluded from feature set**, retained only to validate/derive the target |
| 11 | `HDF` | `hdf` | binary (0/1) | Heat Dissipation Failure — triggers when air/process temperature difference < 8.6 K and rotational speed < 1380 rpm. | Same as above |
| 12 | `PWF` | `pwf` | binary (0/1) | Power Failure — triggers when the product of torque and rotational speed (power) falls outside 3500–9000 W. | Same as above |
| 13 | `OSF` | `osf` | binary (0/1) | Overstrain Failure — triggers when tool wear × torque exceeds a quality-variant-specific threshold (11,000 / 12,000 / 13,000 min·Nm for L/M/H). | Same as above |
| 14 | `RNF` | `rnf` | binary (0/1) | Random Failures — ~0.1% chance of failure regardless of process parameters, independent of all other variables. | Same as above |

## Engineered features added during feature engineering (see pipeline doc)

| Engineered feature | Formula / logic | Rationale |
|---|---|---|
| `temp_diff_k` | `process_temperature_k − air_temperature_k` | Directly mirrors the physical HDF trigger condition; expected strong signal for heat-dissipation risk. |
| `power_w` | `torque_nm × rotational_speed_rpm × (2π/60)` | Mechanical power in watts; mirrors the physical PWF trigger condition. |
| `tool_wear_torque_product` | `tool_wear_min × torque_nm` | Mirrors the physical OSF trigger condition (quality-dependent overstrain threshold). |
| `speed_torque_ratio` | `rotational_speed_rpm / torque_nm` | Captures the inverse relationship between speed and torque common in milling operations. |
| `wear_bucket` | Binned `tool_wear_min` into `[0-80, 80-160, 160-240]` | Captures non-linear wear-stage effects (early life / mid-life / near tool-change window at 200–240 min). |
| `type_L` / `type_M` / `type_H` | One-hot encoding of `type` | Converts categorical quality variant into numeric predictors. |

## Data governance notes
- Dataset is **synthetic** (not real sensor data) — modeled on a real milling machine but should not be described as production industrial telemetry in the final report.
- The five failure-mode sub-flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are **not used as model inputs** — they are constituent components of the target itself, and including them would leak the answer (this is documented explicitly as a leakage risk per the project's Common Risks / Quality Checks).
- No personally identifiable information is present.
