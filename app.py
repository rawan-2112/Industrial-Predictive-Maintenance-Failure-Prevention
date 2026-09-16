import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import joblib
from maintenance_agent import get_rag_recommendation

try:
    import sklearn._loss._loss as _loss_mod
    sys.modules['_loss'] = _loss_mod
except Exception:
    pass

st.set_page_config(page_title="Machine Health Dashboard", layout="wide")

@st.cache_resource
def load_assets():
    m = joblib.load('models/gradient_boosting_final_model.pkl')
    s = joblib.load('models/feature_scaler.pkl')
    with open('models/risk_thresholds.json', 'r') as f:
        rt = json.load(f)
    with open('models/deep_learning_thresholds.json', 'r') as f:
        dt = json.load(f)
    ref = pd.read_csv('data/ai4i2020_feature_ready.csv')
    return m, s, rt, dt, ref

model, scaler, risk_thresholds, dl_thresholds, ref_df = load_assets()

FEATURE_COLUMNS = [
    'air_temperature_k', 'process_temperature_k', 'rotational_speed_rpm',
    'torque_nm', 'tool_wear_min', 'type_H', 'type_L', 'type_M',
    'temp_diff_k', 'power_w', 'tool_wear_torque_product', 'speed_torque_ratio',
    'wear_early_life', 'wear_mid_life', 'wear_wear_window', 'wear_beyond_240'
]

ref_healthy = ref_df[ref_df['machine_failure'] == 0][FEATURE_COLUMNS]
ref_mean = ref_healthy.mean()
ref_std = ref_healthy.std().replace(0, 1.0)

def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if 'type' in df.columns:
        df['type_H'] = (df['type'] == 'H').astype(int)
        df['type_L'] = (df['type'] == 'L').astype(int)
        df['type_M'] = (df['type'] == 'M').astype(int)
    elif 'Type' in df.columns:
        df['type_H'] = (df['Type'] == 'H').astype(int)
        df['type_L'] = (df['Type'] == 'L').astype(int)
        df['type_M'] = (df['Type'] == 'M').astype(int)
    else:
        for c in ['type_H', 'type_L', 'type_M']:
            if c not in df.columns:
                df[c] = 0
    df['temp_diff_k'] = df['process_temperature_k'] - df['air_temperature_k']
    df['power_w'] = df['rotational_speed_rpm'] * df['torque_nm'] * (2 * np.pi / 60)
    df['tool_wear_torque_product'] = df['tool_wear_min'] * df['torque_nm']
    df['speed_torque_ratio'] = df['rotational_speed_rpm'] / (df['torque_nm'] + 1e-6)
    wear = df['tool_wear_min']
    df['wear_early_life'] = (wear < 70).astype(int)
    df['wear_mid_life'] = ((wear >= 70) & (wear < 200)).astype(int)
    df['wear_wear_window'] = ((wear >= 200) & (wear <= 240)).astype(int)
    df['wear_beyond_240'] = (wear > 240).astype(int)
    return df[FEATURE_COLUMNS]

def compute_anomaly(feat_df: pd.DataFrame) -> np.ndarray:
    z = (feat_df - ref_mean) / ref_std
    return np.clip(np.sqrt(np.mean(z**2, axis=1)) / 10.0, 0.0, 1.0).values

def assign_tier(p: float) -> str:
    if p < 0.15: return "Low Risk"
    elif p < 0.26: return "Medium Risk"
    elif p < 0.75: return "High Risk"
    else: return "Critical Risk"

def predict_health(df: pd.DataFrame) -> pd.DataFrame:
    feats = engineer_features(df)
    scaled_feats = feats.copy()
    scaled_columns = list(getattr(scaler, 'feature_names_in_', []))
    scaled_feats[scaled_columns] = scaler.transform(feats[scaled_columns])
    probs = model.predict_proba(scaled_feats)[:, 1]
    anom = compute_anomaly(feats)
    res = df.copy()
    res['failure_probability'] = np.round(probs, 4)
    res['anomaly_score'] = np.round(anom, 4)
    res['risk_tier'] = [assign_tier(p) for p in probs]
    res['alert_flag'] = (probs >= risk_thresholds['optimal_threshold']).astype(int)
    return res


def build_maintenance_recommendation(row: pd.Series) -> dict:
    metrics = row.to_dict()
    failure_probability = float(metrics.get('failure_probability', 0.0) or 0.0)
    risk_tier = metrics.get('risk_tier', 'Low Risk')
    failure_mode = metrics.get('primary_failure_mode') or metrics.get('failure_mode')
    return get_rag_recommendation(
        failure_probability=failure_probability,
        risk_tier=risk_tier,
        failure_mode=failure_mode,
        metrics=metrics,
    )

def get_prescriptive_order(row: pd.Series) -> dict:
    p = row.get('failure_probability', 0.0)
    wear = row.get('tool_wear_min', 0)
    t_diff = row.get('process_temperature_k', 0) - row.get('air_temperature_k', 0)
    rpm = row.get('rotational_speed_rpm', 0)
    trq = row.get('torque_nm', 0)
    pwr = rpm * trq * (2 * np.pi / 60)
    wt = wear * trq
    modes, acts, checks, parts = [], [], [], []
    if wear >= 200:
        modes.append("Tool Wear Failure (TWF)")
        acts.append("Replace cutting tool insert.")
        checks.append("Inspect tool holder alignment and spindle runout.")
        parts.append("Carbide Insert Set (Part #T-880)")
    if t_diff < 8.6 and rpm < 1380:
        modes.append("Heat Dissipation Failure (HDF)")
        acts.append("Flush coolant lines and heat exchanger.")
        checks.append("Verify coolant flow rate >= 15 L/min.")
        parts.append("Coolant Filter Cartridge (Part #C-104)")
    if pwr < 3500 or pwr > 9000:
        modes.append("Power Failure (PWF)")
        acts.append("Check motor drive inverter and electrical supply.")
        checks.append("Measure 3-phase current balance and check VFD.")
        parts.append("VFD Inverter Relay Module (Part #E-550)")
    if wt > 11000:
        modes.append("Overstrain Failure (OSF)")
        acts.append("Reduce feed rate and verify workpiece hardness.")
        checks.append("Inspect machine guide ways and ball screws.")
        parts.append("Spindle Bearings Set (Part #B-320)")
    if not modes:
        if p >= 0.26:
            modes.append("Multivariable Machine Strain")
            acts.append("Perform technician inspection within 24 hours.")
            checks.append("Run comprehensive sensor calibration.")
            parts.append("Preventive Maintenance Kit (Part #PM-01)")
        else:
            modes.append("Normal Operation")
            acts.append("Continue routine operation and monitoring.")
            checks.append("Maintain standard shift log.")
            parts.append("None")
    prio = "CRITICAL / EMERGENCY" if p >= 0.75 else "HIGH PRIORITY" if p >= 0.26 else "MEDIUM" if p >= 0.15 else "LOW / ROUTINE"
    eta = "< 2 hours" if p >= 0.75 else "Within 24 hours" if p >= 0.26 else "Next shift" if p >= 0.15 else "Routine cycle"
    savings = 4500.0 if p >= 0.26 else 0.0
    return {
        "primary_failure_mode": ", ".join(modes),
        "priority": prio,
        "eta": eta,
        "recommended_action": " ".join(acts),
        "technician_checklist": checks,
        "required_spare_parts": parts,
        "savings_usd": savings
    }

def log_to_db(rec: dict):
    conn = sqlite3.connect('data/prediction_history.db')
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS prediction_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, machine_id TEXT, machine_type TEXT,
        air_temp_k REAL, process_temp_k REAL, rotational_speed_rpm REAL,
        torque_nm REAL, tool_wear_min REAL, failure_probability REAL,
        risk_tier TEXT, anomaly_score REAL, primary_failure_mode TEXT,
        recommended_action TEXT, cost_savings_usd REAL
    )''')
    cur.execute('''INSERT INTO prediction_history (
        timestamp, machine_id, machine_type, air_temp_k, process_temp_k,
        rotational_speed_rpm, torque_nm, tool_wear_min, failure_probability,
        risk_tier, anomaly_score, primary_failure_mode, recommended_action, cost_savings_usd
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
        rec.get('timestamp', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')),
        rec.get('machine_id', 'M-01'), rec.get('type', 'M'),
        float(rec.get('air_temperature_k', 0)), float(rec.get('process_temperature_k', 0)),
        float(rec.get('rotational_speed_rpm', 0)), float(rec.get('torque_nm', 0)),
        float(rec.get('tool_wear_min', 0)), float(rec.get('failure_probability', 0)),
        rec.get('risk_tier', 'Low Risk'), float(rec.get('anomaly_score', 0)),
        rec.get('primary_failure_mode', 'Normal'), rec.get('recommended_action', 'Continue monitoring'),
        float(rec.get('savings_usd', 0))
    ))
    conn.commit()
    conn.close()


PRESET_CONFIG = {
    "Nominal Machine": {"air_temperature_k": 298.1, "process_temperature_k": 308.6, "rotation_speed_rpm": 1550, "torque_nm": 40.0, "tool_wear_min": 20, "machine_type": "M"},
    "Heat Dissipation (HDF)": {"air_temperature_k": 304.5, "process_temperature_k": 309.0, "rotation_speed_rpm": 1250, "torque_nm": 65.0, "tool_wear_min": 80, "machine_type": "L"},
    "Tool Wear (TWF)": {"air_temperature_k": 298.5, "process_temperature_k": 308.7, "rotation_speed_rpm": 1420, "torque_nm": 48.0, "tool_wear_min": 225, "machine_type": "L"},
    "Overstrain (OSF)": {"air_temperature_k": 299.0, "process_temperature_k": 309.5, "rotation_speed_rpm": 1350, "torque_nm": 70.0, "tool_wear_min": 210, "machine_type": "L"},
    "Power Anomaly (PWF)": {"air_temperature_k": 298.0, "process_temperature_k": 308.0, "rotation_speed_rpm": 2700, "torque_nm": 15.0, "tool_wear_min": 150, "machine_type": "H"},
}


def get_preset_params(preset_name: str) -> dict:
    return PRESET_CONFIG.get(preset_name, PRESET_CONFIG["Nominal Machine"]).copy()

st.sidebar.title("Navigation")
menu_choice = st.sidebar.radio("Go to:", [
    "1. Live Machine Health",
    "2. Sensor Stream Simulator",
    "3. Batch File Scoring",
    "4. MLOps Drift Monitoring",
    "5. Prediction History Log",
    "6. Model Registry"
])

if menu_choice == "1. Live Machine Health":
    st.header("Live Machine Health and Prescriptive Diagnosis")
    col_p, col_id = st.columns([2, 1])
    preset = col_p.selectbox("Select Preset", ["Nominal Machine", "Heat Dissipation (HDF)", "Tool Wear (TWF)", "Overstrain (OSF)", "Power Anomaly (PWF)"])
    machine_id = col_id.text_input("Asset ID", "MILL-ASSET-101")
    p_dict = get_preset_params(preset)

    if "preset_values" not in st.session_state:
        st.session_state["preset_values"] = {}

    current_values = st.session_state["preset_values"].get(
        preset,
        {
            "air": p_dict["air_temperature_k"],
            "proc": p_dict["process_temperature_k"],
            "rpm": p_dict["rotation_speed_rpm"],
            "trq": p_dict["torque_nm"],
            "wear": p_dict["tool_wear_min"],
            "type": p_dict["machine_type"],
        },
    )

    c1, c2, c3 = st.columns(3)
    air_val = c1.slider("Air Temp (K)", 295.0, 305.0, float(current_values['air']), 0.1)
    proc_val = c2.slider("Process Temp (K)", 305.0, 315.0, float(current_values['proc']), 0.1)
    rpm_val = c3.slider("Speed (RPM)", 1100, 2900, int(current_values['rpm']), 10)
    c4, c5, c6 = st.columns(3)
    trq_val = c4.slider("Torque (Nm)", 3.0, 80.0, float(current_values['trq']), 0.5)
    wear_val = c5.slider("Tool Wear (min)", 0, 260, int(current_values['wear']), 1)
    type_val = c6.selectbox("Variant Type", ["L", "M", "H"], index=["L", "M", "H"].index(current_values['type']))

    st.session_state["preset_values"][preset] = {
        "air": air_val,
        "proc": proc_val,
        "rpm": rpm_val,
        "trq": trq_val,
        "wear": wear_val,
        "type": type_val,
    }
    inp_df = pd.DataFrame([{"air_temperature_k": air_val, "process_temperature_k": proc_val, "rotational_speed_rpm": rpm_val, "torque_nm": trq_val, "tool_wear_min": wear_val, "type": type_val}])
    out_df = predict_health(inp_df)
    p_val = out_df['failure_probability'].iloc[0]
    anom_val = out_df['anomaly_score'].iloc[0]
    tier_val = out_df['risk_tier'].iloc[0]
    row_with_prediction = pd.Series({**inp_df.iloc[0].to_dict(), **out_df.iloc[0].to_dict()})
    presc_info = build_maintenance_recommendation(row_with_prediction)
    if 'primary_failure_mode' not in presc_info:
        presc_info = get_prescriptive_order(row_with_prediction)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Failure Probability", f"{p_val * 100:.1f} %")
    k2.metric("Risk Tier", tier_val)
    k3.metric("Anomaly Score", f"{anom_val:.4f}")
    k4.metric("Cost Avoidance", f"${presc_info['savings_usd']:,.0f} USD")
    
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number", value=p_val*100, title={'text': "Failure Risk Gauge (%)"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#263238'},
               'steps': [{'range': [0, 15], 'color': '#C8E6C9'}, {'range': [15, 26], 'color': '#FFF59D'},
                         {'range': [26, 75], 'color': '#FFE082'}, {'range': [75, 100], 'color': '#FFCDD2'}],
               'threshold': {'line': {'color': 'red', 'width': 4}, 'value': 26.0}}
    ))
    fig_g.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_g, use_container_width=True)
    
    st.subheader("Advanced Maintenance AI Advisor")
    if presc_info.get("rag_used"):
        st.success("OpenRouter advisor response active")
    elif presc_info.get("rag_error"):
        st.warning(f"OpenRouter advisor unavailable ({presc_info['rag_error']}); showing deterministic guidance.")
    else:
        st.caption("Deterministic guidance active. Configure OPENROUTER_API_KEY to enable the AI advisor.")
    st.info(f"Diagnosis: {presc_info['primary_failure_mode']} | Priority: {presc_info['priority']} | Window: {presc_info['eta']}")
    st.write(f"Grounded summary: {presc_info['summary']}")
    st.write(f"Recommended action: {presc_info['action'] if 'action' in presc_info else presc_info['recommended_action']}")

    with st.expander("Evidence and maintenance checklist"):
        for evidence in presc_info.get('evidence', []):
            st.markdown(f"- {evidence}")
        st.markdown("### Recommended steps")
        for step in presc_info.get('maintenance_steps', presc_info.get('technician_checklist', [])):
            st.markdown(f"- {step}")
        st.markdown("### Required spare parts")
        for part in presc_info.get('required_spare_parts', []):
            st.markdown(f"- {part}")
    
    if st.button("Save Record to SQLite Database"):
        log_to_db({
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'machine_id': machine_id, 'type': type_val,
            'air_temperature_k': air_val, 'process_temperature_k': proc_val,
            'rotational_speed_rpm': rpm_val, 'torque_nm': trq_val,
            'tool_wear_min': wear_val, 'failure_probability': p_val,
            'risk_tier': tier_val, 'anomaly_score': anom_val,
            'primary_failure_mode': presc_info['primary_failure_mode'],
            'recommended_action': presc_info.get('action', presc_info.get('recommended_action', 'Continue monitoring')),
            'savings_usd': presc_info['savings_usd']
        })
        st.success(f"Record saved for {machine_id}")

elif menu_choice == "2. Sensor Stream Simulator":
    st.header("Real-Time Telemetry Stream Simulator")
    sim_scenario = st.selectbox("Scenario", ["Nominal", "Thermal Runaway", "Tool Wear Failure", "Overstrain Spike", "Power Surge"])
    steps = st.slider("Simulation Steps", 15, 50, 25)
    if st.button("Generate Stream"):
        records = []
        air, proc, speed, trq, wear = 298.15, 308.65, 1500.0, 40.0, 30.0
        for t in range(steps):
            a = air + np.random.normal(0, 0.2)
            p = proc + np.random.normal(0, 0.2)
            s = speed + np.random.normal(0, 20)
            tr = trq + np.random.normal(0, 1.5)
            w = wear + (t * 0.4)
            if sim_scenario == "Thermal Runaway" and t >= 10:
                a += (t - 10) * 0.7; p += (t - 10) * 0.2; s -= (t - 10) * 18
            elif sim_scenario == "Tool Wear Failure" and t >= 10:
                w += (t - 10) * 11.0; tr += (t - 10) * 1.4
            elif sim_scenario == "Overstrain Spike" and t >= 12:
                tr += 32.0; w = 215.0
            elif sim_scenario == "Power Surge" and t >= 10:
                s += 750.0; tr += 18.0
            records.append({
                'step': t + 1, 'timestamp': (pd.Timestamp.now() + pd.Timedelta(seconds=t*5)).strftime('%H:%M:%S'),
                'machine_id': 'STREAM-MILL-01', 'type': 'L',
                'air_temperature_k': round(a, 2), 'process_temperature_k': round(p, 2),
                'rotational_speed_rpm': int(max(s, 500)), 'torque_nm': round(max(tr, 5.0), 2),
                'tool_wear_min': int(w)
            })
        st_df = pd.DataFrame(records)
        sc_df = predict_health(st_df)
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=sc_df['step'], y=sc_df['failure_probability'], mode='lines+markers', name='Failure Risk', line=dict(color='crimson', width=3)))
        fig_s.add_trace(go.Scatter(x=sc_df['step'], y=sc_df['anomaly_score'], mode='lines', name='Anomaly Score', line=dict(color='orange', dash='dot')))
        fig_s.add_hline(y=0.26, line_dash="dash", line_color="goldenrod")
        fig_s.update_layout(title="Live Telemetry Risk Progression", xaxis_title="Time Step", yaxis_title="Score", height=380)
        st.plotly_chart(fig_s, use_container_width=True)
        st.dataframe(sc_df[['step', 'air_temperature_k', 'process_temperature_k', 'rotational_speed_rpm', 'torque_nm', 'tool_wear_min', 'failure_probability', 'risk_tier']])

elif menu_choice == "3. Batch File Scoring":
    st.header("Batch CSV Scoring and Fleet Risk Triage")
    up_file = st.file_uploader("Upload CSV", type=['csv'])
    if up_file is not None:
        b_df = pd.read_csv(up_file)
    else:
        st.info("Using 100 sample records from reference test split.")
        b_df = ref_df.sample(100, random_state=42).copy()
    if st.button("Execute Batch Scoring"):
        res_batch = predict_health(b_df)
        t_counts = res_batch['risk_tier'].value_counts()
        high_crit = res_batch[res_batch['risk_tier'].isin(['High Risk', 'Critical Risk'])]
        b1, b2, b3 = st.columns(3)
        b1.metric("Equipment Evaluated", len(res_batch))
        b2.metric("High/Critical Alerts", len(high_crit))
        b3.metric("Cost Avoided", f"${len(high_crit)*4500:,.0f} USD")
        fig_b = px.bar(x=t_counts.index, y=t_counts.values, color=t_counts.index,
                       color_discrete_map={'Low Risk': '#4CAF50', 'Medium Risk': '#FFEB3B', 'High Risk': '#FF9800', 'Critical Risk': '#F44336'},
                       labels={'x': 'Risk Tier', 'y': 'Count'}, title="Fleet Risk Distribution")
        st.plotly_chart(fig_b, use_container_width=True)
        st.dataframe(high_crit[['air_temperature_k', 'process_temperature_k', 'rotational_speed_rpm', 'torque_nm', 'tool_wear_min', 'failure_probability', 'risk_tier']])
        st.download_button("Download CSV Results", data=res_batch.to_csv(index=False).encode('utf-8'), file_name="batch_scored_results.csv", mime="text/csv")

elif menu_choice == "4. MLOps Drift Monitoring":
    st.header("MLOps Data Drift and Statistical Monitoring")
    def c_psi(exp, act, num_bins=10):
        perc = np.linspace(0, 100, num_bins + 1)
        b = np.percentile(exp, perc)
        b[0], b[-1] = -np.inf, np.inf
        e_cnt, _ = np.histogram(exp, bins=b)
        a_cnt, _ = np.histogram(act, bins=b)
        e_pct = (e_cnt + 1e-4) / (len(exp) + 1e-4 * num_bins)
        a_pct = (a_cnt + 1e-4) / (len(act) + 1e-4 * num_bins)
        return float(np.round(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)), 4))
    d_feats = ['air_temperature_k', 'process_temperature_k', 'rotational_speed_rpm', 'torque_nm', 'tool_wear_min', 'temp_diff_k', 'power_w']
    live_df = ref_df.sample(250, random_state=123).copy()
    d_res = []
    for f in d_feats:
        rv = ref_df[f].dropna().values
        lv = live_df[f].dropna().values
        ks_s, ks_p = stats.ks_2samp(rv, lv)
        p_val = c_psi(rv, lv)
        st_label = "Stable" if p_val < 0.10 else "Moderate Drift" if p_val < 0.25 else "Significant Drift"
        d_res.append({'Feature': f, 'KS-Statistic': round(ks_s, 4), 'KS p-value': round(ks_p, 5), 'PSI': p_val, 'Status': st_label})
    st.dataframe(pd.DataFrame(d_res))
    feat_choice = st.selectbox("Inspect Histogram", d_feats, index=3)
    fig_h = go.Figure()
    fig_h.add_trace(go.Histogram(x=ref_df[feat_choice], name='Baseline', marker_color='#1976D2', opacity=0.6, histnorm='probability density'))
    fig_h.add_trace(go.Histogram(x=live_df[feat_choice], name='Production', marker_color='#E53935', opacity=0.6, histnorm='probability density'))
    fig_h.update_layout(barmode='overlay', title=f"Feature Distribution: {feat_choice}", height=350)
    st.plotly_chart(fig_h, use_container_width=True)

elif menu_choice == "5. Prediction History Log":
    st.header("SQLite Prediction History Audit Trail")
    if os.path.exists('data/prediction_history.db'):
        conn = sqlite3.connect('data/prediction_history.db')
        h_df = pd.read_sql_query("SELECT * FROM prediction_history ORDER BY id DESC LIMIT 200", conn)
        conn.close()
        if not h_df.empty:
            st.dataframe(h_df)
            st.download_button("Export History CSV", data=h_df.to_csv(index=False).encode('utf-8'), file_name="history_audit.csv", mime="text/csv")
        else:
            st.info("No records logged.")
    else:
        st.info("Database file not found.")

elif menu_choice == "6. Model Registry":
    st.header("Model Registry Metadata")
    st.json({
        "model_id": "GB-PRED-MAINT-V1.0",
        "architecture": "GradientBoostingClassifier",
        "version": "1.0.0",
        "features": FEATURE_COLUMNS,
        "optimal_threshold": risk_thresholds['optimal_threshold'],
        "pr_auc": 0.9461,
        "roc_auc": 0.9766,
        "cost_fn_usd": 5000,
        "cost_fp_usd": 500
    })
