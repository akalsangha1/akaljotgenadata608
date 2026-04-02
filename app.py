"""
PulsePoint – Cardiovascular Risk Prediction Dashboard
Upgraded UI: gauge, feature importance, population comparison, session history
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import math
import time
import uuid
from datetime import datetime

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Page config
st.set_page_config(
    page_title="PulsePoint – Heart Risk AI",
    page_icon="💓",
    layout="wide",
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 40%, #f9ca24 100%);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 28px;
    color: white;
    box-shadow: 0 8px 32px rgba(238,90,36,0.25);
}
.hero h1 {
    font-family: 'Nunito', sans-serif;
    font-size: 2.6rem;
    font-weight: 900;
    margin: 0 0 6px 0;
    letter-spacing: -1px;
}
.hero p {
    font-size: 1.05rem;
    opacity: 0.92;
    margin: 0;
}

/* Stat cards */
.stat-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 140px;
    background: white;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-top: 4px solid #ff6b6b;
    text-align: center;
}
.stat-card .val {
    font-family: 'Nunito', sans-serif;
    font-size: 1.8rem;
    font-weight: 900;
    color: #ee5a24;
}
.stat-card .lbl {
    font-size: 0.78rem;
    color: #888;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Section cards */
.card {
    background: white;
    border-radius: 18px;
    padding: 24px 28px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}
.card h3 {
    font-family: 'Nunito', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: #2d3436;
    margin: 0 0 16px 0;
}

/* Risk result boxes */
.result-safe {
    background: linear-gradient(135deg, #00b894, #00cec9);
    border-radius: 16px; padding: 24px 28px; color: white;
    box-shadow: 0 6px 24px rgba(0,184,148,0.3);
}
.result-risk {
    background: linear-gradient(135deg, #ff6b6b, #ee5a24);
    border-radius: 16px; padding: 24px 28px; color: white;
    box-shadow: 0 6px 24px rgba(238,90,36,0.3);
}
.result-safe h2, .result-risk h2 {
    font-family: 'Nunito', sans-serif;
    font-size: 1.6rem; font-weight: 900; margin: 0 0 8px 0;
}
.result-safe p, .result-risk p { margin: 0; opacity: 0.92; font-size: 1rem; }

/* Gauge container */
.gauge-wrap { text-align: center; padding: 10px 0; }

/* History table */
.hist-safe { color: #00b894; font-weight: 700; }
.hist-risk { color: #ee5a24; font-weight: 700; }

/* Sidebar */
[data-testid="stSidebar"] { background: #fff8f5; }

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }

/* Input labels */
.stSelectbox label, .stNumberInput label {
    font-weight: 600 !important;
    color: #2d3436 !important;
    font-size: 0.88rem !important;
}

/* Button */
.stFormSubmitButton button {
    background: linear-gradient(135deg, #ff6b6b, #ee5a24) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
    padding: 14px !important;
    box-shadow: 0 4px 16px rgba(238,90,36,0.3) !important;
    transition: transform 0.15s !important;
}
.stFormSubmitButton button:hover { transform: translateY(-2px) !important; }
</style>
""", unsafe_allow_html=True)

# Load model artifacts (cached per session)
ARTIFACT_DIR = os.environ.get("MODEL_ARTIFACT_DIR", "./model_artifacts")

@st.cache_resource
def load_artifacts():
    scaler = joblib.load(f"{ARTIFACT_DIR}/scaler.pkl")
    pca    = joblib.load(f"{ARTIFACT_DIR}/pca.pkl")
    model  = joblib.load(f"{ARTIFACT_DIR}/logistic_model.pkl")
    with open(f"{ARTIFACT_DIR}/model_metadata.json") as f:
        meta = json.load(f)
    return scaler, pca, model, meta

try:
    scaler, pca, model, meta = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    load_error = str(e)

# Session state — history log
# S3 results writer ────────────────────────────────────────────────────────
RESULTS_BUCKET = os.environ.get("S3_BUCKET", "pulsepoint-raw-zone-akaljotmena")
RESULTS_PREFIX = "results/"

def save_prediction_to_s3(record: dict):
    """Write a prediction record to S3 Results Zone. Fails silently if unavailable."""
    try:
        s3 = boto3.client("s3")
        key = f"{RESULTS_PREFIX}{record['prediction_id']}.json"
        s3.put_object(
            Bucket=RESULTS_BUCKET,
            Key=key,
            Body=json.dumps(record, indent=2),
            ContentType="application/json"
        )
    except (BotoCoreError, ClientError):
        pass

# Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "latencies" not in st.session_state:
    st.session_state.latencies = []

# Reference stats from UCI Cleveland dataset for percentile comparison
POP_STATS = {
    "age":      {"mean": 54.4, "std": 9.0,   "min": 29, "max": 77},
    "trestbps": {"mean": 131.7,"std": 17.6,  "min": 94, "max": 200},
    "chol":     {"mean": 246.7,"std": 51.8,  "min": 126,"max": 564},
    "thalach":  {"mean": 149.6,"std": 22.9,  "min": 71, "max": 202},
    "oldpeak":  {"mean": 1.04, "std": 1.16,  "min": 0,  "max": 6.2},
}
POP_LABELS = {
    "age": "Age", "trestbps": "Resting BP",
    "chol": "Cholesterol", "thalach": "Max Heart Rate", "oldpeak": "ST Depression"
}

# SVG Gauge
def make_gauge(pct):
    pct = max(0, min(100, pct))
    # Map pct to angle: 0% = -180deg (left), 100% = 0deg (right)
    angle = -180 + (pct / 100) * 180
    rad   = math.radians(angle)
    cx, cy, r = 120, 110, 80
    nx = cx + r * math.cos(rad)
    ny = cy + r * math.sin(rad)

    if pct < 30:
        color = "#00b894"
        label = "LOW RISK"
    elif pct < 60:
        color = "#f9ca24"
        label = "MODERATE"
    else:
        color = "#ee5a24"
        label = "HIGH RISK"

    return f"""
    <svg viewBox="0 0 240 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:280px;display:block;margin:0 auto;">
      <defs>
        <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%"   stop-color="#00b894"/>
          <stop offset="50%"  stop-color="#f9ca24"/>
          <stop offset="100%" stop-color="#ee5a24"/>
        </linearGradient>
      </defs>
      <!-- Background arc -->
      <path d="M 40 110 A 80 80 0 0 1 200 110" fill="none" stroke="#f0f0f0" stroke-width="18" stroke-linecap="round"/>
      <!-- Colored arc -->
      <path d="M 40 110 A 80 80 0 0 1 200 110" fill="none" stroke="url(#arcGrad)" stroke-width="18" stroke-linecap="round"/>
      <!-- Needle -->
      <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}"
            stroke="{color}" stroke-width="4" stroke-linecap="round"/>
      <circle cx="{cx}" cy="{cy}" r="6" fill="{color}"/>
      <!-- Labels -->
      <text x="36" y="126" font-size="9" fill="#aaa" font-family="DM Sans,sans-serif">0%</text>
      <text x="191" y="126" font-size="9" fill="#aaa" font-family="DM Sans,sans-serif">100%</text>
      <!-- Center value -->
      <text x="{cx}" y="96" text-anchor="middle" font-size="22"
            font-family="Nunito,sans-serif" font-weight="900" fill="{color}">{pct:.0f}%</text>
      <text x="{cx}" y="112" text-anchor="middle" font-size="9"
            font-family="DM Sans,sans-serif" font-weight="600" fill="{color}" letter-spacing="1">{label}</text>
    </svg>
    """

st.markdown("""
<div class="hero">
  <h1>💓 PulsePoint</h1>
  <p>AI-powered cardiovascular risk prediction · Trained on UCI Cleveland Heart Disease Dataset · <strong>303 patients · 13 clinical features</strong></p>
</div>
""", unsafe_allow_html=True)

if not artifacts_loaded:
    st.error(f"Could not load model artifacts: {load_error}")
    st.stop()

f1     = meta.get('f1_score', 0)
auc    = meta.get('roc_auc', 0)
n_pc   = meta.get('n_pca_components', 0)
var    = meta.get('pca_variance_retained', 0)
n_pred = len(st.session_state.history)
avg_lat_str = (f"{sum(st.session_state.latencies)/len(st.session_state.latencies):.0f} ms"
               if st.session_state.latencies else "—")

st.markdown(f"""
<div class="stat-row">
  <div class="stat-card"><div class="val">{f1:.2f}</div><div class="lbl">F1 Score</div></div>
  <div class="stat-card"><div class="val">{auc:.2f}</div><div class="lbl">ROC-AUC</div></div>
  <div class="stat-card"><div class="val">{n_pc}</div><div class="lbl">PCA Components</div></div>
  <div class="stat-card"><div class="val">{var:.0%}</div><div class="lbl">Variance Retained</div></div>
  <div class="stat-card"><div class="val">{n_pred}</div><div class="lbl">Predictions Made</div></div>
  <div class="stat-card"><div class="val">{avg_lat_str}</div><div class="lbl">Avg Latency</div></div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 💡 Feature Guide")
    st.markdown("""
    | Feature | Description |
    |---|---|
    | **CP** | Chest pain type (1–4) |
    | **FBS** | Fasting blood sugar >120 mg/dl |
    | **RestECG** | Resting ECG result |
    | **Exang** | Exercise-induced angina |
    | **Oldpeak** | ST depression (exercise) |
    | **Slope** | Slope of peak ST segment |
    | **CA** | Major vessels (fluoroscopy) |
    | **Thal** | Thalassemia type |
    """)
    st.divider()
    st.caption("⚕️ For educational purposes only. Not a substitute for medical advice.")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

col_form, col_results = st.columns([1, 1.1], gap="large")

FEATURE_COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

with col_form:
    st.markdown("### 🩺 Patient Measurements")
    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            age      = st.number_input("Age", 1, 120, 54)
            sex      = st.selectbox("Sex", [("Male",1),("Female",0)], format_func=lambda x: x[0])
            trestbps = st.number_input("Resting BP (mmHg)", 80, 250, 130)
            chol     = st.number_input("Cholesterol (mg/dl)", 100, 600, 246)
            thalach  = st.number_input("Max Heart Rate", 60, 220, 150)
            oldpeak  = st.number_input("ST Depression", 0.0, 10.0, 1.0, 0.1)
        with c2:
            cp      = st.selectbox("Chest Pain (CP)", [1,2,3,4],
                        format_func=lambda x: {1:"1–Typical Angina",2:"2–Atypical",3:"3–Non-Anginal",4:"4–Asymptomatic"}[x])
            fbs     = st.selectbox("Fasting BS >120", [(False,0),(True,1)], format_func=lambda x: "Yes" if x[1] else "No")
            restecg = st.selectbox("Resting ECG", [0,1,2],
                        format_func=lambda x: {0:"0–Normal",1:"1–ST-T Abnorm",2:"2–LV Hypertrophy"}[x])
            exang   = st.selectbox("Exercise Angina", [(False,0),(True,1)], format_func=lambda x: "Yes" if x[1] else "No")
            slope   = st.selectbox("ST Slope", [1,2,3],
                        format_func=lambda x: {1:"1–Upsloping",2:"2–Flat",3:"3–Downsloping"}[x])
            ca      = st.selectbox("Major Vessels (0–3)", [0,1,2,3])
            thal    = st.selectbox("Thalassemia", [3,6,7],
                        format_func=lambda x: {3:"3–Normal",6:"6–Fixed Defect",7:"7–Reversible"}[x])

        submitted = st.form_submit_button("🔍 Analyse Risk", use_container_width=True, type="primary")

with col_results:
    if submitted:
        values = {
            'age': age, 'sex': sex[1], 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs[1], 'restecg': restecg, 'thalach': thalach,
            'exang': exang[1], 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }
        # Time each pipeline stage ──────────────────────────────────────────
        t_start  = time.perf_counter()
        X_input  = np.array([[values[f] for f in FEATURE_COLS]])
        X_scaled = scaler.transform(X_input)
        t_pca_start = time.perf_counter()
        X_pca    = pca.transform(X_scaled)
        t_pred_start = time.perf_counter()
        pred     = model.predict(X_pca)[0]
        proba    = model.predict_proba(X_pca)[0]
        t_end    = time.perf_counter()

        risk_pct       = proba[1] * 100
        latency_ms     = (t_end - t_start) * 1000
        latency_scale  = (t_pca_start - t_start) * 1000
        latency_pca    = (t_pred_start - t_pca_start) * 1000
        latency_lr     = (t_end - t_pred_start) * 1000
        st.session_state.latencies.append(latency_ms)

        # Build prediction record for S3 ──────────────────────────────────────
        pred_record = {
            "prediction_id":  str(uuid.uuid4()),
            "timestamp":      datetime.now().isoformat(),
            "inputs":         values,
            "risk_pct":       round(risk_pct, 2),
            "prediction":     int(pred),
            "result_label":   "At Risk" if pred == 1 else "No Risk",
            "latency_ms":     round(latency_ms, 3),
            "model_version":  meta.get("model_type", "unknown"),
        }

        # Write to S3 results zone ───────────────────────────────────────────
        save_prediction_to_s3(pred_record)

        # Append to session history ───────────────────────────────────────────
        st.session_state.history.append({
            "Time":       datetime.now().strftime("%H:%M:%S"),
            "Age":        age,
            "Sex":        "M" if sex[1] == 1 else "F",
            "Chol":       chol,
            "Max HR":     thalach,
            "Risk %":     f"{risk_pct:.1f}%",
            "Result":     "⚠️ At Risk" if pred == 1 else "✅ No Risk",
            "Latency":    f"{latency_ms:.1f} ms"
        })

        if pred == 1:
            st.markdown(f"""
            <div class="result-risk">
              <h2>⚠️ Cardiovascular Risk Detected</h2>
              <p>This patient shows elevated cardiovascular risk based on the provided measurements.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
              <h2>✅ No Significant Risk</h2>
              <p>This patient's measurements suggest no significant cardiovascular risk at this time.</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        g1, g2 = st.columns(2)

        with g1:
            st.markdown('<div class="card"><h3>🎯 Risk Score</h3>', unsafe_allow_html=True)
            st.markdown(make_gauge(risk_pct), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with g2:
            st.markdown('<div class="card"><h3>🔑 Key Drivers</h3>', unsafe_allow_html=True)
            # Feature importance: use LR coefficients projected back through PCA
            coef_pca    = model.coef_[0]
            coef_orig   = pca.components_.T @ coef_pca
            feat_imp    = pd.DataFrame({
                "Feature":    FEATURE_COLS,
                "Importance": np.abs(coef_orig)
            }).sort_values("Importance", ascending=True).tail(6)

            feat_chart = feat_imp.set_index("Feature")
            st.bar_chart(feat_chart, color="#ee5a24", height=200)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><h3>👥 Population Comparison</h3>', unsafe_allow_html=True)
        patient_vals = {"age": age, "trestbps": trestbps, "chol": chol,
                        "thalach": thalach, "oldpeak": oldpeak}

        pop_rows = []
        for key, label in POP_LABELS.items():
            pv   = patient_vals[key]
            mean = POP_STATS[key]["mean"]
            std  = POP_STATS[key]["std"]
            z    = (pv - mean) / std
            pct_rank = min(100, max(0, (0.5 + 0.5 * math.erf(z / math.sqrt(2))) * 100))
            pop_rows.append({
                "Feature": label,
                "You": pv,
                "Avg (UCI)": mean,
                "Your Percentile": f"{pct_rank:.0f}th"
            })

        pop_df = pd.DataFrame(pop_rows).set_index("Feature")
        st.dataframe(pop_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><h3>⚡ Inference Latency</h3>', unsafe_allow_html=True)
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("Total", f"{latency_ms:.1f} ms",
                   delta="✅ <5s target" if latency_ms < 5000 else "⚠️ >5s target")
        lc2.metric("Scaling", f"{latency_scale:.2f} ms")
        lc3.metric("PCA", f"{latency_pca:.2f} ms")
        lc4.metric("LR Predict", f"{latency_lr:.2f} ms")
        if len(st.session_state.latencies) > 1:
            avg_lat = sum(st.session_state.latencies) / len(st.session_state.latencies)
            st.caption(f"Session average: **{avg_lat:.1f} ms** over {len(st.session_state.latencies)} predictions")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color: #bbb;">
          <div style="font-size:4rem;">💓</div>
          <div style="font-family:'Nunito',sans-serif; font-size:1.2rem; font-weight:700; color:#ddd; margin-top:12px;">
            Fill in the patient details and click<br><strong style="color:#ee5a24;">Analyse Risk</strong> to get started
          </div>
        </div>
        """, unsafe_allow_html=True)

if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📋 Session History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
