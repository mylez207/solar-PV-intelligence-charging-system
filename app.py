"""
=============================================================
  ENG. ALFRED KAFARANSA — DIT M.Eng. Dissertation 2026
  Solar PV-Based Charging System for Electric Three-Wheelers
  Professional Streamlit App — Light Blue & White Theme
=============================================================
"""

import streamlit as st
import numpy as np
import joblib
import os
import math

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

st.set_page_config(
    page_title="SolarEV Intelligence | Kafaransa DIT 2026",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

VALID_USERNAME = "Kafaransa"
VALID_PASSWORD = "kafaransa@2026"

BAJAJI_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 220"
     style="opacity:0.06;position:fixed;bottom:30px;right:30px;width:360px;pointer-events:none;z-index:0;">
  <path d="M60 140 Q60 90 110 80 L340 72 Q390 70 420 90 L460 110 L470 140 Z" fill="#1e90ff"/>
  <path d="M110 80 Q130 40 200 36 L310 34 Q360 34 380 72 L110 80 Z" fill="#1565c0"/>
  <path d="M130 78 Q145 48 200 42 L300 40 Q340 40 355 72 Z" fill="#90caf9" opacity="0.5"/>
  <circle cx="130" cy="152" r="28" fill="none" stroke="#1e90ff" stroke-width="8"/>
  <circle cx="130" cy="152" r="10" fill="#1e90ff"/>
  <circle cx="390" cy="152" r="28" fill="none" stroke="#1e90ff" stroke-width="8"/>
  <circle cx="390" cy="152" r="10" fill="#1e90ff"/>
  <rect x="165" y="36" width="130" height="22" rx="3" fill="#0d47a1" opacity="0.8"/>
  <line x1="165" y1="43" x2="295" y2="43" stroke="#0a3070" stroke-width="1"/>
  <line x1="165" y1="50" x2="295" y2="50" stroke="#0a3070" stroke-width="1"/>
  <line x1="205" y1="36" x2="205" y2="58" stroke="#0a3070" stroke-width="1"/>
  <line x1="245" y1="36" x2="245" y2="58" stroke="#0a3070" stroke-width="1"/>
  <path d="M245 95 L235 118 L248 118 L238 142 L258 112 L244 112 Z" fill="#f59e0b" opacity="0.9"/>
  <ellipse cx="260" cy="182" rx="220" ry="10" fill="#1565c0" opacity="0.2"/>
</svg>
"""

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700;800&display=swap');

:root {
    --bg:        #e8f4fd;
    --bg2:       #d1e9f7;
    --bg3:       #ffffff;
    --sidebar:   #1565c0;
    --card:      #ffffff;
    --card2:     #f0f8ff;
    --border:    #b3d4f0;
    --border-hi: #1e90ff;
    --blue:      #1e90ff;
    --blue-dk:   #1565c0;
    --blue-lt:   #64b5f6;
    --sky:       #e3f2fd;
    --teal:      #00acc1;
    --amber:     #f59e0b;
    --red:       #ef5350;
    --green:     #43a047;
    --text-dk:   #0d2137;
    --text-md:   #1e4976;
    --text-lt:   #5a8aaa;
    --white:     #ffffff;
}

html, body, .main, [data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #dbeafe 0%, #e0f2fe 40%, #f0f9ff 100%) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-dk);
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 700px 500px at 15% 10%, rgba(30,144,255,0.08) 0%, transparent 70%),
        radial-gradient(ellipse 500px 400px at 85% 85%, rgba(0,172,193,0.06) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d47a1 0%, #1565c0 50%, #1976d2 100%) !important;
    border-right: none !important;
    box-shadow: 4px 0 20px rgba(21,101,192,0.3) !important;
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stPasswordInput"] input {
    background: #ffffff !important;
    border: 1.5px solid #90caf9 !important;
    border-radius: 10px !important;
    color: #0d2137 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 12px 16px !important;
    font-size: 1rem !important;
    box-shadow: 0 2px 8px rgba(30,144,255,0.08) !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stPasswordInput"] input:focus {
    border-color: #1e90ff !important;
    box-shadow: 0 0 0 3px rgba(30,144,255,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1e90ff, #1565c0) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    padding: 12px 24px !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 14px rgba(30,144,255,0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #42a5f5, #1e90ff) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(30,144,255,0.45) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #1e90ff !important;
    border-color: #1e90ff !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-radius: 12px !important;
    padding: 5px !important;
    gap: 3px !important;
    border: 1.5px solid #b3d4f0 !important;
    box-shadow: 0 2px 10px rgba(30,144,255,0.08) !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: #1e4976 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.87rem !important;
}
[aria-selected="true"] {
    background: linear-gradient(135deg, #1e90ff, #1565c0) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(30,144,255,0.3) !important;
}

/* ── Selectbox ── */
[data-baseweb="select"] > div {
    background: #ffffff !important;
    border-color: #90caf9 !important;
    color: #0d2137 !important;
    border-radius: 10px !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1.5px solid #b3d4f0 !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 8px rgba(30,144,255,0.08) !important;
}
[data-testid="stMetricValue"] { color: #1565c0 !important; font-family: 'Poppins', sans-serif !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #5a8aaa !important; }

hr { border-color: #b3d4f0 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #e8f4fd; }
::-webkit-scrollbar-thumb { background: #90caf9; border-radius: 3px; }

/* ── LOGIN PAGE ── */
.login-wrap {
    max-width: 360px;
    margin: 30px auto 0;
    padding: 32px 32px 24px;
    background: #ffffff;
    border: 1.5px solid #90caf9;
    border-radius: 18px;
    box-shadow: 0 16px 48px rgba(30,144,255,0.15), 0 4px 16px rgba(0,0,0,0.06);
}
.login-label {
    font-size: 0.74rem;
    font-weight: 600;
    color: #5a8aaa;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 14px 0 5px;
    font-family: 'Inter', sans-serif;
}
.login-error {
    background: rgba(239,83,80,0.08);
    border: 1px solid rgba(239,83,80,0.3);
    border-radius: 8px;
    padding: 9px 13px;
    color: #c62828;
    font-size: 0.83rem;
    margin-top: 12px;
    text-align: center;
}
.login-footer {
    text-align: center;
    color: #90caf9;
    font-size: 0.73rem;
    margin-top: 18px;
    line-height: 1.7;
}

/* ── HEADER BANNER ── */
.header-banner {
    background: linear-gradient(135deg, #1565c0 0%, #1976d2 50%, #1e90ff 100%);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(21,101,192,0.25);
}
.header-banner::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.header-title {
    font-family: 'Poppins', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 5px;
    letter-spacing: -0.02em;
}
.header-subtitle { color: rgba(255,255,255,0.75); font-size: 0.84rem; line-height: 1.6; margin: 0; }
.header-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: #ffffff;
    font-size: 0.71rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
    margin-top: 10px;
    letter-spacing: 0.05em;
}

/* ── CARDS ── */
.stat-card {
    background: #ffffff;
    border: 1.5px solid #b3d4f0;
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 2px 10px rgba(30,144,255,0.07);
    height: 100%;
}
.stat-card:hover {
    border-color: #1e90ff;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(30,144,255,0.18);
}
.stat-card-icon { font-size: 1.6rem; margin-bottom: 6px; }
.stat-card-value {
    font-family: 'Poppins', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: #1565c0;
    margin: 3px 0;
    line-height: 1.1;
}
.stat-card-unit { font-size: 0.78rem; color: #00acc1; margin-bottom: 3px; }
.stat-card-label { font-size: 0.73rem; color: #5a8aaa; }

/* ── COMPARE CARD ── */
.compare-card {
    background: #ffffff;
    border: 1.5px solid #b3d4f0;
    border-radius: 14px;
    padding: 18px 16px;
    margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(30,144,255,0.06);
    transition: box-shadow 0.2s, border-color 0.2s;
}
.compare-card:hover { border-color: #64b5f6; box-shadow: 0 4px 16px rgba(30,144,255,0.12); }
.compare-row { display:flex; justify-content:space-between; align-items:center; margin-top:8px; }
.compare-base { color: #ef5350; font-size: 1.1rem; font-weight: 700; font-family: 'Poppins', sans-serif; }
.compare-new  { color: #1565c0; font-size: 1.1rem; font-weight: 700; font-family: 'Poppins', sans-serif; }

/* ── PREDICTION BOX ── */
.pred-box {
    background: linear-gradient(160deg, #e3f2fd, #f0f9ff);
    border: 2px solid #1e90ff;
    border-radius: 16px;
    padding: 26px 22px;
    text-align: center;
    margin: 14px 0;
    box-shadow: 0 4px 20px rgba(30,144,255,0.15);
}
.pred-label { color: #5a8aaa; font-size: 0.76rem; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px; }
.pred-value { font-family: 'Poppins', sans-serif; font-size: 3rem; font-weight: 800; color: #1565c0; line-height: 1.0; }
.pred-unit  { font-size: 0.93rem; color: #00acc1; margin-top: 4px; }

/* ── SECTION HEADER ── */
.sec-head {
    font-family: 'Poppins', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #1565c0;
    border-bottom: 2px solid #b3d4f0;
    padding-bottom: 7px;
    margin: 18px 0 12px;
}

/* ── PANELS ── */
.info-panel {
    background: #e3f2fd;
    border-left: 3px solid #1e90ff;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #1e4976;
    margin: 10px 0;
    line-height: 1.7;
}
.warn-panel {
    background: #fff8e1;
    border-left: 3px solid #f59e0b;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #7c5200;
    margin: 10px 0;
    line-height: 1.7;
}
.danger-panel {
    background: #fce4ec;
    border-left: 3px solid #ef5350;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #7f1d1d;
    margin: 10px 0;
    line-height: 1.7;
}

/* ── PROGRESS BAR ── */
.bar-outer {
    background: #e3f2fd;
    border-radius: 8px;
    height: 16px;
    overflow: hidden;
    margin: 5px 0;
    border: 1px solid #90caf9;
}
.bar-inner { height: 100%; border-radius: 8px; transition: width 0.5s ease; }

/* ── BADGES ── */
.badge-on  { background:#e8f5e9; color:#2e7d32; border:1px solid #81c784; padding:3px 11px; border-radius:20px; font-weight:700; font-size:0.82rem; }
.badge-off { background:#fce4ec; color:#c62828; border:1px solid #ef9a9a; padding:3px 11px; border-radius:20px; font-weight:700; font-size:0.82rem; }
.badge-hi  { background:#e8f5e9; color:#2e7d32; border:1px solid #81c784; padding:3px 11px; border-radius:20px; font-weight:600; font-size:0.82rem; }
.badge-md  { background:#fff8e1; color:#7c5200; border:1px solid #ffd54f; padding:3px 11px; border-radius:20px; font-weight:600; font-size:0.82rem; }
.badge-lo  { background:#fce4ec; color:#c62828; border:1px solid #ef9a9a; padding:3px 11px; border-radius:20px; font-weight:600; font-size:0.82rem; }

/* ── SIM BLOCK ── */
.sim-block {
    background: #ffffff;
    border: 1.5px solid #b3d4f0;
    border-radius: 12px;
    padding: 18px 20px;
    margin: 10px 0;
    box-shadow: 0 2px 8px rgba(30,144,255,0.06);
}
.sim-block h4 {
    font-family: 'Poppins', sans-serif;
    color: #1565c0;
    font-size: 0.8rem;
    margin: 0 0 8px;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    font-weight: 700;
}
.sim-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #e3f2fd;
    font-size: 0.82rem;
}
.sim-row:last-child { border-bottom: none; }
.sim-key { color: #5a8aaa; }
.sim-val { color: #0d2137; font-weight: 600; font-family: 'Poppins', sans-serif; }

/* ── SIDEBAR NAV USER ── */
.nav-user {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 12px;
    padding: 12px 14px;
    margin: 10px 0;
}
.nav-user-name { font-family:'Poppins',sans-serif; color:#ffffff !important; font-size:0.95rem; font-weight:700; }
.nav-user-role { color:rgba(255,255,255,0.6) !important; font-size:0.75rem; }

/* ── LOGOUT ── */
.logout-btn > button {
    background: rgba(239,83,80,0.15) !important;
    border: 1px solid rgba(239,83,80,0.4) !important;
    color: #ffcdd2 !important;
    font-size: 0.8rem !important;
    padding: 8px 14px !important;
    box-shadow: none !important;
}
.logout-btn > button:hover {
    background: rgba(239,83,80,0.25) !important;
    box-shadow: none !important;
    transform: none !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(BAJAJI_SVG, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# AUTHENTICATION
# ─────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "login_error" not in st.session_state:
    st.session_state.login_error = ""

def do_login(u, p):
    if u == VALID_USERNAME and p == VALID_PASSWORD:
        st.session_state.authenticated = True
        st.session_state.login_error = ""
        st.rerun()
    else:
        st.session_state.login_error = "Incorrect username or password."

if not st.session_state.authenticated:
    st.markdown("<style>[data-testid='stSidebar']{display:none}</style>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding:36px 0 0;">
        <div style="font-size:3rem; margin-bottom:8px;
                    filter:drop-shadow(0 0 16px rgba(30,144,255,0.6));">☀️</div>
        <div style="font-family:'Poppins',sans-serif; font-size:1.4rem; font-weight:700;
                    color:#1565c0; letter-spacing:-0.01em;">Solar PV Charging System</div>
        <div style="color:#90caf9; font-size:0.78rem; margin-top:3px; letter-spacing:0.08em;">
            DIT · M.ENG. SUSTAINABLE ENERGY · 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 1.1, 1])
    with center:
        st.markdown("""<div class="login-wrap">""", unsafe_allow_html=True)
        st.markdown('<p class="login-label" style="margin-top:0;">USERNAME</p>', unsafe_allow_html=True)
        username_input = st.text_input("u", label_visibility="collapsed", placeholder="Enter your username")
        st.markdown('<p class="login-label">PASSWORD</p>', unsafe_allow_html=True)
        password_input = st.text_input("p", type="password", label_visibility="collapsed", placeholder="Enter your password")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Sign In  →", use_container_width=True):
            do_login(username_input, password_input)
        if st.session_state.login_error:
            st.markdown(f'<div class="login-error">⚠  {st.session_state.login_error}</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="login-footer">
            🔒 Research access only · DIT 2026<br>
            Supervisors: Prof. K.A. Greyson · Dr. P.V. Chombo
        </div></div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def soc_color(v):
    return "#1565c0" if v >= 60 else "#f59e0b" if v >= 30 else "#ef5350"

def soc_badge(v):
    if v >= 60: return f'<span class="badge-hi">🟢 GOOD &nbsp;{v:.1f}%</span>'
    elif v >= 30: return f'<span class="badge-md">🟡 MODERATE &nbsp;{v:.1f}%</span>'
    else: return f'<span class="badge-lo">🔴 CRITICAL &nbsp;{v:.1f}%</span>'

# ─────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️  Loading LSTM models…")
def load_models():
    mdls = {}
    base = "models"
    if not TF_AVAILABLE:
        return mdls
    specs = {
        "soc":    ("model1_SOC.h5",            "model1_SOC.pkl",            "scaler_X.pkl"),
        "dur":    ("model2_ChargeDuration.h5",  "model2_ChargeDuration.pkl", "scaler_X.pkl"),
        "grid":   ("model3_GridAvail.h5",       "model3_GridAvail.pkl",      "scaler_X.pkl"),
        "energy": ("model4_EnergyDemand.h5",    "model4_EnergyDemand.pkl",   "scaler_enrg_X.pkl"),
    }
    for key, (h5, pkl, scx) in specs.items():
        paths = [os.path.join(base, p) for p in [h5, pkl, scx]]
        if all(os.path.exists(p) for p in paths):
            try:
                bundle = joblib.load(paths[1])
                mdls[key] = {
                    "model":    keras_load(paths[0], compile=False),
                    "scaler_X": joblib.load(paths[2]),
                    "scaler_y": bundle.get("scaler_y"),
                    "lookback": bundle.get("lookback"),
                    "features": bundle.get("features"),
                }
            except Exception as e:
                st.warning(f"Model {key} load error: {e}")
    return mdls

def mock_predict(key, inp):
    if key == "soc":    return round(float(inp.get("soc_pct", 65)) + np.random.uniform(-2, 4), 1)
    if key == "dur":    return round(inp.get("dod_pct", 40) / 100 * 2.78 + np.random.uniform(-0.05, 0.05), 2)
    if key == "grid":   return round(min(1, max(0, float(inp.get("grid_avail", 1)) + np.random.uniform(-0.04, 0.04))), 3)
    if key == "energy": return round(float(inp.get("dist_km", 94)) * 0.051 + np.random.uniform(-0.1, 0.1), 3)
    return 0.0

def run_prediction(models, key, window, raw):
    if key not in models:
        return mock_predict(key, raw), True
    m = models[key]
    lb, feats, sX, sY, model = m["lookback"], m["features"], m["scaler_X"], m["scaler_y"], m["model"]
    try:
        raw_mat = np.array([[window.get(f, [0]*lb)[i] for f in feats]
                             for i in range(lb)], dtype=np.float32)
        if hasattr(sX, "n_features_in_") and sX.n_features_in_ == len(feats):
            X_sc = sX.transform(raw_mat)
        else:
            feat_all = list(sX.feature_names_in_) if hasattr(sX, "feature_names_in_") else []
            if feat_all:
                idx = [feat_all.index(f) for f in feats if f in feat_all]
                dummy = np.zeros((lb, sX.n_features_in_), dtype=np.float32)
                for il, ig in enumerate(idx): dummy[:, ig] = raw_mat[:, il]
                X_sc = sX.transform(dummy)[:, idx]
            else:
                X_sc = raw_mat
        y_sc = model.predict(X_sc.reshape(1, lb, len(feats)), verbose=0)
        val = float(y_sc[0][0]) if key == "grid" else float(sY.inverse_transform(y_sc)[0][0])
        return val, False
    except Exception as e:
        st.warning(f"Prediction error ({key}): {e}")
        return mock_predict(key, raw), True

models = load_models() if TF_AVAILABLE else {}
demo_mode = not TF_AVAILABLE or not models

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:16px 0 20px;">
        <span style="font-size:2.4rem; filter:drop-shadow(0 0 12px rgba(255,255,255,0.5));">☀️</span>
        <div style="font-family:'Poppins',sans-serif; font-size:1.1rem; font-weight:700;
                    color:#ffffff; margin-top:6px;">SolarEV Intelligence</div>
        <div style="color:rgba(255,255,255,0.55); font-size:0.7rem; letter-spacing:0.08em; margin-top:2px;">
            DIT · M.ENG. 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="nav-user">
        <div class="nav-user-name">👤 {VALID_USERNAME}</div>
        <div class="nav-user-role">Researcher · Full Access</div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div style="font-size:0.67rem;color:rgba(255,255,255,0.5);letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px;font-family:\'Poppins\',sans-serif;font-weight:600;">System Specs</div>', unsafe_allow_html=True)
    for k, v in [
        ("🔋 Battery", "100 Ah · 48 V · 4.8 kWh"),
        ("⚡ Charger", "40 A CC-CV"),
        ("🕐 Ideal charge", "2.50 hrs"),
        ("⚙️ Actual (90% η)", "2.78 hrs"),
        ("☀️ PV+Grid hybrid", "2.28 hrs"),
        ("🌞 PSH Dar es Salaam", "5.17 hrs/day"),
        ("📍 Location", "DIT, -6.814°, 39.281°"),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:4px 0;
                    border-bottom:1px solid rgba(255,255,255,0.1);font-size:0.75rem;">
            <span style="color:rgba(255,255,255,0.55);">{k}</span>
            <span style="color:#ffffff;font-weight:500;">{v}</span>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div style="font-size:0.67rem;color:rgba(255,255,255,0.5);letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px;font-family:\'Poppins\',sans-serif;font-weight:600;">Model Accuracy</div>', unsafe_allow_html=True)
    for name, acc, color in [
        ("SOC Prediction",    "RMSE ±3.78%",   "#90caf9"),
        ("Charge Duration",   "RMSE ±4.4 min", "#80deea"),
        ("Grid Availability", "Acc 98.82%",    "#fff176"),
        ("Energy Demand",     "RMSE 3.4%",     "#ce93d8"),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:4px 0;
                    border-bottom:1px solid rgba(255,255,255,0.1);font-size:0.75rem;">
            <span style="color:rgba(255,255,255,0.55);">{name}</span>
            <span style="color:{color};font-weight:600;">{acc}</span>
        </div>""", unsafe_allow_html=True)

    st.divider()
    if demo_mode:
        st.markdown('<div style="background:rgba(245,158,11,0.15);border-left:3px solid #f59e0b;border-radius:0 8px 8px 0;padding:10px 12px;font-size:0.78rem;color:#fff176;">⚠️ <b>Demo Mode</b> — model files not found.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background:rgba(255,255,255,0.1);border-left:3px solid #90caf9;border-radius:0 8px 8px 0;padding:10px 12px;font-size:0.78rem;color:#ffffff;">✅ <b>{len(models)} LSTM models</b> loaded.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:rgba(255,255,255,0.3);font-size:0.7rem;margin-top:16px;line-height:1.9;">
        Alfred Kafaransa<br>Prof. K.A. Greyson · Dr. P.V. Chombo
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <div class="header-title">☀️ Solar PV Charging Intelligence System</div>
    <div class="header-subtitle">
        Modelling of a Solar PV-Based Charging System for Electric Three-Wheelers with AC-to-DC Grid Interface &nbsp;·&nbsp;
        Dar es Salaam Institute of Technology · M.Eng. Sustainable Energy Engineering
    </div>
    <div style="margin-top:10px;">
        <span class="header-badge">☀️ HYBRID PV+GRID</span>
        <span class="header-badge">🤖 4 LSTM MODELS</span>
        <span class="header-badge">📍 DAR ES SALAAM</span>
        <span class="header-badge">🚗 10 VEHICLES · 89 DAYS</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠  Dashboard",
    "🔋  SOC Prediction",
    "⏱️  Charge Duration",
    "🔌  Grid Availability",
    "📈  Energy Demand",
    "📊  System Comparison",
])

# ══════════════════════════════════════════════════════════
# TAB 0 — DASHBOARD  (no KPI cards — replaced with study summary)
# ══════════════════════════════════════════════════════════
with tabs[0]:
    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown('<div class="sec-head">System Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sim-block">
            <h4>📡 Study & Data Summary</h4>
            <div class="sim-row"><span class="sim-key">Data records</span><span class="sim-val">8,543 · 15-min intervals · Feb–Apr 2026</span></div>
            <div class="sim-row"><span class="sim-key">Vehicles tracked</span><span class="sim-val">10 electric three-wheelers (Bajaj)</span></div>
            <div class="sim-row"><span class="sim-key">Study area</span><span class="sim-val">DIT Factory, Dar es Salaam</span></div>
            <div class="sim-row"><span class="sim-key">LSTM architecture</span><span class="sim-val">2-layer LSTM + Dropout + Dense</span></div>
            <div class="sim-row"><span class="sim-key">Training split</span><span class="sim-val">70% train · 15% validation · 15% test</span></div>
            <div class="sim-row"><span class="sim-key">PV system size</span><span class="sim-val">13 × 200 W panels = 2.60 kW installed</span></div>
            <div class="sim-row"><span class="sim-key">Battery system</span><span class="sim-val">100 Ah · 48 V · 4.8 kWh Li-ion</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="sim-block">
            <h4>🔬 Model Targets</h4>
            <div class="sim-row"><span class="sim-key">Model 1 — SOC</span><span class="sim-val">Battery charge level (%) · next 15 min</span></div>
            <div class="sim-row"><span class="sim-key">Model 2 — Duration</span><span class="sim-val">Hours needed to fully charge</span></div>
            <div class="sim-row"><span class="sim-key">Model 3 — Grid</span><span class="sim-val">Grid ON / OFF · next 15 min</span></div>
            <div class="sim-row"><span class="sim-key">Model 4 — Energy</span><span class="sim-val">Daily energy demand (kWh)</span></div>
        </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="sec-head">Key Research Findings</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sim-block">
            <h4>⚡ Hybrid PV+Grid vs Grid-Only</h4>
            <div class="sim-row"><span class="sim-key">Charging time reduced</span><span class="sim-val" style="color:#1565c0;">2.78 → 2.28 hrs (−18%)</span></div>
            <div class="sim-row"><span class="sim-key">Daily downtime saved</span><span class="sim-val" style="color:#1565c0;">−1.38 hrs/day</span></div>
            <div class="sim-row"><span class="sim-key">Income recovered/day</span><span class="sim-val" style="color:#1565c0;">TZS 5,423</span></div>
            <div class="sim-row"><span class="sim-key">Annual income saved</span><span class="sim-val" style="color:#1565c0;">TZS 1,979,265</span></div>
            <div class="sim-row"><span class="sim-key">Grid dependency reduced</span><span class="sim-val" style="color:#1565c0;">100% → 51% (−49%)</span></div>
            <div class="sim-row"><span class="sim-key">PV energy share</span><span class="sim-val" style="color:#f59e0b;">49% of daily demand</span></div>
            <div class="sim-row"><span class="sim-key">Power availability</span><span class="sim-val" style="color:#43a047;">98.99% → 100%</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="sim-block">
            <h4>📊 Result Colour Guide</h4>
            <div class="sim-row"><span class="sim-key">🟢 Green / Blue</span><span class="sim-val" style="color:#2e7d32;">Safe operating range</span></div>
            <div class="sim-row"><span class="sim-key">🟡 Amber</span><span class="sim-val" style="color:#7c5200;">Moderate — attention needed</span></div>
            <div class="sim-row"><span class="sim-key">🔴 Red</span><span class="sim-val" style="color:#c62828;">Critical — act immediately</span></div>
            <div class="sim-row"><span class="sim-key">Confidence %</span><span class="sim-val">Model certainty (Grid model)</span></div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 1 — SOC PREDICTION
# ══════════════════════════════════════════════════════════
with tabs[1]:
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown('<div class="sec-head">🔋 Model 1 — SOC Prediction · Sensor Inputs</div>', unsafe_allow_html=True)
        soc_pct = st.slider("Current SOC (%)",          15.0, 100.0, 72.0, 0.5)
        vbat    = st.slider("Battery Voltage (V)",      44.9,  54.0,  50.9, 0.01)
        ibat_c  = st.slider("Charging Current (A)",      0.0,  41.7,   5.0, 0.1)
        ibat_d  = st.slider("Discharge Current (A)",     0.0,  20.0,   0.0, 0.1)
        tbat    = st.slider("Battery Temperature (°C)", 20.0,  55.0,  32.0, 0.5)
        ghi     = st.slider("Solar Irradiance (W/m²)",   0.0, 900.0, 450.0, 10.0)
        ppv     = st.slider("PV Power Output (W)",       0.0, 2500.0, 900.0, 10.0)
        grid_s1 = st.selectbox("Grid Available?", [1,0], format_func=lambda x: "Yes ✅" if x else "No ❌")
        hour_s1 = st.slider("Hour of Day", 0, 23, 10)
        csrc_s1 = st.selectbox("Charging Source", [0,1,2], format_func=lambda x: ["None 🚫","Grid ⚡","Solar PV ☀️"][x])
        hs1 = math.sin(2*math.pi*hour_s1/24); hc1 = math.cos(2*math.pi*hour_s1/24)
        window_s1 = {
            "V15_SOC_pct":[soc_pct]*12,"V12_Vbat_V":[vbat]*12,"V13_Ibat_c_A":[ibat_c]*12,
            "V14_Ibat_d_A":[ibat_d]*12,"V17_Tbat_C":[tbat]*12,"V1_ghi_Wm2":[ghi]*12,
            "V8_Ppv_W":[ppv]*12,"V29_grid_avail":[float(grid_s1)]*12,
            "hour_sin":[hs1]*12,"hour_cos":[hc1]*12,"charging_source_enc":[float(csrc_s1)]*12,
        }
        btn1 = st.button("🔮 Predict Next SOC", type="primary", use_container_width=True, key="btn1")

    with col2:
        st.markdown('<div class="sec-head">Current Battery State</div>', unsafe_allow_html=True)
        bc = soc_color(soc_pct)
        st.markdown(f"""
        <p style="color:#5a8aaa;font-size:0.78rem;margin-bottom:3px;">Current Battery SOC</p>
        <div class="bar-outer">
          <div class="bar-inner" style="width:{soc_pct}%;background:{bc};"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.76rem;color:#90caf9;margin-top:2px;">
          <span>0%</span><span style="color:{bc};font-weight:700;">{soc_pct:.1f}%</span><span>100%</span>
        </div>
        <br>{soc_badge(soc_pct)}""", unsafe_allow_html=True)
        st.divider()
        if btn1:
            with st.spinner("Running LSTM inference…"):
                pred, is_mock = run_prediction(models, "soc", window_s1, {"soc_pct": soc_pct})
            pred = max(0.0, min(100.0, pred)); delta = pred - soc_pct
            bc2 = soc_color(pred); note = " · Demo" if is_mock else ""
            st.markdown(f"""
            <div class="pred-box">
              <div class="pred-label">PREDICTED SOC — NEXT 15 MIN{note}</div>
              <div class="pred-value">{pred:.1f}%</div>
              <div class="pred-unit">{'↑' if delta>0 else '↓'} {abs(delta):.1f}% change from current</div>
            </div>
            <div class="bar-outer">
              <div class="bar-inner" style="width:{pred}%;background:{bc2};"></div>
            </div>
            <br>{soc_badge(pred)}""", unsafe_allow_html=True)
            if pred>=70: interp,panel="✅ <strong>Battery is in a good state.</strong> Continue current charging mode.","info-panel"
            elif pred>=40: interp,panel="⚠️ <strong>SOC is moderate.</strong> Consider activating PV or increasing charging current.","warn-panel"
            else: interp,panel="🚨 <strong>Critical SOC predicted.</strong> Connect to charging source immediately.","danger-panel"
            st.markdown(f'<div class="{panel}">{interp}</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="sim-block">
                <h4>🔬 Simulation Details</h4>
                <div class="sim-row"><span class="sim-key">Lookback window</span><span class="sim-val">12 steps = 3 hours</span></div>
                <div class="sim-row"><span class="sim-key">Input SOC</span><span class="sim-val">{soc_pct:.1f}%</span></div>
                <div class="sim-row"><span class="sim-key">Predicted SOC</span><span class="sim-val" style="color:#1565c0;">{pred:.1f}%</span></div>
                <div class="sim-row"><span class="sim-key">Change</span><span class="sim-val" style="color:{'#43a047' if delta>=0 else '#ef5350'};">{'+'if delta>=0 else ''}{delta:.1f}%</span></div>
                <div class="sim-row"><span class="sim-key">Accuracy</span><span class="sim-val">±3.78% error margin</span></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:50px 20px;">
                <div style="font-size:3.5rem;opacity:0.15;margin-bottom:10px;">🔋</div>
                <div style="font-size:0.88rem;color:#90caf9;">Set sensor values and click
                <strong style="color:#1565c0;">Predict Next SOC</strong></div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 2 — CHARGE DURATION
# ══════════════════════════════════════════════════════════
with tabs[2]:
    col1, col2 = st.columns([1,1], gap="large")
    with col1:
        st.markdown('<div class="sec-head">⏱️ Model 2 — Charging Duration · Inputs</div>', unsafe_allow_html=True)
        soc_t2  = st.slider("Current SOC (%)",         15.0,100.0,45.0,0.5,  key="s2a")
        dod_t2  = st.slider("Depth of Discharge (%)",   0.0, 85.0,55.0,0.5,  key="s2b")
        vbat_t2 = st.slider("Battery Voltage (V)",     44.9, 54.0,48.5,0.01, key="s2c")
        ibat_t2 = st.slider("Charging Current (A)",     0.0, 41.7,20.0,0.1,  key="s2d")
        ghi_t2  = st.slider("Solar Irradiance (W/m²)",  0.0,900.0,300.0,10.0,key="s2e")
        ppv_t2  = st.slider("PV Power (W)",             0.0,2500.0,600.0,10.0,key="s2f")
        grid_t2 = st.selectbox("Grid Available?",[1,0],key="s2g",format_func=lambda x:"Yes ✅" if x else "No ❌")
        hour_t2 = st.slider("Hour of Day",0,23,9,key="s2h")
        dow_t2  = st.slider("Day of Week (0=Mon)",0,6,1,key="s2i")
        csrc_t2 = st.selectbox("Charging Source",[0,1,2],key="s2j",format_func=lambda x:["None 🚫","Grid ⚡","Solar PV ☀️"][x])
        hs2=math.sin(2*math.pi*hour_t2/24);hc2=math.cos(2*math.pi*hour_t2/24);ds2=math.sin(2*math.pi*dow_t2/7)
        window_t2={"V15_SOC_pct":[soc_t2]*24,"V1_ghi_Wm2":[ghi_t2]*24,"V8_Ppv_W":[ppv_t2]*24,
                   "V29_grid_avail":[float(grid_t2)]*24,"V13_Ibat_c_A":[ibat_t2]*24,
                   "V12_Vbat_V":[vbat_t2]*24,"V16_DoD_pct":[dod_t2]*24,
                   "hour_sin":[hs2]*24,"hour_cos":[hc2]*24,"dow_sin":[ds2]*24,
                   "charging_source_enc":[float(csrc_t2)]*24}
        btn2 = st.button("⚡ Predict Charge Duration",type="primary",use_container_width=True,key="btn2")

    with col2:
        st.markdown('<div class="sec-head">Baseline Reference</div>', unsafe_allow_html=True)
        t_grid=(100-soc_t2)/100*2.78; t_hybrid=(100-soc_t2)/100*2.28
        c1,c2=st.columns(2)
        with c1:
            st.markdown(f"""<div class="stat-card"><div class="stat-card-icon">🔌</div>
              <div class="stat-card-value">{t_grid:.2f}</div><div class="stat-card-unit">hrs</div>
              <div class="stat-card-label">Grid-only estimate</div></div>""",unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="stat-card"><div class="stat-card-icon">☀️</div>
              <div class="stat-card-value">{t_hybrid:.2f}</div><div class="stat-card-unit">hrs</div>
              <div class="stat-card-label">PV+Grid estimate</div></div>""",unsafe_allow_html=True)
        st.divider()
        if btn2:
            with st.spinner("Running LSTM inference…"):
                pred_d,is_mock=run_prediction(models,"dur",window_t2,{"dod_pct":dod_t2})
            pred_d=max(0.0,pred_d);pred_m=pred_d*60;saving=max(0,t_grid-pred_d);note=" · Demo" if is_mock else ""
            st.markdown(f"""
            <div class="pred-box">
              <div class="pred-label">PREDICTED CHARGE DURATION{note}</div>
              <div class="pred-value">{pred_d:.2f}</div>
              <div class="pred-unit">hours &nbsp;·&nbsp; {pred_m:.0f} minutes to full charge</div>
            </div>""",unsafe_allow_html=True)
            st.progress(max(0.0,min(1.0,soc_t2/100)),text=f"Current: {soc_t2:.0f}% — {100-soc_t2:.0f}% to full")
            if pred_d<=t_hybrid: interp,panel=f"✅ <strong>Optimal speed.</strong> Duration ({pred_d:.2f} hrs) meets PV+Grid target.","info-panel"
            elif pred_d<=t_grid: interp,panel=f"⚡ <strong>Good performance.</strong> {pred_d:.2f} hrs — faster than grid-only.","info-panel"
            else: interp,panel=f"⚠️ <strong>Slower than expected.</strong> Consider activating PV source.","warn-panel"
            st.markdown(f'<div class="{panel}">{interp}</div>',unsafe_allow_html=True)
            st.markdown(f"""
            <div class="sim-block">
                <h4>🔬 Simulation Details</h4>
                <div class="sim-row"><span class="sim-key">Lookback</span><span class="sim-val">24 steps = 6 hrs</span></div>
                <div class="sim-row"><span class="sim-key">Predicted</span><span class="sim-val" style="color:#1565c0;">{pred_d:.2f} hrs ({pred_m:.0f} min)</span></div>
                <div class="sim-row"><span class="sim-key">Grid-only</span><span class="sim-val" style="color:#ef5350;">{t_grid:.2f} hrs</span></div>
                <div class="sim-row"><span class="sim-key">PV+Grid</span><span class="sim-val" style="color:#f59e0b;">{t_hybrid:.2f} hrs</span></div>
                <div class="sim-row"><span class="sim-key">Time saved</span><span class="sim-val" style="color:#43a047;">−{saving:.2f} hrs ({saving*60:.0f} min)</span></div>
                <div class="sim-row"><span class="sim-key">Accuracy</span><span class="sim-val">±4.4 min error</span></div>
            </div>""",unsafe_allow_html=True)
        else:
            st.markdown("""<div style="text-align:center;padding:50px 20px;">
                <div style="font-size:3.5rem;opacity:0.15;margin-bottom:10px;">⏱️</div>
                <div style="font-size:0.88rem;color:#90caf9;">Set parameters and click
                <strong style="color:#1565c0;">Predict Charge Duration</strong></div>
            </div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 3 — GRID AVAILABILITY
# ══════════════════════════════════════════════════════════
with tabs[3]:
    col1,col2=st.columns([1,1],gap="large")
    with col1:
        st.markdown('<div class="sec-head">🔌 Model 3 — Grid Availability · Inputs</div>',unsafe_allow_html=True)
        grid_now=st.selectbox("Current Grid Status",[1,0],format_func=lambda x:"ON ✅" if x else "OFF ❌",key="g3a")
        out_dur=st.slider("Outage Duration (hrs)",0.0,1.8,0.0,0.05)
        n_out=st.slider("Outages Today (count)",0,5,0)
        vgrid=st.slider("Grid Supply Voltage (V)",195.0,245.0,228.0,0.5)
        fgrid=st.slider("Grid Frequency (Hz)",49.0,51.0,50.0,0.1)
        delta_v=st.slider("Voltage Deviation from 230V (V)",-30.0,15.0,-2.0,0.5)
        hour_t3=st.slider("Hour of Day",0,23,14,key="g3b")
        dow_t3=st.slider("Day of Week (0=Mon)",0,6,2,key="g3c")
        hs3=math.sin(2*math.pi*hour_t3/24);hc3=math.cos(2*math.pi*hour_t3/24)
        ds3=math.sin(2*math.pi*dow_t3/7);dc3=math.cos(2*math.pi*dow_t3/7)
        window_t3={"V29_grid_avail":[float(grid_now)]*48,"V30_outage_dur_hr":[out_dur]*48,
                   "V31_n_outages":[float(n_out)]*48,"V33_deltaV_V":[delta_v]*48,
                   "V26_Vgrid_V":[vgrid]*48,"V32_Fgrid_Hz":[fgrid]*48,
                   "hour_sin":[hs3]*48,"hour_cos":[hc3]*48,"dow_sin":[ds3]*48,"dow_cos":[dc3]*48}
        btn3=st.button("🔌 Predict Grid Status",type="primary",use_container_width=True,key="btn3")

    with col2:
        st.markdown('<div class="sec-head">Grid Status & Prediction</div>',unsafe_allow_html=True)
        badge='<span class="badge-on">✅ GRID IS ON</span>' if grid_now else '<span class="badge-off">❌ GRID IS OFF</span>'
        st.markdown(f"<p style='color:#5a8aaa;font-size:0.76rem;margin-bottom:4px;'>Current Status</p>{badge}<br><br>",unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        with c1: st.metric("Grid Uptime","98.99%")
        with c2: st.metric("Avg Outage","0.20 hrs")
        with c3: st.metric("Daily Outages","0.33")
        st.divider()
        if btn3:
            with st.spinner("Running LSTM classifier…"):
                pred_p,is_mock=run_prediction(models,"grid",window_t3,{"grid_avail":grid_now})
            pred_lbl=1 if pred_p>=0.5 else 0
            bar_c="#1565c0" if pred_p>=0.5 else "#ef5350"
            note=" · Demo" if is_mock else ""
            st.markdown(f"""
            <div class="pred-box" style="border-color:{bar_c};">
              <div class="pred-label">NEXT 15-MIN GRID PREDICTION{note}</div>
              <div class="pred-value" style="color:{bar_c};">{"ON ✅" if pred_lbl else "OFF ❌"}</div>
              <div class="pred-unit">Confidence: {pred_p*100:.1f}%</div>
            </div>""",unsafe_allow_html=True)
            bw=int(pred_p*100)
            st.markdown(f"""
            <p style="color:#5a8aaa;font-size:0.76rem;margin-bottom:3px;">Probability of Grid ON</p>
            <div class="bar-outer">
              <div class="bar-inner" style="width:{bw}%;background:{bar_c};"></div>
            </div>
            <p style="text-align:right;color:{bar_c};font-weight:700;font-size:0.88rem;">{pred_p*100:.1f}%</p>""",unsafe_allow_html=True)
            if pred_lbl==1 and pred_p>=0.85: interp,panel="✅ <strong>Grid highly likely to remain ON.</strong> Safe to plan grid charging sessions.","info-panel"
            elif pred_lbl==1: interp,panel="⚡ <strong>Grid predicted ON.</strong> Proceed with charging but keep PV backup ready.","warn-panel"
            else: interp,panel="🚨 <strong>Grid outage predicted.</strong> Switch to solar PV charging immediately.","danger-panel"
            st.markdown(f'<div class="{panel}">{interp}</div>',unsafe_allow_html=True)
            st.markdown(f"""
            <div class="sim-block">
                <h4>🔬 Simulation Details</h4>
                <div class="sim-row"><span class="sim-key">Lookback</span><span class="sim-val">48 steps = 12 hrs</span></div>
                <div class="sim-row"><span class="sim-key">Current grid</span><span class="sim-val">{"ON" if grid_now else "OFF"}</span></div>
                <div class="sim-row"><span class="sim-key">Grid voltage</span><span class="sim-val">{vgrid:.1f} V</span></div>
                <div class="sim-row"><span class="sim-key">Outages today</span><span class="sim-val">{n_out}</span></div>
                <div class="sim-row"><span class="sim-key">Prediction</span><span class="sim-val" style="color:{bar_c};">{"ON" if pred_lbl else "OFF"} · {pred_p*100:.1f}% confidence</span></div>
                <div class="sim-row"><span class="sim-key">Model accuracy</span><span class="sim-val">98.82% on test data</span></div>
            </div>""",unsafe_allow_html=True)
        else:
            st.markdown("""<div style="text-align:center;padding:50px 20px;">
                <div style="font-size:3.5rem;opacity:0.15;margin-bottom:10px;">🔌</div>
                <div style="font-size:0.88rem;color:#90caf9;">Set grid parameters and click
                <strong style="color:#1565c0;">Predict Grid Status</strong></div>
            </div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 4 — ENERGY DEMAND
# ══════════════════════════════════════════════════════════
with tabs[4]:
    col1,col2=st.columns([1,1],gap="large")
    with col1:
        st.markdown('<div class="sec-head">📈 Model 4 — Energy Demand · 7-Day Inputs</div>',unsafe_allow_html=True)
        st.markdown('<p style="color:#5a8aaa;font-size:0.78rem;font-style:italic;margin-bottom:10px;">Enter average values from the past 7 days</p>',unsafe_allow_html=True)
        dist_km =st.slider("Avg Daily Distance (km)",        10.0,120.0, 94.6,1.0)
        op_hours=st.slider("Avg Operating Hours/Day",         2.0, 18.0,  9.0,0.5)
        downtime=st.slider("Avg Charging Downtime (hrs/day)", 0.0,  8.0,  5.5,0.1)
        pv_kwh  =st.slider("Avg PV Energy Contributed (kWh)",0.0,  5.0,  3.6,0.05)
        g_frac  =st.slider("Grid Availability Fraction",      0.0,  1.0, 0.99,0.01)
        n_out4  =st.slider("Avg Outages Per Day",             0.0,  5.0, 0.33,0.1)
        dow_t4  =st.slider("Day of Week for Forecast (0=Mon)",0,    6,    0)
        window_t4={"dist_km":[dist_km]*7,"op_hours":[op_hours]*7,"downtime_hr":[downtime]*7,
                   "pv_energy_kwh":[pv_kwh]*7,"grid_avail_frac":[g_frac]*7,
                   "n_outages":[n_out4]*7,"day_of_week":[float(dow_t4)]*7}
        btn4=st.button("📈 Forecast Energy Demand",type="primary",use_container_width=True,key="btn4")

    with col2:
        st.markdown('<div class="sec-head">Forecast Result</div>',unsafe_allow_html=True)
        baseline=dist_km*50.84/1000
        c1,c2=st.columns(2)
        with c1:
            st.markdown(f"""<div class="stat-card"><div class="stat-card-icon">📏</div>
              <div class="stat-card-value">{baseline:.2f}</div><div class="stat-card-unit">kWh/day</div>
              <div class="stat-card-label">Formula estimate</div></div>""",unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="stat-card"><div class="stat-card-icon">📊</div>
              <div class="stat-card-value">4.83</div><div class="stat-card-unit">kWh/day</div>
              <div class="stat-card-label">Study fleet average</div></div>""",unsafe_allow_html=True)
        st.divider()
        if btn4:
            with st.spinner("Running LSTM inference…"):
                pred_e,is_mock=run_prediction(models,"energy",window_t4,{"dist_km":dist_km})
            pred_e=max(0.0,pred_e)
            pv_cover=min(100.0,pv_kwh/pred_e*100) if pred_e>0 else 0
            grid_need=max(0.0,pred_e-pv_kwh);sessions=math.ceil(pred_e/4.8)
            note=" · Demo" if is_mock else ""
            st.markdown(f"""
            <div class="pred-box">
              <div class="pred-label">PREDICTED ENERGY DEMAND TOMORROW{note}</div>
              <div class="pred-value">{pred_e:.3f}</div>
              <div class="pred-unit">kWh / day</div>
            </div>""",unsafe_allow_html=True)
            pv_pct=int(pv_cover);gr_pct=100-pv_pct
            st.markdown(f"""
            <p style="color:#5a8aaa;font-size:0.76rem;margin:8px 0 3px;">Energy Source Mix</p>
            <div class="bar-outer" style="height:20px;">
              <div style="display:flex;height:100%;border-radius:8px;overflow:hidden;">
                <div style="width:{pv_pct}%;background:linear-gradient(90deg,#f59e0b,#fbbf24);"></div>
                <div style="width:{gr_pct}%;background:linear-gradient(90deg,#1565c0,#1e90ff);"></div>
              </div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.76rem;margin-top:3px;">
              <span style="color:#f59e0b;">☀️ PV: {pv_pct}% ({pv_kwh:.2f} kWh)</span>
              <span style="color:#1565c0;">⚡ Grid: {gr_pct}% ({grid_need:.2f} kWh)</span>
            </div>""",unsafe_allow_html=True)
            if pv_cover>=70: interp,panel=f"✅ <strong>Excellent PV coverage ({pv_cover:.1f}%).</strong> Solar supplies most of tomorrow's demand.","info-panel"
            elif pv_cover>=40: interp,panel=f"⚡ <strong>Moderate PV ({pv_cover:.1f}%).</strong> Grid must supply {grid_need:.2f} kWh.","warn-panel"
            else: interp,panel=f"⚠️ <strong>Low PV coverage ({pv_cover:.1f}%).</strong> High grid dependency today.","warn-panel"
            st.markdown(f'<div class="{panel}">{interp}</div>',unsafe_allow_html=True)
            st.markdown(f"""
            <div class="sim-block">
                <h4>🔬 Simulation Details</h4>
                <div class="sim-row"><span class="sim-key">Lookback</span><span class="sim-val">7 days</span></div>
                <div class="sim-row"><span class="sim-key">LSTM prediction</span><span class="sim-val" style="color:#1565c0;">{pred_e:.3f} kWh/day</span></div>
                <div class="sim-row"><span class="sim-key">PV coverage</span><span class="sim-val" style="color:#f59e0b;">{pv_cover:.1f}% ({pv_kwh:.2f} kWh)</span></div>
                <div class="sim-row"><span class="sim-key">Grid required</span><span class="sim-val" style="color:#00acc1;">{grid_need:.2f} kWh</span></div>
                <div class="sim-row"><span class="sim-key">Sessions needed</span><span class="sim-val">{sessions} cycle(s)</span></div>
                <div class="sim-row"><span class="sim-key">Accuracy</span><span class="sim-val">RMSE 3.4% of mean</span></div>
            </div>""",unsafe_allow_html=True)
        else:
            st.markdown("""<div style="text-align:center;padding:50px 20px;">
                <div style="font-size:3.5rem;opacity:0.15;margin-bottom:10px;">📈</div>
                <div style="font-size:0.88rem;color:#90caf9;">Enter 7-day averages and click
                <strong style="color:#1565c0;">Forecast Energy Demand</strong></div>
            </div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 5 — SYSTEM COMPARISON
# ══════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="sec-head">Performance Comparison — Grid-Only vs Hybrid PV+Grid</div>',unsafe_allow_html=True)

    for icon,title,base,new,improve,explanation in [
        ("🔋","Charging Power Availability","98.99%","100.00%","+5.2%","PV backup covers outage periods — full 100% availability achieved."),
        ("⏱️","Avg Charge Duration / Session","2.78 hrs","2.28 hrs","−18% faster","30 minutes saved per session, increasing daily vehicle availability."),
        ("🕐","Daily Charging Downtime","5.50 hrs","4.13 hrs","−1.38 hrs/day","1.38 fewer hours in downtime per day — more operational time."),
        ("💰","Daily Income Lost (TZS)","8,343","2,920","TZS 5,423 recovered","TZS 5,423 per day recovered — over TZS 1.97M per year."),
        ("🔌","Grid Dependency","100%","51%","−49% reduction","Half of all energy now comes from solar PV."),
        ("☀️","PV Energy Contribution","0%","49.0%","+49% renewable","Solar panels supply 49% of total charging energy."),
    ]:
        st.markdown(f"""
        <div class="compare-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div style="display:flex;align-items:center;gap:9px;">
                    <span style="font-size:1.3rem;">{icon}</span>
                    <span style="font-family:'Poppins',sans-serif;font-size:0.9rem;color:#0d2137;font-weight:600;">{title}</span>
                </div>
                <span style="color:#00acc1;font-size:0.78rem;font-weight:600;
                             background:#e0f7fa;border:1px solid #80deea;
                             padding:2px 9px;border-radius:12px;">{improve}</span>
            </div>
            <div class="compare-row">
                <div>
                    <div style="color:#90caf9;font-size:0.66rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:2px;">Grid-Only (Baseline)</div>
                    <div class="compare-base">{base}</div>
                </div>
                <div style="color:#b3d4f0;font-size:1.4rem;padding:0 12px;">→</div>
                <div>
                    <div style="color:#90caf9;font-size:0.66rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:2px;">Hybrid PV+Grid (Proposed)</div>
                    <div class="compare-new">{new}</div>
                </div>
            </div>
            <div style="color:#5a8aaa;font-size:0.78rem;margin-top:8px;
                        padding-top:7px;border-top:1px solid #e3f2fd;line-height:1.5;">
                💡 {explanation}
            </div>
        </div>""",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    col_a,col_b=st.columns(2,gap="large")

    with col_a:
        st.markdown('<div class="sec-head">Operational Challenge Severity (0–100)</div>',unsafe_allow_html=True)
        for name,score,color in [
            ("Charging Downtime", 55.0,"#ef5350"),
            ("Income Loss",       27.8,"#f59e0b"),
            ("Battery Stress",    27.6,"#f59e0b"),
            ("Outage Duration",   24.8,"#f59e0b"),
            ("Low SOC Frequency",  8.9,"#43a047"),
            ("Grid Unreliability", 1.0,"#43a047"),
        ]:
            st.markdown(f"""
            <div style="margin:7px 0;">
              <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:2px;">
                <span style="color:#1e4976;">{name}</span>
                <span style="color:{color};font-weight:700;">{score}/100</span>
              </div>
              <div class="bar-outer" style="height:12px;">
                <div class="bar-inner" style="width:{score}%;background:{color};"></div>
              </div>
            </div>""",unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="sec-head">LSTM Model Summary</div>',unsafe_allow_html=True)
        for num,icon,name,task,lb,acc,color in [
            ("1","🔋","SOC Prediction",   "Regression",    "3 hrs", "RMSE ±3.78%",  "#1565c0"),
            ("2","⏱️","Charge Duration",  "Regression",    "6 hrs", "RMSE ±4.4 min","#00acc1"),
            ("3","🔌","Grid Availability","Classification","12 hrs","Acc 98.82%",   "#f59e0b"),
            ("4","📈","Energy Demand",    "Regression",    "7 days","RMSE 3.4%",    "#7b1fa2"),
        ]:
            st.markdown(f"""
            <div class="sim-block" style="margin:7px 0;padding:12px 16px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="display:flex;gap:9px;align-items:center;">
                        <span style="font-size:1.2rem;">{icon}</span>
                        <div>
                            <div style="font-family:'Poppins',sans-serif;color:#0d2137;font-size:0.85rem;font-weight:700;">Model {num} — {name}</div>
                            <div style="color:#5a8aaa;font-size:0.72rem;margin-top:1px;">{task} · {lb} lookback</div>
                        </div>
                    </div>
                    <span style="color:{color};font-weight:700;font-size:0.8rem;
                                 background:#f0f8ff;padding:2px 9px;
                                 border-radius:10px;border:1px solid {color}55;">{acc}</span>
                </div>
            </div>""",unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="text-align:center;color:#90caf9;font-size:0.78rem;padding:14px;line-height:2.2;">
        <strong style="color:#1565c0;">Alfred Kafaransa</strong> &nbsp;·&nbsp;
        M.Eng. Sustainable Energy Engineering &nbsp;·&nbsp;
        Dar es Salaam Institute of Technology &nbsp;·&nbsp; 2026<br>
        Supervisors: <strong style="color:#1565c0;">Prof. Kenedy Aliila Greyson</strong> &amp;
        <strong style="color:#1565c0;">Dr. Pius V. Chombo</strong>
    </div>""",unsafe_allow_html=True)
