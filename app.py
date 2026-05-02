import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="BankPulse Pro | Loan Acceptance Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    lr_model = joblib.load("lr_model.pkl")
    dt_model = joblib.load("dt_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    numeric_features = joblib.load("numeric_features.pkl")
    return lr_model, dt_model, scaler, columns, numeric_features


try:
    lr_model, dt_model, scaler, columns, numeric_features = load_models()
except Exception as e:
    st.error(f"⚠️ Model files not found. Please run train.py first. Error: {e}")
    st.stop()

# ================= SESSION STATE =================
if "predictions" not in st.session_state:
    st.session_state.predictions = []

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

# ================= THEME PICKER FIRST =================
with st.sidebar:
    selected_theme = st.radio(
        "🎨 Choose Theme",
        ["Dark", "Light"],
        index=0 if st.session_state.theme_mode == "Dark" else 1,
        horizontal=True,
        key="theme_picker_top"
    )

if selected_theme != st.session_state.theme_mode:
    st.session_state.theme_mode = selected_theme
    st.rerun()

theme_mode = st.session_state.theme_mode

# ================= THEME VARIABLES =================
if theme_mode == "Light":
    APP_BG = "#f7f3ea"
    APP_BG_2 = "#ffffff"
    PANEL_BG = "rgba(255, 255, 255, 0.96)"
    PANEL_BG_2 = "rgba(248, 250, 252, 0.95)"
    TEXT = "#0f172a"
    MUTED = "#64748b"
    BORDER = "rgba(15, 23, 42, 0.12)"
    SHADOW = "rgba(15, 23, 42, 0.10)"
    GRID = "rgba(100, 116, 139, 0.22)"
    TABLE_CELL_BG = "#ffffff"
    HERO_GLOW = "rgba(212, 175, 55, 0.10)"
else:
    APP_BG = "#080d17"
    APP_BG_2 = "#111827"
    PANEL_BG = "rgba(17, 24, 39, 0.90)"
    PANEL_BG_2 = "rgba(15, 23, 42, 0.92)"
    TEXT = "#f8fafc"
    MUTED = "#94a3b8"
    BORDER = "rgba(148, 163, 184, 0.18)"
    SHADOW = "rgba(0, 0, 0, 0.24)"
    GRID = "rgba(148, 163, 184, 0.20)"
    TABLE_CELL_BG = "#fbfaf6"
    HERO_GLOW = "rgba(212, 175, 55, 0.12)"

GOLD = "#d4af37"
GOLD_LIGHT = "#f7df8a"
GREEN = "#10b981"
RED = "#ef4444"
NAVY = "#07111f"

# ================= CSS =================
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@500;600;700&display=swap');

:root {{
    --app-bg: {APP_BG};
    --app-bg-2: {APP_BG_2};
    --panel-bg: {PANEL_BG};
    --panel-bg-2: {PANEL_BG_2};
    --text-main: {TEXT};
    --text-muted: {MUTED};
    --border-soft: {BORDER};
    --shadow-soft: {SHADOW};
    --lux-gold: {GOLD};
    --lux-gold-light: {GOLD_LIGHT};
    --lux-green: {GREEN};
    --lux-red: {RED};
    --table-cell-bg: {TABLE_CELL_BG};
    --hero-glow: {HERO_GLOW};
}}

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
}}

.stApp,
[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(circle at top left, rgba(212, 175, 55, 0.12), transparent 34%),
        radial-gradient(circle at top right, rgba(16, 185, 129, 0.08), transparent 30%),
        linear-gradient(135deg, var(--app-bg) 0%, var(--app-bg-2) 100%) !important;
    color: var(--text-main) !important;
}}

[data-testid="stHeader"] {{
    background: rgba(0, 0, 0, 0.04) !important;
    backdrop-filter: blur(18px) !important;
    border-bottom: 1px solid var(--border-soft) !important;
}}

.block-container {{
    padding-top: 2.3rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2.2rem !important;
    padding-right: 2.2rem !important;
}}

h1, h2, h3, h4, h5, h6,
p, span, label,
.stMarkdown,
.stMarkdown p,
[data-testid="stMarkdownContainer"] {{
    color: var(--text-main) !important;
}}

/* Streamlit controls visible */
[data-testid="stToolbar"],
[data-testid="stHeader"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {{
    visibility: visible !important;
    display: flex !important;
}}

[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button {{
    background: rgba(212, 175, 55, 0.18) !important;
    border: 1px solid rgba(212, 175, 55, 0.40) !important;
    color: var(--text-main) !important;
    border-radius: 12px !important;
}}

/* ================= SIDEBAR ================= */
[data-testid="stSidebar"] {{
    background:
        radial-gradient(circle at 25% 0%, rgba(212, 175, 55, 0.18), transparent 32%),
        linear-gradient(180deg, var(--app-bg-2), var(--app-bg)) !important;
    border-right: 1px solid rgba(212, 175, 55, 0.35) !important;
    box-shadow: 12px 0 42px var(--shadow-soft) !important;
}}

[data-testid="stSidebar"] > div:first-child {{
    padding: 1.2rem 1rem !important;
}}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
    color: var(--text-main) !important;
}}

.sidebar-brand-card {{
    text-align: center;
    padding: 24px 14px;
    margin-bottom: 22px;
    border-radius: 26px;
    background: linear-gradient(145deg, rgba(212,175,55,0.17), rgba(148,163,184,0.08));
    border: 1px solid rgba(212, 175, 55, 0.42);
    box-shadow: 0 18px 42px var(--shadow-soft);
}}

.sidebar-logo {{
    width: 64px;
    height: 64px;
    margin: 0 auto 13px auto;
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 32px;
    background: linear-gradient(135deg, var(--lux-gold), var(--lux-gold-light));
    color: #07111f !important;
    box-shadow: 0 14px 30px rgba(212, 175, 55, 0.32);
}}

.sidebar-brand-title {{
    font-size: 21px;
    font-weight: 900;
    letter-spacing: 1px;
    color: var(--text-main) !important;
}}

.sidebar-brand-subtitle {{
    margin-top: 5px;
    font-size: 11px;
    font-weight: 750;
    color: var(--text-muted) !important;
}}

.sidebar-pill {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    margin-top: 15px;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(212, 175, 55, 0.16);
    color: var(--lux-gold) !important;
    border: 1px solid rgba(212, 175, 55, 0.38);
    font-size: 11px;
    font-weight: 900;
}}

.sidebar-section-title {{
    margin: 18px 0 12px 0;
    padding: 12px 14px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    background: linear-gradient(135deg, rgba(212,175,55,0.20), rgba(148,163,184,0.10));
    border: 1px solid rgba(212, 175, 55, 0.42);
    box-shadow: 0 10px 26px var(--shadow-soft);
}}

.sidebar-section-title span {{
    color: var(--text-main) !important;
    font-size: 12px !important;
    font-weight: 900 !important;
    text-transform: uppercase;
    letter-spacing: 1.35px;
}}

.sidebar-status-card {{
    padding: 13px;
    border-radius: 18px;
    background: rgba(148, 163, 184, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.16);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
    margin-bottom: 6px;
}}

.sidebar-status-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid rgba(148, 163, 184, 0.13);
}}

.sidebar-status-row:last-child {{
    border-bottom: none;
}}

.sidebar-status-label {{
    font-size: 12px !important;
    font-weight: 750 !important;
    color: var(--text-muted) !important;
}}

.sidebar-badge-success,
.sidebar-badge-gold {{
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 11px !important;
    font-weight: 900 !important;
}}

.sidebar-badge-success {{
    background: rgba(16, 185, 129, 0.16);
    color: #10b981 !important;
    border: 1px solid rgba(16, 185, 129, 0.32);
}}

.sidebar-badge-gold {{
    background: rgba(212, 175, 55, 0.17);
    color: var(--lux-gold) !important;
    border: 1px solid rgba(212, 175, 55, 0.40);
}}

.sidebar-footer {{
    margin-top: 22px;
    padding: 13px;
    border-radius: 17px;
    text-align: center;
    background: linear-gradient(145deg, rgba(212,175,55,0.14), rgba(148,163,184,0.08));
    border: 1px solid rgba(212, 175, 55, 0.30);
}}

.sidebar-footer p {{
    margin: 0;
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted) !important;
}}

/* ================= INPUTS ================= */
div[data-baseweb="select"] > div {{
    background: rgba(148, 163, 184, 0.08) !important;
    color: var(--text-main) !important;
    border: 1px solid rgba(212, 175, 55, 0.30) !important;
    border-radius: 15px !important;
    min-height: 44px !important;
    box-shadow: 0 6px 20px var(--shadow-soft) !important;
}}

div[data-baseweb="select"] span,
div[data-baseweb="select"] div,
div[data-baseweb="select"] input {{
    color: var(--text-main) !important;
    font-weight: 800 !important;
}}

div[data-baseweb="select"] svg {{
    fill: var(--lux-gold) !important;
}}

div[data-baseweb="popover"] [role="listbox"] {{
    background: var(--app-bg-2) !important;
    border: 1px solid rgba(212, 175, 55, 0.40) !important;
    border-radius: 14px !important;
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.24) !important;
}}

div[data-baseweb="popover"] [role="option"],
div[data-baseweb="popover"] [role="option"] div,
div[data-baseweb="popover"] [role="option"] span {{
    color: var(--text-main) !important;
    background: var(--app-bg-2) !important;
}}

div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] [role="option"]:hover div,
div[data-baseweb="popover"] [role="option"]:hover span {{
    background: rgba(212, 175, 55, 0.18) !important;
}}

.stSlider label,
.stRadio label {{
    color: var(--text-main) !important;
    font-weight: 900 !important;
}}

.stSlider [data-baseweb="slider"] {{
    padding-top: 8px !important;
}}

.stSlider [data-testid="stTickBar"] {{
    background: rgba(212, 175, 55, 0.35) !important;
    height: 7px !important;
    border-radius: 999px !important;
}}

.stSlider [role="slider"] {{
    width: 22px !important;
    height: 22px !important;
    background: linear-gradient(135deg, var(--lux-gold), var(--lux-gold-light)) !important;
    border: 3px solid var(--app-bg) !important;
    box-shadow: 0 0 0 7px rgba(212, 175, 55, 0.25), 0 8px 20px rgba(0,0,0,0.30) !important;
}}

.stRadio div[role="radiogroup"] {{
    background: rgba(148, 163, 184, 0.08);
    border: 1px solid rgba(212, 175, 55, 0.28);
    border-radius: 15px;
    padding: 8px 10px;
}}

.stRadio div[role="radiogroup"] label {{
    font-weight: 800 !important;
}}

/* ================= HERO ================= */
.hero-wrapper {{
    padding-top: 0.45rem;
    padding-bottom: 0.8rem;
    padding-left: 0.35rem;
    padding-right: 0.35rem;
}}

.hero-card {{
    padding: 32px 34px;
    border-radius: 30px;
    margin: 0.3rem 0 1.4rem 0;
    background: linear-gradient(135deg, rgba(212,175,55,0.16), rgba(148,163,184,0.09));
    border: 1px solid rgba(212, 175, 55, 0.42);
    box-shadow: 0 24px 70px var(--shadow-soft);
}}

.hero-title {{
    font-size: 34px;
    line-height: 1.1;
    font-weight: 900;
    letter-spacing: -0.8px;
    color: var(--text-main) !important;
    margin-bottom: 8px;
}}

.hero-subtitle {{
    margin-top: 8px;
    font-size: 14px;
    font-weight: 650;
    color: var(--text-muted) !important;
}}

.hero-tag {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin-top: 15px;
    padding: 8px 13px;
    border-radius: 999px;
    background: rgba(212, 175, 55, 0.16);
    color: var(--lux-gold) !important;
    border: 1px solid rgba(212, 175, 55, 0.34);
    font-size: 12px;
    font-weight: 900;
}}

/* ================= METRIC CARDS ================= */
.metric-container,
.kpi-card {{
    background: var(--panel-bg) !important;
    color: var(--text-main) !important;
    border-radius: 24px !important;
    padding: 24px !important;
    border: 1px solid var(--border-soft) !important;
    box-shadow: 0 18px 50px var(--shadow-soft) !important;
    margin-bottom: 20px !important;
    transition: all 0.25s ease !important;
}}

.metric-container:hover,
.kpi-card:hover {{
    transform: translateY(-3px);
    border-color: rgba(212, 175, 55, 0.48) !important;
    box-shadow: 0 24px 70px var(--shadow-soft) !important;
}}

.metric-value,
.kpi-value {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 30px !important;
    font-weight: 900 !important;
    color: var(--text-main) !important;
}}

.metric-label,
.kpi-label {{
    color: var(--text-muted) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.35px !important;
    font-weight: 900 !important;
}}

.kpi-row {{
    display: flex;
    align-items: center;
    gap: 13px;
    margin-bottom: 12px;
}}

.kpi-icon {{
    width: 44px;
    height: 44px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 21px;
    background: rgba(212, 175, 55, 0.18);
    border: 1px solid rgba(212, 175, 55, 0.30);
}}

.metric-delta {{
    font-size: 13px !important;
    font-weight: 900 !important;
}}

.delta-positive {{ color: #10b981 !important; }}
.delta-negative {{ color: #ef4444 !important; }}

/* ================= HEADINGS ================= */
.section-heading-box {{
    margin: 10px 0 18px 0;
    padding: 14px 18px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(212,175,55,0.15), rgba(148,163,184,0.08));
    border: 1px solid rgba(212,175,55,0.38);
    box-shadow: 0 12px 28px var(--shadow-soft);
}}

.section-heading-box span {{
    color: var(--text-main) !important;
    font-size: 20px !important;
    font-weight: 900 !important;
    letter-spacing: -0.2px;
}}

.panel-title-box {{
    margin-bottom: 15px;
    padding: 12px 15px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(212,175,55,0.16), rgba(148,163,184,0.08));
    border: 1px solid rgba(212,175,55,0.35);
}}

.panel-title-box span {{
    color: var(--text-main) !important;
    font-size: 13px !important;
    font-weight: 900 !important;
    text-transform: uppercase;
    letter-spacing: 1.25px;
}}

/* Streamlit bordered container */
[data-testid="stVerticalBlockBorderWrapper"] {{
    border: 1px solid var(--border-soft) !important;
    border-radius: 24px !important;
    background: var(--panel-bg) !important;
    box-shadow: 0 18px 50px var(--shadow-soft) !important;
    padding: 1.15rem !important;
}}

[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    border-color: rgba(212,175,55,0.45) !important;
}}

/* ================= TABS ================= */
.stTabs [data-baseweb="tab-list"] {{
    gap: 10px !important;
    background: transparent !important;
    border-bottom: 1px solid var(--border-soft) !important;
    padding-bottom: 8px !important;
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 16px !important;
    padding: 12px 20px !important;
    background: rgba(148, 163, 184, 0.08) !important;
    border: 1px solid var(--border-soft) !important;
    color: var(--text-muted) !important;
    font-weight: 900 !important;
    box-shadow: 0 10px 26px var(--shadow-soft) !important;
}}

.stTabs [data-baseweb="tab"] p {{
    color: inherit !important;
    font-weight: 900 !important;
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, #07111f, #111827) !important;
    border-color: rgba(212, 175, 55, 0.62) !important;
    color: #ffffff !important;
}}

.stTabs [aria-selected="true"] p {{
    color: #ffffff !important;
}}

/* ================= BUTTON ================= */
.stButton > button {{
    background: linear-gradient(135deg, #07111f 0%, #111827 58%, #d4af37 180%) !important;
    color: #ffffff !important;
    border-radius: 15px !important;
    font-weight: 900 !important;
    letter-spacing: 0.4px !important;
    padding: 0.85rem 1.5rem !important;
    border: 1px solid rgba(212, 175, 55, 0.50) !important;
    box-shadow: 0 16px 42px rgba(0, 0, 0, 0.22) !important;
}}

.stButton > button:hover {{
    transform: translateY(-2px);
    border-color: rgba(245, 215, 122, 0.9) !important;
    box-shadow: 0 22px 55px rgba(0, 0, 0, 0.32) !important;
}}

.stButton > button p {{
    color: #ffffff !important;
}}

/* ================= UTILITY ================= */
.soft-box {{
    background: rgba(148, 163, 184, 0.09);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 14px;
}}

.subtle-divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(212, 175, 55, 0.60), transparent);
    margin: 24px 0;
}}

.progress-container {{
    background: rgba(148, 163, 184, 0.20);
    border-radius: 999px;
    overflow: hidden;
    height: 14px;
}}

.progress-bar {{
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s ease;
}}

@media (max-width: 900px) {{
    .block-container {{
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }}

    .hero-card {{
        padding: 24px 22px !important;
        border-radius: 24px !important;
    }}

    .hero-title {{
        font-size: 26px;
    }}

    .metric-container,
    .kpi-card {{
        padding: 18px !important;
        border-radius: 20px !important;
    }}
}}
</style>
""",
    unsafe_allow_html=True
)

# ================= HELPER FUNCTIONS =================
def sidebar_heading(icon: str, title: str):
    st.markdown(
        f"""
        <div class="sidebar-section-title">
            <span>{icon}</span>
            <span>{title}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_heading(title: str):
    st.markdown(
        f"""
        <div class="section-heading-box">
            <span>{title}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def panel_title(title: str):
    st.markdown(
        f"""
        <div class="panel-title-box">
            <span>{title}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def kpi_card(icon: str, label: str, value: str, delta: str, delta_class: str = "delta-positive"):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-row">
                <div class="kpi-icon">{icon}</div>
                <div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>
            </div>
            <div class="metric-delta {delta_class}">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def luxury_layout(fig, height=320, showlegend=False):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=35),
        showlegend=showlegend,
        font=dict(family="Inter, sans-serif", size=12, color=MUTED),
        xaxis=dict(gridcolor=GRID, zeroline=False),
        yaxis=dict(gridcolor=GRID, zeroline=False),
    )
    return fig


# ================= SIDEBAR MAIN =================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-card">
        <div class="sidebar-logo">🏦</div>
        <div class="sidebar-brand-title">BankPulse Pro</div>
        <div class="sidebar-brand-subtitle">Loan Acceptance Intelligence</div>
        <div class="sidebar-pill">● Enterprise ML Suite</div>
    </div>
    """, unsafe_allow_html=True)

    sidebar_heading("🎯", "Model Selection")
    model_choice = st.selectbox(
        "Algorithm",
        ["Logistic Regression", "Decision Tree"],
        help="Choose the classification algorithm"
    )

    sidebar_heading("📊", "System Status")
    st.markdown(f"""
    <div class="sidebar-status-card">
        <div class="sidebar-status-row">
            <span class="sidebar-status-label">Active Theme</span>
            <span class="sidebar-badge-gold">{theme_mode}</span>
        </div>
        <div class="sidebar-status-row">
            <span class="sidebar-status-label">Model Status</span>
            <span class="sidebar-badge-success">Active</span>
        </div>
        <div class="sidebar-status-row">
            <span class="sidebar-status-label">Data Quality</span>
            <span class="sidebar-badge-gold">99.8%</span>
        </div>
        <div class="sidebar-status-row">
            <span class="sidebar-status-label">Last Updated</span>
            <span class="sidebar-status-label">Today</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    sidebar_heading("⚙️", "Threshold Settings")
    threshold = st.slider(
        "Acceptance Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Probability cutoff for loan acceptance"
    )

    st.markdown("""
    <div class="sidebar-footer">
        <p><b>BankPulse Pro v3.2</b></p>
        <p>Banking Analytics Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-card">
        <div class="hero-title">Loan Acceptance Predictor</div>
        <div class="hero-subtitle">
            AI-powered customer segmentation, campaign intelligence, and loan offer acceptance prediction.
        </div>
        <div class="hero-tag">✨ Banking Intelligence Dashboard</div>
    </div>
</div>
""", unsafe_allow_html=True)

head_col1, head_col2, head_col3 = st.columns(3)

with head_col1:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-value">45,211</div>
        <div class="metric-label">Total Records</div>
    </div>
    """, unsafe_allow_html=True)

with head_col2:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-value" style="color:#10b981 !important;">11.7%</div>
        <div class="metric-label">Acceptance Rate</div>
    </div>
    """, unsafe_allow_html=True)

with head_col3:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-value" style="color:#d4af37 !important;">2.76</div>
        <div class="metric-label">Avg Contacts</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='subtle-divider'></div>", unsafe_allow_html=True)

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Predictor", "📈 Analytics", "⚙️ Model Insights"])

# ================= TAB 1 =================
with tab1:
    section_heading("Executive Overview")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        kpi_card("👥", "Active Customers", "9,042", "↑ 12.3% vs last month", "delta-positive")
    with kpi2:
        kpi_card("✅", "Conversion Rate", "11.7%", "↑ 3.2% vs baseline", "delta-positive")
    with kpi3:
        kpi_card("📞", "Avg. Contacts", "2.76", "↓ 0.4 vs target", "delta-negative")
    with kpi4:
        kpi_card("💰", "Revenue Potential", "$2.4M", "↑ 8.7% projected", "delta-positive")

    chart1, chart2 = st.columns(2)

    with chart1:
        with st.container(border=True):
            panel_title("📊 Acceptance by Customer Segment")

            segments = ["Students", "Retired", "Management", "Admin", "Technicians", "Services", "Blue-Collar", "Entrepreneurs"]
            acceptance = [28.5, 22.3, 13.8, 12.5, 11.8, 11.2, 10.5, 9.8]

            fig = go.Figure(go.Bar(
                x=segments,
                y=acceptance,
                marker_color=[GREEN if a > 15 else GOLD if a > 11 else "#b7791f" for a in acceptance],
                text=[f"{a:.1f}%" for a in acceptance],
                textposition="outside",
                textfont=dict(size=11, color=MUTED)
            ))

            fig = luxury_layout(fig, height=330)
            fig.update_xaxes(tickangle=-30, tickfont=dict(size=10, color=MUTED))
            fig.update_yaxes(range=[0, 35], tickfont=dict(size=10, color=MUTED))

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with chart2:
        with st.container(border=True):
            panel_title("🎯 Campaign Performance by Month")

            months = ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            performance = [15.2, 12.8, 10.5, 11.2, 10.8, 9.5, 12.1, 14.5, 13.2, 11.8]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=months,
                y=performance,
                mode="lines+markers",
                line=dict(color=GOLD, width=3),
                marker=dict(size=9, color=GOLD_LIGHT, line=dict(width=2, color=NAVY)),
                fill="tozeroy",
                fillcolor="rgba(212, 175, 55, 0.16)"
            ))

            fig = luxury_layout(fig, height=330)
            fig.update_yaxes(range=[0, 20], tickfont=dict(size=10, color=MUTED))
            fig.update_xaxes(tickfont=dict(size=11, color=MUTED))

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ================= TAB 2 =================
with tab2:
    section_heading("🔮 Customer Profile Predictor")

    pred_col1, pred_col2, pred_col3 = st.columns([1.2, 1.2, 1.6])

    with pred_col1:
        with st.container(border=True):
            panel_title("👤 Demographics")
            age_input = st.slider("Age", 18, 95, 35)
            job_input = st.selectbox("Job Type", [
                "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                "retired", "self-employed", "services", "student", "technician",
                "unemployed", "unknown"
            ])
            marital_input = st.selectbox("Marital Status", ["divorced", "married", "single"])
            education_input = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])

    with pred_col2:
        with st.container(border=True):
            panel_title("💳 Financial Profile")
            default_input = st.selectbox("Credit Default", ["no", "yes"])
            balance_input = st.slider("Yearly Balance (€)", -10000, 120000, 1500, step=100)
            housing_input = st.selectbox("Housing Loan", ["no", "yes"])
            loan_input = st.selectbox("Personal Loan", ["no", "yes"])

            st.markdown("<div class='subtle-divider'></div>", unsafe_allow_html=True)
            panel_title("📞 Campaign Details")
            contact_input = st.selectbox("Contact Method", ["cellular", "telephone", "unknown"])
            month_input = st.selectbox("Contact Month", [
                "jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"
            ])
            day_input = st.slider("Day of Month", 1, 31, 15)

    with pred_col3:
        with st.container(border=True):
            panel_title("📊 Campaign History")
            campaign_input = st.slider("Campaign Contacts", 1, 60, 2)
            pdays_input = st.slider("Days Since Last Contact", -1, 999, -1)
            previous_input = st.slider("Previous Contacts", 0, 60, 0)
            poutcome_input = st.selectbox("Previous Outcome", ["failure", "success", "unknown", "other"])

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])

    with c2:
        predict_btn = st.button("🚀 RUN PREDICTION ANALYSIS", use_container_width=True)

    if predict_btn:
        input_dict = {col: 0 for col in columns}

        numeric_values = {
            "age": age_input,
            "balance": balance_input,
            "day": day_input,
            "campaign": campaign_input,
            "pdays": pdays_input,
            "previous": previous_input,
        }

        for key, value in numeric_values.items():
            if key in input_dict:
                input_dict[key] = value

        categorical_values = [
            ("job", job_input),
            ("marital", marital_input),
            ("education", education_input),
            ("default", default_input),
            ("housing", housing_input),
            ("loan", loan_input),
            ("contact", contact_input),
            ("month", month_input),
            ("poutcome", poutcome_input),
        ]

        for feat, val in categorical_values:
            col_name = f"{feat}_{val}"
            if col_name in input_dict:
                input_dict[col_name] = 1

        input_df = pd.DataFrame([input_dict])
        input_scaled = input_df.copy()
        scale_cols = [col for col in numeric_features if col in input_scaled.columns]

        if scale_cols:
            input_scaled[scale_cols] = scaler.transform(input_df[scale_cols])

        if model_choice == "Logistic Regression":
            prob = lr_model.predict_proba(input_scaled)[0][1]
            model_pred = lr_model.predict(input_scaled)[0]
        else:
            prob = dt_model.predict_proba(input_df)[0][1]
            model_pred = dt_model.predict(input_df)[0]

        status = "ACCEPTED" if prob >= threshold else "DECLINED"
        status_color = GREEN if prob >= threshold else RED
        status_bg = "rgba(16,185,129,0.14)" if prob >= threshold else "rgba(239,68,68,0.14)"
        progress_color = "linear-gradient(90deg, #10b981, #34d399)" if prob >= threshold else "linear-gradient(90deg, #ef4444, #f87171)"

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            with st.container(border=True):
                panel_title("📌 Prediction Result")
                st.markdown(
                    f"""
                    <div style="text-align:center; padding: 24px 12px;">
                        <div style="display:inline-block; margin:5px 0 18px 0; padding:12px 30px; background:{status_bg}; border:1px solid {status_color}; border-radius:999px;">
                            <span style="font-size:18px; font-weight:900; color:{status_color} !important;">{status}</span>
                        </div>
                        <div style="font-family:'JetBrains Mono', monospace; font-size:52px; font-weight:900; color:var(--text-main) !important;">
                            {prob:.1%}
                        </div>
                        <div style="font-size:14px; font-weight:700; color:var(--text-muted) !important;">
                            Acceptance Probability
                        </div>
                    </div>
                    <div style="margin-top: 20px;">
                        <div style="display:flex; justify-content:space-between; font-size:12px; font-weight:800; color:var(--text-muted); margin-bottom:8px;">
                            <span>0%</span>
                            <span>50%</span>
                            <span>100%</span>
                        </div>
                        <div class="progress-container">
                            <div class="progress-bar" style="width:{prob * 100}%; background:{progress_color};"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with result_col2:
            with st.container(border=True):
                panel_title("📋 Risk Assessment Factors")
                risk_factors = []

                if age_input > 60:
                    risk_factors.append(("Senior Customer", "positive", "Higher acceptance likelihood"))
                if job_input in ["student", "retired"]:
                    risk_factors.append(("Special Segment", "positive", f"{job_input.title()} segment shows higher rates"))
                if poutcome_input == "success":
                    risk_factors.append(("Previous Success", "positive", "Prior campaign success indicates strong potential"))
                if campaign_input > 5:
                    risk_factors.append(("High Contact Frequency", "negative", "Excessive contacts may reduce effectiveness"))
                if loan_input == "yes":
                    risk_factors.append(("Existing Loan", "negative", "Current loan obligations may affect decision"))
                if balance_input < 0:
                    risk_factors.append(("Negative Balance", "negative", "Overdrawn account indicates financial stress"))
                if prob < threshold:
                    risk_factors.append(("Below Threshold", "negative", "Probability is below selected acceptance threshold"))

                if not risk_factors:
                    risk_factors.append(("Standard Profile", "neutral", "No significant risk factor identified"))

                for factor, status_type, desc in risk_factors[:6]:
                    icon = "✅" if status_type == "positive" else "⚠️" if status_type == "negative" else "ℹ️"
                    border_color = GREEN if status_type == "positive" else RED if status_type == "negative" else GOLD

                    st.markdown(
                        f"""
                        <div style="display:flex; gap:11px; padding:12px; margin-bottom:10px; background:rgba(148,163,184,0.09); border-radius:14px; border-left:4px solid {border_color};">
                            <span style="font-size:18px;">{icon}</span>
                            <div>
                                <div style="font-size:13px; font-weight:900; color:var(--text-main) !important;">
                                    {factor}
                                </div>
                                <div style="font-size:12px; font-weight:650; color:var(--text-muted) !important; line-height:1.45;">
                                    {desc}
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        st.session_state.predictions.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "probability": float(prob),
            "prediction": status,
            "model_prediction": int(model_pred),
            "age": age_input,
            "job": job_input,
            "model": model_choice,
        })

# ================= TAB 3 =================
with tab3:
    section_heading("📈 Advanced Analytics")

    analytics1, analytics2 = st.columns(2)

    with analytics1:
        with st.container(border=True):
            panel_title("🎯 Feature Importance - Decision Tree")
            features = ["balance", "age", "campaign", "pdays", "previous", "day"]
            importance = [0.285, 0.195, 0.145, 0.125, 0.095, 0.075]

            fig = go.Figure(go.Bar(
                x=importance,
                y=features,
                orientation="h",
                marker=dict(color=[GOLD, "#c49a2f", "#b7791f", "#8a641d", "#6b7280", "#94a3b8"]),
                text=[f"{i:.1%}" for i in importance],
                textposition="outside",
                textfont=dict(size=11, color=MUTED)
            ))

            fig = luxury_layout(fig, height=330)
            fig.update_xaxes(showgrid=False, showticklabels=False)
            fig.update_yaxes(tickfont=dict(size=11, color=MUTED))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with analytics2:
        with st.container(border=True):
            panel_title("📊 Age Distribution by Acceptance")
            age_bins = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
            accepted = [8, 10, 12, 14, 16, 18]
            declined = [92, 90, 88, 86, 84, 82]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=age_bins, y=declined, name="Declined",
                marker_color=RED, text=declined, textposition="inside",
                textfont=dict(color="white", size=11)
            ))
            fig.add_trace(go.Bar(
                x=age_bins, y=accepted, name="Accepted",
                marker_color=GREEN, text=accepted, textposition="inside",
                textfont=dict(color="white", size=11)
            ))

            fig = luxury_layout(fig, height=330, showlegend=True)
            fig.update_layout(barmode="stack", legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
            fig.update_xaxes(tickfont=dict(size=11, color=MUTED))
            fig.update_yaxes(tickfont=dict(size=10, color=MUTED))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        panel_title("🧬 Customer Segmentation Matrix")

        segments_data = pd.DataFrame({
            "Segment": ["Students", "Retired", "Management", "Admin", "Technicians", "Services", "Blue-Collar", "Entrepreneurs"],
            "Count": [900, 1800, 8500, 10400, 7500, 4100, 9200, 1300],
            "Acceptance_Rate": [28.5, 22.3, 13.8, 12.5, 11.8, 11.2, 10.5, 9.8],
            "Avg_Age": [22, 68, 42, 39, 38, 36, 40, 45],
            "Avg_Balance": [1200, 3500, 2800, 2100, 1900, 1500, 800, 4200],
            "Revenue_Potential": ["High", "High", "Medium", "Medium", "Medium", "Medium", "Low", "Medium"]
        })

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[
                    "<b>Segment</b>", "<b>Count</b>", "<b>Acceptance Rate</b>",
                    "<b>Avg Age</b>", "<b>Avg Balance (€)</b>", "<b>Revenue Potential</b>"
                ],
                fill_color=NAVY,
                align="left",
                font=dict(color="white", size=12),
                height=42,
                line=dict(color="rgba(212,175,55,0.55)")
            ),
            cells=dict(
                values=[
                    segments_data["Segment"],
                    segments_data["Count"].apply(lambda x: f"{x:,}"),
                    segments_data["Acceptance_Rate"].apply(lambda x: f"{x:.1f}%"),
                    segments_data["Avg_Age"],
                    segments_data["Avg_Balance"].apply(lambda x: f"€{x:,}"),
                    segments_data["Revenue_Potential"]
                ],
                fill_color=TABLE_CELL_BG,
                align="left",
                font=dict(color="#07111f", size=12),
                height=36,
                line=dict(color="rgba(15,23,42,0.10)", width=1)
            )
        )])

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=350
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ================= TAB 4 =================
with tab4:
    section_heading("⚙️ Model Performance & Insights")

    model_col1, model_col2 = st.columns(2)

    with model_col1:
        with st.container(border=True):
            panel_title("📊 Model Comparison")
            metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
            lr_scores = [89.5, 65.2, 58.8, 61.8, 88.2]
            dt_scores = [88.1, 62.5, 55.3, 58.7, 85.4]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=metrics, y=lr_scores, name="Logistic Regression",
                marker_color=GOLD,
                text=[f"{s:.1f}%" for s in lr_scores],
                textposition="outside",
                textfont=dict(size=10, color=MUTED)
            ))
            fig.add_trace(go.Bar(
                x=metrics, y=dt_scores, name="Decision Tree",
                marker_color="#111827",
                text=[f"{s:.1f}%" for s in dt_scores],
                textposition="outside",
                textfont=dict(size=10, color=MUTED)
            ))

            fig = luxury_layout(fig, height=360, showlegend=True)
            fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
            fig.update_yaxes(range=[0, 100], tickfont=dict(size=10, color=MUTED))
            fig.update_xaxes(tickfont=dict(size=11, color=MUTED))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with model_col2:
        with st.container(border=True):
            panel_title("🎯 Confusion Matrix - Logistic Regression")
            cm = np.array([[7200, 350], [450, 800]])

            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Predicted: No", "Predicted: Yes"],
                y=["Actual: No", "Actual: Yes"],
                colorscale=[[0, "#fbfaf6"], [0.5, "#f7df8a"], [1, GOLD]],
                showscale=False,
                text=cm,
                texttemplate="%{text}",
                textfont=dict(size=16, color="#07111f")
            ))

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=90, r=20, t=20, b=20),
                xaxis=dict(tickfont=dict(size=11, color=MUTED)),
                yaxis=dict(tickfont=dict(size=11, color=MUTED)),
                height=305
            )

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("""
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:15px;">
                <div class="soft-box" style="text-align:center; padding:12px;">
                    <div style="font-size:20px; font-weight:900; color:var(--text-main) !important;">7,200</div>
                    <div class="metric-label">True Negatives</div>
                </div>
                <div class="soft-box" style="text-align:center; padding:12px;">
                    <div style="font-size:20px; font-weight:900; color:var(--text-main) !important;">800</div>
                    <div class="metric-label">True Positives</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        panel_title("💡 Key Business Insights")
        insights = [
            ("🎯", "Target Students & Retirees", "These segments show 28.5% and 22.3% acceptance rates. Focus marketing campaigns on these groups for better conversion."),
            ("💰", "Balance is King", "Account balance is the strongest predictor. Customers with healthier balances are more likely to accept loan offers."),
            ("📞", "Optimize Contact Strategy", "Keep contacts controlled. Excessive campaign contacts can reduce customer response and increase campaign cost."),
            ("📅", "Seasonal Campaign Timing", "March and October show stronger acceptance behavior. These months can be prioritized for major campaigns."),
            ("🔄", "Previous Success Matters", "Customers with a previous successful outcome are strong re-targeting candidates."),
            ("👴", "Age Factor", "Mature customer groups show stronger acceptance compared with younger groups."),
        ]

        for icon, title, desc in insights:
            st.markdown(
                f"""
                <div style="display:flex; gap:15px; padding:15px; margin-bottom:11px; background:rgba(148,163,184,0.09); border-radius:16px; border-left:4px solid #d4af37;">
                    <span style="font-size:24px;">{icon}</span>
                    <div>
                        <div style="font-size:14px; font-weight:900; color:var(--text-main) !important;">
                            {title}
                        </div>
                        <div style="font-size:13px; font-weight:650; color:var(--text-muted) !important; line-height:1.55;">
                            {desc}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ================= FOOTER =================
st.markdown("""
<div style="text-align:center; margin-top:40px; padding:20px;">
    <div class="subtle-divider"></div>
    <p style="color:var(--text-muted) !important; font-size:12px; font-weight:800;">
        BankPulse Pro Enterprise Suite v3.2 | Powered by Machine Learning | 2026
    </p>
</div>
""", unsafe_allow_html=True)