"""
app.py — EV Intelligence Platform entry point
Run:  python -m streamlit run ev_intelligence/app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

st.set_page_config(
    page_title="EV Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data once ─────────────────────────────────────────────────────────────
from engine.data_loader import load_all

@st.cache_data(show_spinner="Loading & cleaning all datasets…")
def get_data():
    return load_all()

data = get_data()

# ── Dark / Light mode toggle ───────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# Sidebar brand + toggle
st.sidebar.markdown('<div class="sidebar-brand">EV Intelligence</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-sub">Global Market Expansion & Site Selection</div>', unsafe_allow_html=True)

dark_mode = st.sidebar.toggle("Dark mode", value=st.session_state.dark_mode, key="dark_mode")

# ── Dynamic CSS based on mode ─────────────────────────────────────────────────
if dark_mode:
    _bg        = "#0d1117"
    _surface   = "#161b22"
    _border    = "#30363d"
    _text      = "#e6edf3"
    _muted     = "#8b949e"
    _accent    = "#58a6ff"
    _metric_v  = "#58a6ff"
    _card_bg   = "#161b22"
    _sidebar_bg = "#161b22"
    _tab_sel   = "#1f6feb"
    _grid      = "#21262d"
else:
    _bg        = "#ffffff"
    _surface   = "#f6f8fa"
    _border    = "#d0d7de"
    _text      = "#1f2328"
    _muted     = "#57606a"
    _accent    = "#0550ae"
    _metric_v  = "#0550ae"
    _card_bg   = "#f6f8fa"
    _sidebar_bg = "#f6f8fa"
    _tab_sel   = "#0550ae"
    _grid      = "#d0d7de"

st.markdown(f"""
<style>
  [data-testid="stAppViewContainer"] {{ background: {_bg}; color: {_text}; }}
  [data-testid="stSidebar"]          {{ background: {_sidebar_bg}; border-right: 1px solid {_border}; }}
  [data-testid="stHeader"]           {{ background: transparent; }}
  .stApp                             {{ background: {_bg}; }}

  /* Sidebar brand */
  .sidebar-brand {{
    padding: 1.2rem 1rem 0.5rem;
    font-size: 1.35rem;
    font-weight: 700;
    color: {_accent};
    letter-spacing: -0.5px;
  }}
  .sidebar-sub {{
    padding: 0 1rem 1rem;
    font-size: 0.78rem;
    color: {_muted};
    border-bottom: 1px solid {_border};
    margin-bottom: 0.5rem;
  }}

  /* Metric cards */
  [data-testid="metric-container"] {{
    background: {_card_bg};
    border: 1px solid {_border};
    border-radius: 10px;
    padding: 1rem 1.2rem;
  }}
  [data-testid="metric-container"] label {{ color: {_muted} !important; font-size: 0.78rem !important; }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{ color: {_metric_v} !important; font-size: 1.6rem !important; font-weight: 700 !important; }}

  /* Page header */
  .page-header {{
    background: {_card_bg};
    border: 1px solid {_border};
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
  }}
  .page-header h1 {{ color: {_text} !important; font-size: 1.75rem !important; font-weight: 700 !important; margin: 0 0 0.3rem !important; }}
  .page-header p  {{ color: {_muted} !important; font-size: 0.9rem !important; margin: 0 !important; }}

  /* Force all heading text to be visible */
  h1, h2, h3, h4 {{ color: {_text} !important; }}
  p, li, span, div {{ color: inherit; }}
  [data-testid="stMarkdownContainer"] p {{ color: {_text}; }}
  [data-testid="stMarkdownContainer"] h1,
  [data-testid="stMarkdownContainer"] h2,
  [data-testid="stMarkdownContainer"] h3 {{ color: {_text} !important; }}

  /* Section headers */
  .section-title {{
    color: {_text};
    font-size: 1.1rem;
    font-weight: 600;
    border-left: 3px solid {_accent};
    padding-left: 0.75rem;
    margin: 1.5rem 0 0.75rem;
  }}

  /* Tables */
  [data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}

  /* Tab styling */
  [data-baseweb="tab-list"] {{ background: {_card_bg} !important; border-radius: 8px; padding: 4px; gap: 2px; border-bottom: 1px solid {_border} !important; }}
  [data-baseweb="tab"]      {{ color: {_muted} !important; border-radius: 6px !important; font-size: 0.85rem !important; padding: 6px 14px !important; white-space: nowrap !important; }}
  [data-baseweb="tab"]:hover {{ color: {_text} !important; background: {_surface} !important; }}
  [data-baseweb="tab"][aria-selected="true"] {{ background: {_tab_sel} !important; color: #ffffff !important; font-weight: 600 !important; }}
  button[data-baseweb="tab"] {{ color: {_muted} !important; }}
  button[data-baseweb="tab"][aria-selected="true"] {{ color: #fff !important; background: {_tab_sel} !important; }}

  /* Divider */
  hr {{ border-color: {_border} !important; }}

  /* Plotly chart bg fix */
  .js-plotly-plot .plotly {{ border-radius: 10px; }}

  /* Info/warning boxes */
  [data-testid="stAlert"] {{ border-radius: 8px; }}

  /* Buttons — always white text on blue */
  [data-testid="baseButton-primary"] {{ background: #1f6feb !important; border-color: #1f6feb !important; border-radius: 6px !important; color: #ffffff !important; }}
  [data-testid="baseButton-secondary"] {{ border-radius: 6px !important; color: {_text} !important; }}
  button[kind="primary"] {{ color: #ffffff !important; }}
</style>
""", unsafe_allow_html=True)

# ── Inject data into builtins so all exec'd pages can access it ────────────────
import builtins
builtins.data = data
builtins.st   = st

# ── Sidebar nav ───────────────────────────────────────────────────────────────
PAGES = {
    "Overview":                  "pages/00_overview.py",
    "EDA & Market Trends":       "pages/01_eda.py",
    "Demand & Charging Map":     "pages/02_map_demand.py",
    "Growth Forecast":           "pages/03_forecast.py",
    "Plant Site Clustering":     "pages/04_clustering.py",
    "Supply Chain Map":          "pages/05_map_supply.py",
    "Policy & Tariff RAG":       "pages/06_rag_policy.py",
}

selection = st.sidebar.radio("", list(PAGES.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")
ev   = data["ev_population"]
sta  = data["ev_stations"]
st.sidebar.markdown(f"""
<div style='font-size:0.75rem;color:{_muted};padding:0 0.5rem;'>
  <div style='margin-bottom:4px;'><b style='color:{_text}'>{len(ev):,}</b> EV registrations</div>
  <div style='margin-bottom:4px;'><b style='color:{_text}'>{len(sta):,}</b> charging stations</div>
  <div style='margin-bottom:4px;'><b style='color:{_text}'>{sta["country"].nunique()}</b> countries covered</div>
</div>
""", unsafe_allow_html=True)

# ── Route to page ──────────────────────────────────────────────────────────────
page_path = os.path.join(os.path.dirname(__file__), PAGES[selection])
with open(page_path, "r", encoding="utf-8") as fh:
    code = fh.read()

exec(compile(code, page_path, "exec"), {
    "data": data,
    "st":   st,
    "__file__": page_path,
    "__name__": "__main__",
})
