"""pages/00_overview.py"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ev   = data["ev_population"]
sta  = data["ev_stations"]
ports= data["ports"]
mc   = data["mining_companies"]
comm = data["mining_commodities"]

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h1>Global EV Market Expansion &amp; Site Selection</h1>
  <p>End-to-end intelligence platform — EDA · Machine Learning · Geospatial Analysis · RAG Policy Simulation</p>
</div>
""", unsafe_allow_html=True)

# ── KPI cards ────────────────────────────────────────────────────────────────
ev_comm = comm[comm["ev_relevant"]] if "ev_relevant" in comm.columns else comm
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("EV Registrations",   f"{len(ev):,}")
c2.metric("Charging Stations",  f"{len(sta):,}")
c3.metric("World Ports",        f"{len(ports):,}")
c4.metric("Mining Companies",   f"{len(mc):,}")
c5.metric("EV Mineral Records", f"{len(ev_comm):,}")

st.markdown("---")

# ── Three summary charts ──────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

CHART_LAYOUT = dict(
    paper_bgcolor="#161b22",
    plot_bgcolor="#161b22",
    font_color="#e6edf3",
    margin=dict(t=10, b=10, l=10, r=10),
    height=220,
)

HOVER_STYLE = dict(
    hoverlabel=dict(bgcolor="#161b22", bordercolor="#58a6ff",
                    font_size=13, font_color="#e6edf3")
)

# Country code → full name mapping for display
COUNTRY_NAMES = {
    "US": "United States", "CN": "China", "DE": "Germany",
    "GB": "United Kingdom", "FR": "France", "NO": "Norway",
    "NL": "Netherlands", "KR": "South Korea", "AU": "Australia",
    "CA": "Canada", "JP": "Japan", "IN": "India", "SE": "Sweden",
    "AT": "Austria", "BE": "Belgium", "CH": "Switzerland",
    "DK": "Denmark", "FI": "Finland", "IT": "Italy", "NZ": "New Zealand",
    "PL": "Poland", "PT": "Portugal", "ES": "Spain",
}

with col1:
    st.markdown('<div class="section-title">EV Type Split</div>', unsafe_allow_html=True)
    if "ev_type" in ev.columns:
        vc = ev["ev_type"].value_counts().reset_index()
        vc.columns = ["Type","Count"]
        fig = px.pie(vc, names="Type", values="Count", hole=0.55,
                     color_discrete_sequence=["#58a6ff","#3fb950","#f0883e"])
        fig.update_layout(**CHART_LAYOUT, showlegend=True,
                          legend=dict(font_size=11, bgcolor="rgba(0,0,0,0)"))
        fig.update_traces(
            textfont_color="#e6edf3",
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
            **{k: v for k, v in HOVER_STYLE.items()},
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<div class="section-title">Top Countries — Charging Stations</div>', unsafe_allow_html=True)
    top_c = sta["country"].value_counts().head(8).reset_index()
    top_c.columns = ["Country","Stations"]
    # Expand country codes to full names where known
    top_c["Country"] = top_c["Country"].apply(lambda c: COUNTRY_NAMES.get(str(c).upper(), str(c)))
    fig2 = px.bar(top_c, x="Stations", y="Country", orientation="h",
                  color="Stations", color_continuous_scale=["#1f3a5c","#58a6ff"])
    fig2.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, yaxis_title="")
    fig2.update_traces(
        hovertemplate="<b>%{y}</b><br>Stations: %{x:,}<extra></extra>",
        **{k: v for k, v in HOVER_STYLE.items()},
    )
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    st.markdown('<div class="section-title">Top EV-Critical Minerals</div>', unsafe_allow_html=True)
    top_m = ev_comm.groupby("mineral")["production"].sum().nlargest(8).reset_index()
    top_m.columns = ["Mineral","Production"]
    top_m["Mineral"] = top_m["Mineral"].str.title()
    fig3 = px.bar(top_m, x="Production", y="Mineral", orientation="h",
                  color="Production", color_continuous_scale=["#1a3a2a","#3fb950"])
    fig3.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, yaxis_title="")
    fig3.update_traces(
        hovertemplate="<b>%{y}</b><br>Production: %{x:,.0f} t<extra></extra>",
        **{k: v for k, v in HOVER_STYLE.items()},
    )
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ── Dataset detail cards ──────────────────────────────────────────────────────
st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

cards = [
    ("EV Population",     f"{len(ev):,} registrations",        f"{ev['model_year'].nunique()} model years · {ev['make'].nunique()} brands · Washington State",     "#58a6ff"),
    ("Charging Stations",  f"{len(sta):,} stations",            f"{sta['country'].nunique()} countries · Operational only · Global coverage",                         "#3fb950"),
    ("World Port Index",   f"{len(ports):,} ports",             f"NGA dataset · Very Large to Small harbours · Global shipping nodes",                               "#f0883e"),
    ("Mining Companies",   f"{len(mc):,} companies",            f"EV-relevant minerals flagged · {mc['ev_relevant'].sum()} EV-critical entries",                     "#bc8cff"),
    ("Mining Commodities", f"{len(ev_comm):,} EV-mineral rows", f"{comm['country'].nunique()} countries · {comm['mineral'].nunique()} minerals · 2018–2022",         "#39c5cf"),
]

cols = st.columns(len(cards))
for col, (title, value, detail, color) in zip(cols, cards):
    col.markdown(f"""
    <div style="background:#161b22;border:1px solid #30363d;border-top:3px solid {color};
                border-radius:10px;padding:1rem 1.1rem;height:140px;">
      <div style="font-size:0.85rem;font-weight:600;color:{color};margin-bottom:0.3rem;">{title}</div>
      <div style="font-size:1.25rem;font-weight:700;color:#e6edf3;margin-bottom:0.4rem;">{value}</div>
      <div style="font-size:0.73rem;color:#8b949e;line-height:1.4;">{detail}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Architecture diagram ──────────────────────────────────────────────────────
st.markdown('<div class="section-title">System Architecture</div>', unsafe_allow_html=True)
st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:1.5rem;font-family:monospace;font-size:0.82rem;color:#e6edf3;line-height:1.8;">
<span style="color:#58a6ff;">RAW DATA LAYER</span><br>
&nbsp;&nbsp;EV Registrations (279K) &nbsp;·&nbsp; Charging Stations (9K) &nbsp;·&nbsp; Port Index (3.6K) &nbsp;·&nbsp; Mining Commodities<br><br>
<span style="color:#30363d;">──────────────────────────────────────────────</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼ &nbsp;<span style="color:#3fb950;">data_loader.py</span>&nbsp; (clean · validate · coordinate-parse)<br><br>
<span style="color:#f0883e;">ANALYTICS LAYER</span>&nbsp;&nbsp;<span style="color:#3fb950;">ml_engine.py</span><br>
&nbsp;&nbsp;EDA Trends &nbsp;·&nbsp; Ridge Regression Forecast &nbsp;·&nbsp; K-Means Clustering &nbsp;·&nbsp; Viability Scoring<br><br>
<span style="color:#30363d;">──────────────────────────────────────────────</span><br>
&nbsp;&nbsp;&nbsp;▼ Map 1 (Demand) &nbsp;&nbsp;&nbsp;▼ Plant GPS Centroids &nbsp;&nbsp;&nbsp;▼ Map 2 (Supply Chain)<br><br>
<span style="color:#bc8cff;">RAG LAYER</span>&nbsp;&nbsp;<span style="color:#3fb950;">rag_engine.py</span><br>
&nbsp;&nbsp;ChromaDB Vector Store &nbsp;·&nbsp; all-MiniLM-L6-v2 Embeddings &nbsp;·&nbsp; Policy Docs &nbsp;·&nbsp; Tariff Simulation
</div>
""", unsafe_allow_html=True)
