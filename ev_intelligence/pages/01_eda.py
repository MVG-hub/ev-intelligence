"""pages/01_eda.py"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from engine.ml_engine import ev_type_trend, price_vs_range, brand_market_share, range_evolution

ev   = data["ev_population"]
comm = data["mining_commodities"]

DARK = dict(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#e6edf3",
            xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"))

HOVER_STYLE = dict(
    hoverlabel=dict(bgcolor="#161b22", bordercolor="#58a6ff",
                    font_size=13, font_color="#e6edf3")
)

st.markdown("""
<div class="page-header">
  <h1>EDA &amp; Market Trends</h1>
  <p>Deep exploratory analysis of EV adoption, brand positioning, range evolution, and global mineral supply.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Adoption Trend",
    "Range by Brand",
    "Brand Share",
    "Range Evolution",
    "Minerals",
])

# ── Tab 1: BEV vs PHEV ────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">BEV vs PHEV Year-over-Year Registrations</div>', unsafe_allow_html=True)
    trend = ev_type_trend(ev)
    if not trend.empty:
        fig = go.Figure()
        palette = {"BEV": "#58a6ff", "PHEV": "#3fb950", "UNKNOWN": "#8b949e"}
        fill_colors = {"BEV": "rgba(88,166,255,0.08)", "PHEV": "rgba(63,185,80,0.08)", "UNKNOWN": "rgba(139,148,158,0.08)"}
        for col in trend.columns:
            fig.add_trace(go.Scatter(
                x=trend.index.astype(int), y=trend[col].astype(int),
                mode="lines+markers", name=str(col),
                line=dict(color=palette.get(str(col), "#bc8cff"), width=2.5),
                marker=dict(size=7),
                fill="tozeroy",
                fillcolor=fill_colors.get(str(col), "rgba(188,140,255,0.08)"),
                hovertemplate=f"<b>{col}</b><br>Year: %{{x}}<br>Registrations: %{{y:,}}<extra></extra>",
            ))
        fig.update_layout(**DARK, height=420, hovermode="x unified",
                          xaxis_title="Model Year", yaxis_title="Registrations",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                      bgcolor="rgba(0,0,0,0)"),
                          **HOVER_STYLE)
        st.plotly_chart(fig, use_container_width=True)

        # YoY growth table
        if "BEV" in trend.columns:
            yoy = trend["BEV"].pct_change().dropna() * 100
            yoy_df = yoy.reset_index()
            yoy_df.columns = ["Model Year", "BEV YoY Growth (%)"]
            yoy_df["Model Year"] = yoy_df["Model Year"].astype(int)
            yoy_df["BEV YoY Growth (%)"] = yoy_df["BEV YoY Growth (%)"].round(1)
            yoy_df["Trend"] = yoy_df["BEV YoY Growth (%)"].apply(lambda x: "▲" if x > 0 else "▼")
            st.dataframe(yoy_df.tail(12), use_container_width=True, hide_index=True)

# ── Tab 2: Range by Brand ─────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Electric Range Distribution by Brand (Top 12)</div>', unsafe_allow_html=True)
    pvr = price_vs_range(ev, top_n_makes=12)
    if not pvr.empty:
        order = (pvr.groupby("make")["electric_range"]
                    .median()
                    .sort_values(ascending=False)
                    .index.tolist())
        fig2 = px.box(
            pvr, x="make", y="electric_range", color="make",
            category_orders={"make": order},
            labels={"electric_range": "Electric Range (miles)", "make": "Brand"},
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig2.update_layout(**DARK, height=440, showlegend=False,
                           xaxis_tickangle=-30, xaxis_title="Brand",
                           yaxis_title="Electric Range (miles)",
                           **HOVER_STYLE)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-title">Brand Range Statistics</div>', unsafe_allow_html=True)
        stats = pvr.groupby("make")["electric_range"].agg(
            Median="median", Mean="mean", Max="max", Count="count"
        ).round(1).sort_values("Median", ascending=False).reset_index()
        stats.columns = ["Brand", "Median (mi)", "Mean (mi)", "Max (mi)", "# Vehicles"]
        st.dataframe(stats, use_container_width=True, hide_index=True)

# ── Tab 3: Brand Market Share ─────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Brand Market Share — Top 20 Makes</div>', unsafe_allow_html=True)
    bms = brand_market_share(ev, top_n=20)
    fig3 = px.bar(
        bms.sort_values("count"),
        x="count", y="make", orientation="h",
        color="count", color_continuous_scale=["#1f3a5c","#58a6ff"],
        labels={"count": "Registrations", "make": "Brand"},
        text="count",
    )
    fig3.update_traces(
        texttemplate="%{text:,}", textposition="outside",
        textfont_color="#e6edf3",
        hovertemplate="<b>%{y}</b><br>Registrations: %{x:,}<extra></extra>",
        **{k: v for k, v in HOVER_STYLE.items()},
    )
    fig3.update_layout(**DARK, height=560, coloraxis_showscale=False,
                       xaxis_title="Registrations", yaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)

# ── Tab 4: Range Evolution ────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Electric Range Evolution by Model Year</div>', unsafe_allow_html=True)
    evo = range_evolution(ev)
    if not evo.empty:
        fig4 = go.Figure()
        series = [("mean","Mean","#58a6ff","dash"),("median","Median","#3fb950","solid"),
                  ("max","Max","#f0883e","dot"),("p75","75th %ile","#bc8cff","dashdot")]
        for col, label, color, dash in series:
            if col in evo.columns:
                fig4.add_trace(go.Scatter(
                    x=evo["model_year"].astype(int), y=evo[col].round(1),
                    mode="lines+markers", name=label,
                    line=dict(color=color, width=2, dash=dash), marker=dict(size=6),
                    hovertemplate=f"<b>{label}</b><br>Year: %{{x}}<br>Range: %{{y:.1f}} mi<extra></extra>",
                ))
        fig4.update_layout(**DARK, height=420, hovermode="x unified",
                           xaxis_title="Model Year", yaxis_title="Range (miles)",
                           legend=dict(bgcolor="rgba(0,0,0,0)"),
                           **HOVER_STYLE)
        st.plotly_chart(fig4, use_container_width=True)

        top15 = (
            ev[ev.get("range_known", ev["electric_range"] > 0)]
            .groupby(["make","model"])["electric_range"].max()
            .reset_index().sort_values("electric_range", ascending=False).head(15)
        )
        top15.columns = ["Brand","Model","Max Range (miles)"]
        st.markdown('<div class="section-title">Top 15 Longest-Range Models</div>', unsafe_allow_html=True)
        st.dataframe(top15, use_container_width=True, hide_index=True)

# ── Tab 5: Minerals ───────────────────────────────────────────────────────────
with tab5:
    ev_comm = comm[comm["ev_relevant"]] if "ev_relevant" in comm.columns else comm
    st.markdown('<div class="section-title">EV-Critical Mineral Production 2018–2022</div>', unsafe_allow_html=True)

    mineral_options = sorted(ev_comm["mineral"].dropna().unique().tolist())

    # Session-state backed mineral selection so removing and re-adding works
    if "mineral_selection" not in st.session_state:
        defaults = [m for m in ["lithium","cobalt","nickel","copper","graphite"] if m in mineral_options]
        st.session_state.mineral_selection = defaults if defaults else mineral_options[:5]

    col_ms, col_reset = st.columns([5, 1])
    with col_ms:
        selected = st.multiselect(
            "Select minerals to display:",
            mineral_options,
            key="mineral_selection",
        )
    with col_reset:
        st.markdown("<br>", unsafe_allow_html=True)  # vertical align
        if st.button("Reset to All"):
            st.session_state.mineral_selection = mineral_options
            st.rerun()

    if selected:
        sub = ev_comm[ev_comm["mineral"].isin(selected)]
        trend_m = sub.groupby(["year","mineral"])["production"].sum().reset_index()
        fig5 = px.line(trend_m, x="year", y="production", color="mineral",
                       labels={"production":"Production (tonnes)","year":"Year","mineral":"Mineral"},
                       markers=True, color_discrete_sequence=px.colors.qualitative.Safe)
        fig5.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>Year: %{x}<br>Production: %{y:,.0f} t<extra></extra>",
            **{k: v for k, v in HOVER_STYLE.items()},
        )
        fig5.update_layout(**DARK, height=400, hovermode="x unified",
                           legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-title">Top 5 Producing Countries per Mineral (Latest Year)</div>', unsafe_allow_html=True)
    latest = ev_comm["year"].max()
    top_prod = (
        ev_comm[ev_comm["year"] == latest]
        .groupby(["mineral","country"])["production"].sum().reset_index()
    )
    top_prod = top_prod.sort_values(["mineral","production"], ascending=[True, False])
    top_prod = top_prod.groupby("mineral").head(5).reset_index(drop=True)
    top_prod["mineral_label"] = top_prod["mineral"].str.title()

    if not top_prod.empty:
        # Use horizontal bars per mineral with independent y-axes (matches=None)
        fig6 = px.bar(
            top_prod, x="production", y="country", color="mineral_label",
            facet_row="mineral_label",
            orientation="h",
            labels={"production":"Production (t)", "country":"Country", "mineral_label":"Mineral"},
            color_discrete_sequence=px.colors.qualitative.Safe,
            height=max(400, len(top_prod["mineral"].unique()) * 130),
        )
        fig6.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#e6edf3", showlegend=False,
            margin=dict(l=120, r=20, t=30, b=30),
        )
        # Independent x-axis scale per facet row so small minerals are visible
        fig6.update_xaxes(matches=None, showticklabels=True)
        fig6.update_yaxes(matches=None, showticklabels=True, tickfont_color="#e6edf3")
        fig6.for_each_annotation(lambda a: a.update(
            text=a.text.split("=")[-1],
            font_size=11, font_color="#e6edf3",
        ))
        fig6.update_traces(
            hovertemplate="<b>%{y}</b><br>Production: %{x:,.0f} t<extra></extra>",
            **{k: v for k, v in HOVER_STYLE.items()},
        )
        st.plotly_chart(fig6, use_container_width=True)
