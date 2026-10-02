"""pages/02_map_demand.py"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

ev  = data["ev_population"]
sta = data["ev_stations"]

DARK_GEO = dict(
    paper_bgcolor="#0d1117", font_color="#e6edf3",
    margin=dict(t=0, b=0, l=0, r=0),
)

st.markdown("""
<div class="page-header">
  <h1>Demand &amp; Charging Infrastructure</h1>
  <p>Global EV registration density overlaid with charging station coverage.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Map Controls")
    ev_type_opts = [t for t in ["BEV","PHEV","UNKNOWN"] if t in ev["ev_type"].unique()]
    ev_type_filter = st.multiselect("EV Type", ev_type_opts, default=["BEV","PHEV"])
    yr_min, yr_max = int(ev["model_year"].min()), int(ev["model_year"].max())
    year_range = st.slider("Model Year", yr_min, yr_max, (2015, yr_max))
    show_stations = st.checkbox("Show Charging Stations", value=True)
    density_res   = st.select_slider("Grid Resolution (°)", [0.25, 0.5, 1.0, 2.0], value=0.5)

# ── Filter ────────────────────────────────────────────────────────────────────
ev_f = ev[
    ev["ev_type"].isin(ev_type_filter if ev_type_filter else ev_type_opts) &
    ev["model_year"].between(year_range[0], year_range[1])
].copy()

# ── Grid aggregation ──────────────────────────────────────────────────────────
res = float(density_res)
ev_f["lat_bin"] = (ev_f["latitude"]  / res).round() * res
ev_f["lon_bin"] = (ev_f["longitude"] / res).round() * res
grid = (
    ev_f.groupby(["lat_bin","lon_bin"])
        .agg(count=("latitude","size"), top_make=("make", lambda x: x.value_counts().index[0]))
        .reset_index()
        .rename(columns={"lat_bin":"lat","lon_bin":"lon"})
)

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
c1.metric("Filtered EVs",    f"{len(ev_f):,}")
c2.metric("Grid Cells",      f"{len(grid):,}")
c3.metric("Charging Stations", f"{len(sta):,}")
c4.metric("Countries",       f"{sta['country'].nunique()}")

tab1, tab2 = st.tabs(["Global Bubble Map", "Density Heatmap"])

# ── Tab 1: Bubble Map ─────────────────────────────────────────────────────────
with tab1:
    fig = go.Figure()

    # Registration bubbles
    fig.add_trace(go.Scattergeo(
        lat=grid["lat"], lon=grid["lon"],
        mode="markers",
        marker=dict(
            size=np.sqrt(grid["count"].clip(1)) * 3.5,
            color=grid["count"],
            colorscale=[[0,"#1f3a5c"],[0.4,"#1f6feb"],[0.7,"#58a6ff"],[1.0,"#ffffff"]],
            showscale=True,
            colorbar=dict(
                title=dict(text="Registrations", font_color="#e6edf3"),
                tickfont=dict(color="#e6edf3"),
                x=0.93, len=0.55,
            ),
            sizemode="area", opacity=0.8,
            line=dict(width=0),
        ),
        text=grid.apply(lambda r: f"<b>{r['top_make']}</b><br>{int(r['count']):,} EVs", axis=1),
        hoverinfo="text+name",
        name="EV Registrations",
    ))

    # Charging stations
    if show_stations and not sta.empty:
        sample = sta.sample(min(6000, len(sta)), random_state=42)
        fig.add_trace(go.Scattergeo(
            lat=sample["latitude"], lon=sample["longitude"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=5, color="#3fb950",
                        opacity=0.6, line=dict(width=0)),
            name="Charging Station",
            hovertext=sample.apply(
                lambda r: f"{r.get('title', r.get('operator','Station'))}<br>{r['country']}", axis=1),
            hoverinfo="text",
        ))

    fig.update_geos(
        showland=True,        landcolor="#1c2128",
        showocean=True,       oceancolor="#0d1117",
        showcountries=True,   countrycolor="#30363d",
        showcoastlines=True,  coastlinecolor="#30363d",
        showlakes=True,       lakecolor="#0d1117",
        projection_type="natural earth",
        bgcolor="#0d1117",
    )
    fig.update_layout(
        **DARK_GEO, height=580,
        geo_bgcolor="#0d1117",
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.08, xanchor="left", x=0,
            bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363d", borderwidth=1,
            font_color="#e6edf3",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Density Heatmap ────────────────────────────────────────────────────
with tab2:
    sample_df = ev_f.sample(min(40000, len(ev_f)), random_state=42)
    # Use go.Densitymapbox directly — px.density_mapbox was removed in newer Plotly
    fig2 = go.Figure(go.Densitymapbox(
        lat=sample_df["latitude"],
        lon=sample_df["longitude"],
        radius=10,
        colorscale="Plasma",
        showscale=True,
    ))
    fig2.update_layout(
        mapbox=dict(style="open-street-map", zoom=3,
                    center=dict(lat=47.5, lon=-122.0)),
        paper_bgcolor="#0d1117", font_color="#e6edf3",
        height=560, margin=dict(t=0, b=0, l=0, r=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Station density by country bar
    st.markdown('<div class="section-title">Charging Station Distribution by Country</div>', unsafe_allow_html=True)
    country_sta = sta["country"].value_counts().head(20).reset_index()
    country_sta.columns = ["Country","Stations"]
    fig3 = px.bar(country_sta, x="Country", y="Stations",
                  color="Stations", color_continuous_scale=["#1a3a2a","#3fb950"],
                  text="Stations")
    fig3.update_traces(texttemplate="%{text:,}", textposition="outside", textfont_color="#e6edf3")
    fig3.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                       font_color="#e6edf3", height=360,
                       coloraxis_showscale=False,
                       xaxis=dict(tickangle=-30, gridcolor="#21262d"),
                       yaxis=dict(gridcolor="#21262d"),
                       margin=dict(t=10,b=10))
    st.plotly_chart(fig3, use_container_width=True)
