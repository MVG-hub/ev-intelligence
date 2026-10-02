"""pages/04_clustering.py"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from engine.ml_engine import build_cluster_features, run_kmeans

ev    = data["ev_population"]
sta   = data["ev_stations"]
ports = data["ports"]

DARK_GEO = dict(paper_bgcolor="#0d1117", font_color="#e6edf3", margin=dict(t=0,b=0,l=0,r=0))

st.markdown("""
<div class="page-header">
  <h1>Manufacturing Plant Site Clustering</h1>
  <p>K-Means clustering across demand density, charging infrastructure &amp; port proximity to surface GPS-precise plant locations.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Clustering Controls")
    n_clusters = st.slider("Number of Clusters (K)", 3, 12, 7)
    grid_res   = st.select_slider("Grid Resolution (°)", [0.5, 1.0, 2.0], value=1.0)
    run_btn    = st.button("Re-run Clustering", type="primary")

@st.cache_data(show_spinner="Running K-Means clustering…")
def run_clustering(_ev, _sta, _ports, k, res):
    feat = build_cluster_features(_ev, _sta, _ports, grid_resolution=res)
    labeled, cents = run_kmeans(feat, n_clusters=k)
    return labeled, cents

if run_btn:
    st.cache_data.clear()

labeled_df, centroids = run_clustering(ev, sta, ports, n_clusters, grid_res)

# ── KPIs ──────────────────────────────────────────────────────────────────────
top = centroids.iloc[0]
c1,c2,c3,c4 = st.columns(4)
c1.metric("Top Site GPS",      f"{top['latitude']:.2f}°N, {top['longitude']:.2f}°")
c2.metric("Viability #1",      f"{top['viability_score']:.4f}")
c3.metric("Grid Cells",        f"{len(labeled_df):,}")
c4.metric("Clusters (K)",      str(n_clusters))

tab1, tab2, tab3 = st.tabs(["Cluster Map", "Viability Analysis", "Ranked Sites"])

# ── Tab 1: Cluster Map ────────────────────────────────────────────────────────
with tab1:
    fig = go.Figure()
    colors = px.colors.qualitative.Vivid

    for cid in sorted(labeled_df["cluster"].unique()):
        sub = labeled_df[labeled_df["cluster"] == cid]
        fig.add_trace(go.Scattergeo(
            lat=sub["lat_bin"], lon=sub["lon_bin"],
            mode="markers",
            marker=dict(
                size=np.sqrt(sub["registration_count"].clip(1)) * 2.5,
                color=colors[cid % len(colors)],
                opacity=0.4, sizemode="area",
                line=dict(width=0),
            ),
            name=f"Cluster {cid}",
            hoverinfo="skip",
        ))

    for _, row in centroids.iterrows():
        size = 20 + row["viability_score"] * 12
        fig.add_trace(go.Scattergeo(
            lat=[row["latitude"]], lon=[row["longitude"]],
            mode="markers+text",
            marker=dict(
                symbol="star", size=size, color="#ffd700",
                line=dict(width=1.5, color="#0d1117"),
            ),
            text=[f"#{int(row['rank'])}"],
            textposition="top center",
            textfont=dict(color="#e6edf3", size=11, family="Arial Black"),
            name=f"Site #{int(row['rank'])} (score {row['viability_score']:.3f})",
            hovertext=(
                f"<b>Recommended Site #{int(row['rank'])}</b><br>"
                f"GPS: {row['latitude']:.3f}°, {row['longitude']:.3f}°<br>"
                f"Viability: {row['viability_score']:.4f}<br>"
                f"Demand: {row['avg_demand']:,.0f} registrations/cell<br>"
                f"Station Density: {row['avg_station_density']:.0f} stations/200km<br>"
                f"Nearest Port: {row['avg_port_km']:.0f} km"
            ),
            hoverinfo="text",
        ))

    fig.update_geos(
        showland=True, landcolor="#1c2128",
        showocean=True, oceancolor="#0d1117",
        showcountries=True, countrycolor="#30363d",
        showcoastlines=True, coastlinecolor="#30363d",
        projection_type="natural earth", bgcolor="#0d1117",
    )
    fig.update_layout(
        **DARK_GEO, height=580, geo_bgcolor="#0d1117",
        legend=dict(font_size=9, orientation="h",
                    yanchor="bottom", y=-0.15,
                    bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363d", borderwidth=1,
                    font_color="#e6edf3"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Viability Analysis ─────────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Viability Score Breakdown</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        x_labels = centroids["rank"].astype(str).tolist()
        fig2.add_trace(go.Bar(name="Demand (norm)", x=x_labels,
                               y=(centroids["avg_demand"] / centroids["avg_demand"].max()).round(3),
                               marker_color="#58a6ff"))
        fig2.add_trace(go.Bar(name="Station Density (norm)", x=x_labels,
                               y=(centroids["avg_station_density"] / max(float(centroids["avg_station_density"].max()), 1)).round(3),
                               marker_color="#3fb950"))
        max_port = centroids["avg_port_km"].max() or 1
        fig2.add_trace(go.Bar(name="Port Proximity (inv)", x=x_labels,
                               y=(1 - centroids["avg_port_km"] / max_port).round(3),
                               marker_color="#f0883e"))
        fig2.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#e6edf3",
            barmode="group", xaxis_title="Site Rank", yaxis_title="Normalised Score",
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
            height=340, hovermode="x unified",
            legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Top-3 Sites Radar</div>', unsafe_allow_html=True)
        top3 = centroids.head(3)
        radar_cats  = ["Demand","Station Density","Port Proximity","Viability"]
        radar_cols  = ["avg_demand","avg_station_density","avg_port_km","viability_score"]
        radar_colors = ["#58a6ff","#3fb950","#f0883e"]

        fig3 = go.Figure()
        for i, (_, row) in enumerate(top3.iterrows()):
            norms = [
                row["avg_demand"]          / (centroids["avg_demand"].max() or 1),
                row["avg_station_density"] / (centroids["avg_station_density"].max() or 1),
                1 - row["avg_port_km"]     / (centroids["avg_port_km"].max() or 1),
                row["viability_score"],
            ]
            norms.append(norms[0])
            cats = radar_cats + [radar_cats[0]]
            fig3.add_trace(go.Scatterpolar(
                r=norms, theta=cats, fill="toself",
                name=f"Site #{int(row['rank'])}",
                line_color=radar_colors[i], opacity=0.75,
            ))
        fig3.update_layout(
            paper_bgcolor="#0d1117", font_color="#e6edf3",
            polar=dict(
                bgcolor="#161b22",
                radialaxis=dict(visible=True, range=[0,1],
                                gridcolor="#30363d", tickfont_color="#8b949e"),
                angularaxis=dict(gridcolor="#30363d", tickfont_color="#e6edf3"),
            ),
            height=340,
            legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

# ── Tab 3: Ranked sites table ─────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">All Recommended Plant Sites — Ranked by Viability</div>', unsafe_allow_html=True)
    disp = centroids[[
        "rank","latitude","longitude","viability_score",
        "total_registrations","avg_station_density","avg_port_km","cells_in_cluster",
    ]].copy()
    disp.columns = ["Rank","Latitude","Longitude","Viability Score",
                     "Total EV Demand","Station Density","Nearest Port (km)","Grid Cells"]
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.download_button("Download Sites CSV",
                        disp.to_csv(index=False).encode(),
                        "plant_sites.csv", "text/csv")

    st.markdown("""
    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                padding:1rem 1.2rem;margin-top:1rem;font-size:0.82rem;color:#8b949e;">
      <b style="color:#e6edf3;">Viability Score Formula:</b><br>
      <code style="color:#58a6ff;">0.45 × (demand/max_demand) + 0.30 × (stations/max_stations) + 0.25 × (1 − port_km/max_port_km)</code><br>
      Scores normalised to [0, 1]. Higher = better candidate for assembly plant.
    </div>
    """, unsafe_allow_html=True)
