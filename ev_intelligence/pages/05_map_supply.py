"""pages/05_map_supply.py"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from engine.ml_engine import build_cluster_features, run_kmeans, _haversine_km

ev    = data["ev_population"]
sta   = data["ev_stations"]
ports = data["ports"]
comm  = data["mining_commodities"]

st.markdown("""
<div class="page-header">
  <h1>Supply Chain &amp; Port Logistics</h1>
  <p>Import/export routing between recommended plant sites, raw mineral sources, and marine shipping ports.</p>
</div>
""", unsafe_allow_html=True)

# ── Get centroids ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Building supply-chain map…")
def get_centroids(_ev, _sta, _ports):
    feat = build_cluster_features(_ev, _sta, _ports, grid_resolution=1.0)
    _, cents = run_kmeans(feat, n_clusters=7)
    return cents

centroids = get_centroids(ev, sta, ports)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Map Controls")
    show_ports    = st.checkbox("Show Ports",            value=True)
    show_minerals = st.checkbox("Show Mineral Sources",  value=True)
    show_routes   = st.checkbox("Show Supply Routes",    value=True)
    top_n_ports   = st.slider("Max Ports Shown", 50, 500, 200)
    mineral_filter = st.multiselect(
        "Minerals to Highlight",
        ["lithium","cobalt","nickel","copper","graphite","rare earths","manganese"],
        default=["lithium","cobalt","nickel","copper"],
    )

# ── Mineral production locations (approximate country centroids) ───────────────
COUNTRY_COORDS = {
    "china":          (35.86, 104.19),
    "australia":      (-25.27, 133.77),
    "democratic republic of the congo": (-4.03, 21.75),
    "congo":          (-4.03, 21.75),
    "chile":          (-35.67, -71.54),
    "argentina":      (-38.41, -63.61),
    "russia":         (61.52, 105.31),
    "canada":         (56.13, -106.34),
    "united states":  (37.09, -95.71),
    "brazil":         (-14.23, -51.92),
    "south africa":   (-30.55, 22.93),
    "indonesia":      (-0.78, 113.92),
    "philippines":    (12.87, 121.77),
    "morocco":        (31.79, -7.09),
    "peru":           (-9.18, -75.01),
    "bolivia":        (-16.29, -63.58),
    "zimbabwe":       (-19.01, 29.15),
    "madagascar":     (-18.76, 46.86),
    "mozambique":     (-18.66, 35.52),
    "zambia":         (-13.13, 27.84),
    "myanmar":        (21.91, 95.95),
    "finland":        (61.92, 25.74),
    "norway":         (60.47, 8.46),
    "sweden":         (60.12, 18.64),
    "germany":        (51.16, 10.45),
    "france":         (46.22, 2.21),
    "japan":          (36.20, 138.25),
    "south korea":    (35.90, 127.76),
    "mexico":         (23.63, -102.55),
    "india":          (20.59, 78.96),
}

def get_coords(country_str):
    c = str(country_str).strip().lower()
    for key, coords in COUNTRY_COORDS.items():
        if key in c or c in key:
            return coords
    return None

ev_comm = comm[comm["ev_relevant"]] if "ev_relevant" in comm.columns else comm
if mineral_filter:
    ev_comm = ev_comm[ev_comm["mineral"].isin(mineral_filter)]

latest = ev_comm["year"].max()
min_data = (
    ev_comm[ev_comm["year"] == latest]
    .groupby(["country","mineral"])["production"].sum()
    .reset_index()
)
min_data["coords"] = min_data["country"].apply(get_coords)
min_data = min_data[min_data["coords"].notna()].copy()
min_data["lat"] = min_data["coords"].apply(lambda c: c[0])
min_data["lon"] = min_data["coords"].apply(lambda c: c[1])

# ── Port filter ───────────────────────────────────────────────────────────────
PORT_ORDER = {"V":4,"L":3,"M":2,"S":1}
ports_f = ports.copy()
if "harborsize" in ports_f.columns:
    ports_f["_rank"] = ports_f["harborsize"].map(PORT_ORDER).fillna(0)
    ports_f = ports_f.nlargest(top_n_ports, "_rank")

# ── Build figure ──────────────────────────────────────────────────────────────
fig = go.Figure()

MINERAL_COLORS = {
    "lithium":    "#a78bfa", "cobalt":    "#f472b6",
    "nickel":     "#34d399", "copper":    "#fb923c",
    "graphite":   "#94a3b8", "rare earths":"#fbbf24",
    "manganese":  "#38bdf8",
}

# Ports
if show_ports and not ports_f.empty:
    fig.add_trace(go.Scattergeo(
        lat=ports_f["latitude"], lon=ports_f["longitude"],
        mode="markers",
        marker=dict(symbol="diamond", size=5, color="#38bdf8", opacity=0.6,
                    line=dict(width=0)),
        name="Shipping Port",
        hovertext=ports_f.apply(
            lambda r: f"<b>{r.get('port_name','Port')}</b><br>{r.get('country','')}<br>Size: {r.get('harborsize','?')}",
            axis=1
        ),
        hoverinfo="text",
    ))

# Mineral sources
if show_minerals and not min_data.empty:
    for mineral in min_data["mineral"].unique():
        sub = min_data[min_data["mineral"] == mineral]
        max_prod = sub["production"].max() or 1
        fig.add_trace(go.Scattergeo(
            lat=sub["lat"], lon=sub["lon"],
            mode="markers",
            marker=dict(
                symbol="hexagram",
                size=(sub["production"] / max_prod * 20 + 6).clip(6, 30),
                color=MINERAL_COLORS.get(mineral, "#e2e8f0"),
                opacity=0.9,
                sizemode="area",
                line=dict(width=0.5, color="#0d1117"),
            ),
            name=f"{mineral.title()} (mine)",
            hovertext=sub.apply(
                lambda r: f"<b>{r['country']}</b><br>{r['mineral'].title()}: {r['production']:,.0f} t",
                axis=1
            ),
            hoverinfo="text",
        ))

# Plant centroids
for _, row in centroids.iterrows():
    fig.add_trace(go.Scattergeo(
        lat=[row["latitude"]], lon=[row["longitude"]],
        mode="markers+text",
        marker=dict(symbol="star", size=22, color="#ffd700",
                    line=dict(width=2, color="#0d1117")),
        text=[f"P{int(row['rank'])}"],
        textposition="top center",
        textfont=dict(color="#e6edf3", size=10, family="Arial Black"),
        name=f"Plant #{int(row['rank'])}",
        hovertext=(
            f"<b>Plant Site #{int(row['rank'])}</b><br>"
            f"GPS: {row['latitude']:.3f}°, {row['longitude']:.3f}°<br>"
            f"Viability: {row['viability_score']:.4f}"
        ),
        hoverinfo="text",
    ))

# Supply routes: plant → 3 nearest ports
if show_routes and not ports_f.empty:
    for _, plant in centroids.iterrows():
        dists = _haversine_km(
            plant["latitude"], plant["longitude"],
            ports_f["latitude"].values, ports_f["longitude"].values,
        )
        for i in np.argsort(dists)[:2]:
            p = ports_f.iloc[i]
            opacity = max(0.15, 0.7 - dists[i] / 5000)
            fig.add_trace(go.Scattergeo(
                lat=[plant["latitude"], p["latitude"]],
                lon=[plant["longitude"], p["longitude"]],
                mode="lines",
                line=dict(width=1.2, color=f"rgba(251,191,36,{opacity:.2f})"),
                showlegend=False, hoverinfo="skip",
            ))

fig.update_geos(
    showland=True, landcolor="#1c2128",
    showocean=True, oceancolor="#0d1117",
    showcountries=True, countrycolor="#30363d",
    showcoastlines=True, coastlinecolor="#30363d",
    showlakes=False,
    projection_type="natural earth", bgcolor="#0d1117",
)
fig.update_layout(
    paper_bgcolor="#0d1117", font_color="#e6edf3",
    height=620, geo_bgcolor="#0d1117",
    margin=dict(t=0,b=0,l=0,r=0),
    legend=dict(
        font_size=9, bgcolor="rgba(22,27,34,0.92)",
        bordercolor="#30363d", borderwidth=1,
        x=0.01, y=0.99, xanchor="left", yanchor="top",
        font_color="#e6edf3",
    ),
)
st.plotly_chart(fig, use_container_width=True)

# ── Distance matrix ───────────────────────────────────────────────────────────
if not ports_f.empty:
    st.markdown('<div class="section-title">Plant → Nearest Port Distance Matrix</div>', unsafe_allow_html=True)
    rows = []
    for _, plant in centroids.head(7).iterrows():
        dists = _haversine_km(
            plant["latitude"], plant["longitude"],
            ports_f["latitude"].values, ports_f["longitude"].values,
        )
        for rank_p, i in enumerate(np.argsort(dists)[:3], 1):
            p = ports_f.iloc[i]
            rows.append({
                "Plant Site": f"Site #{int(plant['rank'])} ({plant['latitude']:.2f}°, {plant['longitude']:.2f}°)",
                "Viability":  f"{plant['viability_score']:.4f}",
                "Port":       str(p.get("port_name","—")).title(),
                "Country":    str(p.get("country","—")),
                "Distance (km)": int(round(dists[i])),
                "Harbour Size":  str(p.get("harborsize","—")),
                "Nearest Rank":  rank_p,
            })
    dist_df = pd.DataFrame(rows)
    st.dataframe(dist_df, use_container_width=True, hide_index=True)
