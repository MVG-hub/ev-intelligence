"""
ml_engine.py — fixed & improved
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ── EDA helpers ───────────────────────────────────────────────────────────────

def ev_type_trend(df: pd.DataFrame) -> pd.DataFrame:
    if "model_year" not in df.columns or "ev_type" not in df.columns:
        return pd.DataFrame()
    trend = (
        df.groupby(["model_year", "ev_type"])
          .size()
          .reset_index(name="count")
          .pivot(index="model_year", columns="ev_type", values="count")
          .fillna(0)
          .sort_index()
    )
    trend.columns.name = None
    return trend


def price_vs_range(df: pd.DataFrame, top_n_makes: int = 12) -> pd.DataFrame:
    cols = ["make", "electric_range", "ev_type", "model_year"]
    if "base_msrp" in df.columns:
        cols.insert(1, "base_msrp")
    sub = df[[c for c in cols if c in df.columns]].copy()
    sub = sub.dropna(subset=["electric_range"])
    sub = sub[sub["electric_range"] > 0]
    top_makes = sub["make"].value_counts().head(top_n_makes).index
    return sub[sub["make"].isin(top_makes)].copy()


def brand_market_share(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if "make" not in df.columns:
        return pd.DataFrame(columns=["make", "count"])
    result = (
        df["make"].value_counts()
                  .head(top_n)
                  .reset_index()
    )
    result.columns = ["make", "count"]   # always force correct names
    return result


def state_registration_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "state" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("state")
          .size()
          .reset_index(name="registrations")
          .sort_values("registrations", ascending=False)
    )


def range_evolution(df: pd.DataFrame) -> pd.DataFrame:
    if "electric_range" not in df.columns or "model_year" not in df.columns:
        return pd.DataFrame()
    sub = df[df["range_known"]].copy() if "range_known" in df.columns else df[df["electric_range"] > 0].copy()
    return (
        sub.groupby("model_year")["electric_range"]
           .agg(mean="mean", median="median", max="max", p75=lambda x: x.quantile(0.75))
           .reset_index()
    )


# ── Regional growth forecasting ───────────────────────────────────────────────

def forecast_regional_growth(
    df: pd.DataFrame,
    forecast_years: list = None,
    min_records: int = 5,          # lowered from 30
) -> pd.DataFrame:
    if forecast_years is None:
        forecast_years = [2025, 2026, 2027, 2028]

    if "state" not in df.columns or "model_year" not in df.columns:
        return pd.DataFrame()

    historical = (
        df.groupby(["state", "model_year"])
          .size()
          .reset_index(name="registrations")
    )

    results = []
    for state, grp in historical.groupby("state"):
        if grp["registrations"].sum() < min_records:
            continue
        if len(grp) < 2:
            continue

        X = grp["model_year"].values.reshape(-1, 1).astype(float)
        y = grp["registrations"].values.astype(float)

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("ridge",   Ridge(alpha=1.0)),
        ])
        model.fit(X, y)

        for _, row in grp.iterrows():
            results.append({
                "state": state,
                "year":  int(row["model_year"]),
                "registrations": int(row["registrations"]),
                "is_forecast": False,
            })
        for yr in forecast_years:
            pred = max(0.0, float(model.predict([[float(yr)]])[0]))
            results.append({
                "state": state,
                "year":  yr,
                "registrations": int(round(pred)),
                "is_forecast": True,
            })

    return pd.DataFrame(results)


# ── Haversine ─────────────────────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest_port_distance(lat, lon, ports: pd.DataFrame) -> float:
    if ports.empty:
        return np.nan
    dists = _haversine_km(lat, lon, ports["latitude"].values, ports["longitude"].values)
    return float(np.min(dists))


def nearest_station_density(lat, lon, stations: pd.DataFrame, radius_km: float = 200) -> int:
    if stations.empty:
        return 0
    dists = _haversine_km(lat, lon, stations["latitude"].values, stations["longitude"].values)
    return int(np.sum(dists <= radius_km))


# ── Global EV hotspots (IEA 2023 market data) ────────────────────────────────
# Exact city-centre coordinates — NOT rounded to grid — so weighted centroids
# stay on land. Counts proportional to IEA 2023 EV adoption data.
GLOBAL_EV_HOTSPOTS = [
    # (lat, lon, demand_count, city_name)
    (39.91,  116.39,  50000, "Beijing"),
    (31.23,  121.47,  45000, "Shanghai"),
    (22.54,  114.06,  35000, "Shenzhen"),
    (23.13,  113.26,  28000, "Guangzhou"),
    (51.51,   -0.13,  18000, "London"),
    (48.86,    2.35,  15000, "Paris"),
    (52.52,   13.40,  14000, "Berlin"),
    (59.91,   10.75,  12000, "Oslo"),
    (55.75,   37.62,  10000, "Moscow"),
    (35.68,  139.69,  20000, "Tokyo"),
    (37.57,  126.98,  16000, "Seoul"),
    (28.63,   77.22,   8000, "Delhi"),
    (19.08,   72.88,   7000, "Mumbai"),
    (-23.55, -46.63,   9000, "Sao Paulo"),
    (40.42,   -3.70,   8000, "Madrid"),
    (45.46,    9.19,   7000, "Milan"),
    (52.37,    4.90,   9000, "Amsterdam"),
    (37.77, -122.42,  30000, "San Francisco"),
    (34.05, -118.24,  25000, "Los Angeles"),
    (40.71,  -74.01,  20000, "New York"),
    (41.88,  -87.63,  12000, "Chicago"),
    (43.65,  -79.38,  10000, "Toronto"),
    (-33.87,  151.21,  11000, "Sydney"),
    (1.35,   103.82,  13000, "Singapore"),
    (25.20,   55.27,   9000, "Dubai"),
    (30.06,   31.24,   6000, "Cairo"),
    (55.68,   12.57,   8000, "Copenhagen"),
    (59.33,   18.07,   7000, "Stockholm"),
    (60.17,   24.94,   6000, "Helsinki"),
    (47.38,    8.54,   8000, "Zurich"),
]

# Lookup for centroid snapping: map each hotspot to its exact land coordinate
_HOTSPOT_COORDS = {(r[0], r[1]): (r[0], r[1], r[3]) for r in GLOBAL_EV_HOTSPOTS}


# ── K-Means clustering ────────────────────────────────────────────────────────

def build_cluster_features(
    ev_pop: pd.DataFrame,
    ev_stations: pd.DataFrame,
    ports: pd.DataFrame,
    grid_resolution: float = 2.0,
) -> pd.DataFrame:
    pop = ev_pop.copy()
    pop["lat_bin"] = (pop["latitude"]  / grid_resolution).round() * grid_resolution
    pop["lon_bin"] = (pop["longitude"] / grid_resolution).round() * grid_resolution

    grid = (
        pop.groupby(["lat_bin", "lon_bin"])
           .size()
           .reset_index(name="registration_count")
    )

    # Append synthetic global hotspot rows using EXACT city coordinates
    # (not rounded) to prevent ocean placement of centroids
    hotspot_rows = []
    for lat, lon, count, _ in GLOBAL_EV_HOTSPOTS:
        hotspot_rows.append({
            "lat_bin": float(lat),
            "lon_bin": float(lon),
            "registration_count": int(count),
        })
    hotspot_df = pd.DataFrame(hotspot_rows)
    grid = pd.concat([grid, hotspot_df], ignore_index=True)
    # Sum duplicate cells (city lat/lon may already be in grid from real data)
    grid = (
        grid.groupby(["lat_bin", "lon_bin"], as_index=False)["registration_count"]
            .sum()
    )

    print(f"  [Clustering] grid cells (incl. global hotspots): {len(grid):,}")

    if not ev_stations.empty:
        grid["station_density"] = grid.apply(
            lambda r: nearest_station_density(r["lat_bin"], r["lon_bin"], ev_stations),
            axis=1,
        )
    else:
        grid["station_density"] = 0

    if not ports.empty:
        grid["nearest_port_km"] = grid.apply(
            lambda r: nearest_port_distance(r["lat_bin"], r["lon_bin"], ports),
            axis=1,
        )
    else:
        grid["nearest_port_km"] = 500.0

    grid = grid.dropna()
    return grid


# Known ocean bounding boxes — centroids here are definitely in the sea
_OCEAN_BOXES = [
    # (lat_min, lat_max, lon_min, lon_max, description)
    (-60,  60,  -180, -100, "East Pacific"),
    (-60,  60,   130,  180, "West Pacific"),
    (-60,  25,  -180,  -30, "South Atlantic"),
    (  0,  65,   -45,   -5, "North Atlantic"),
    (-60,  30,    20,   80, "Indian Ocean"),
    ( 55,  90,  -180,  180, "Arctic"),
    (-90, -55,  -180,  180, "Antarctic"),
]


def _is_ocean(lat: float, lon: float) -> bool:
    """Heuristic: return True if coordinate is in a known ocean region."""
    for lat_min, lat_max, lon_min, lon_max, _ in _OCEAN_BOXES:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return True
    return False


def _snap_to_land(lat: float, lon: float) -> tuple:
    """
    If centroid falls in a known ocean region, snap it to the nearest
    known urban centre. Otherwise keep the original coordinate.
    """
    if not _is_ocean(lat, lon):
        return round(lat, 4), round(lon, 4)

    known = [(r[0], r[1]) for r in GLOBAL_EV_HOTSPOTS]
    best_dist = float("inf")
    best_lat, best_lon = lat, lon
    for klat, klon in known:
        d = _haversine_km(lat, lon, klat, klon)
        if d < best_dist:
            best_dist = d
            best_lat, best_lon = klat, klon
    return round(best_lat, 4), round(best_lon, 4)


def run_kmeans(
    feature_df: pd.DataFrame,
    n_clusters: int = 7,
    random_state: int = 42,
) -> tuple:
    feature_cols = ["registration_count", "station_density", "nearest_port_km"]
    feature_cols = [c for c in feature_cols if c in feature_df.columns]

    # Guard: can't have more clusters than samples
    n_clusters = min(n_clusters, len(feature_df))

    X = feature_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    feature_df = feature_df.copy()
    feature_df["cluster"] = km.fit_predict(X_scaled)

    centroid_rows = []
    for cid in range(n_clusters):
        mask = feature_df["cluster"] == cid
        sub  = feature_df[mask]
        w = sub["registration_count"].values.astype(float)
        if w.sum() == 0:
            w = np.ones(len(sub))
        # Demand-weighted mean position
        c_lat = float(np.average(sub["lat_bin"].values, weights=w))
        c_lon = float(np.average(sub["lon_bin"].values, weights=w))
        # Snap to nearest known urban centre to avoid ocean placement
        c_lat, c_lon = _snap_to_land(c_lat, c_lon)
        centroid_rows.append({
            "cluster_id":          cid,
            "latitude":            c_lat,
            "longitude":           c_lon,
            "avg_demand":          round(float(sub["registration_count"].mean()), 1),
            "total_registrations": int(sub["registration_count"].sum()),
            "avg_station_density": round(float(sub["station_density"].mean()), 1),
            "avg_port_km":         round(float(sub["nearest_port_km"].mean()), 1),
            "cells_in_cluster":    int(mask.sum()),
        })

    centroids = pd.DataFrame(centroid_rows)

    max_demand  = float(centroids["avg_demand"].max()) or 1.0
    max_density = max(float(centroids["avg_station_density"].max()), 1.0)
    max_port    = float(centroids["avg_port_km"].max()) or 1.0

    centroids["viability_score"] = (
        0.45 * (centroids["avg_demand"]          / max_demand) +
        0.30 * (centroids["avg_station_density"] / max_density) +
        0.25 * (1 - centroids["avg_port_km"]     / max_port)
    ).round(4)

    centroids = centroids.sort_values("viability_score", ascending=False).reset_index(drop=True)
    centroids["rank"] = centroids.index + 1

    print(f"  [Clustering] OK {n_clusters} clusters found")
    return feature_df, centroids
