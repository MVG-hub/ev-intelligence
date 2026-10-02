"""
data_loader.py
--------------
Loads, validates, and deeply cleans all six source datasets.
Returns clean DataFrames ready for ML and visualisation.
"""

import re
import os
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(__file__), "..", "..")
_P = lambda f: os.path.normpath(os.path.join(_BASE, f))

EV_POP_PATH         = _P("Electric_Vehicle_Population_Data.csv")
EV_STATIONS_PATH    = _P("ev_stations_2025.csv")
PORT_INDEX_PATH     = _P("World_Port_Index.csv")
MINING_CO_PATH      = _P("116_world_mining_companies_clean.csv")
MINING_COMM_PATH    = _P("world_mining_commodities_clean.csv")
COMMODITY_INFO_PATH = _P("commodity_info.xlsx")

# EV battery minerals that matter for plant site selection
EV_MINERALS = {
    "lithium", "cobalt", "nickel", "copper", "manganese",
    "graphite", "rare earths", "aluminium", "aluminum",
    "zinc", "platinum", "palladium", "titanium", "molybdenum",
}


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════

def _drop_full_duplicates(df: pd.DataFrame, label: str) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        print(f"  [{label}] dropped {removed} duplicate rows")
    return df


def _parse_point(point_str):
    """Parse 'POINT (lon lat)' → (lat, lon) floats."""
    if pd.isna(point_str):
        return np.nan, np.nan
    m = re.search(r"POINT\s*\(([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)\)", str(point_str))
    if m:
        lon, lat = float(m.group(1)), float(m.group(2))
        return lat, lon
    return np.nan, np.nan


def _clip_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows whose lat/lon are outside physical bounds."""
    if "latitude" in df.columns and "longitude" in df.columns:
        mask = (
            df["latitude"].between(-90, 90) &
            df["longitude"].between(-180, 180)
        )
        removed = (~mask).sum()
        if removed:
            print(f"    clipped {removed} out-of-bounds coordinates")
        df = df[mask]
    return df


def _standardise_country(series: pd.Series) -> pd.Series:
    """Upper-strip country codes; expand obvious abbreviations."""
    mapping = {
        "USA": "United States", "US": "United States",
        "UK": "United Kingdom", "GB": "United Kingdom",
        "UAE": "United Arab Emirates",
        "KR": "South Korea", "CN": "China",
    }
    return series.astype(str).str.strip().replace(mapping)


# ══════════════════════════════════════════════════════════════════════════════
# 1 · EV Population (registrations)
# ══════════════════════════════════════════════════════════════════════════════

def load_ev_population() -> pd.DataFrame:
    print("[EV Population] loading …")
    df = pd.read_csv(EV_POP_PATH, low_memory=False)

    # ── normalise column names ──
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True)
                  .str.strip("_")
    )

    df = _drop_full_duplicates(df, "EV Population")

    # ── parse coordinates from WKT POINT string ──
    if "vehicle_location" in df.columns:
        df[["latitude", "longitude"]] = pd.DataFrame(
            df["vehicle_location"].apply(_parse_point).tolist(),
            index=df.index,
        )
    else:
        df["latitude"] = np.nan
        df["longitude"] = np.nan

    # ── numeric coercion ──
    for col in ["electric_range", "base_msrp", "model_year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── range: 0 often means "not researched" — keep but flag ──
    df["range_known"] = df["electric_range"].gt(0)

    # ── MSRP: $0 → NaN (missing / not disclosed) ──
    if "base_msrp" in df.columns:
        df.loc[df["base_msrp"] == 0, "base_msrp"] = np.nan

    # ── model year bounds: keep only 2000-2026 ──
    if "model_year" in df.columns:
        before = len(df)
        df = df[df["model_year"].between(2000, 2026, inclusive="both")]
        print(f"  [EV Population] removed {before - len(df)} rows with invalid model_year")

    # ── standardise EV type labels ──
    if "electric_vehicle_type" in df.columns:
        df["ev_type"] = (
            df["electric_vehicle_type"]
            .str.upper()
            .str.extract(r"(BEV|PHEV)", expand=False)
            .fillna("UNKNOWN")
        )

    # ── trim string columns ──
    # NOTE: state must stay UPPERCASE (e.g. "WA", "CA") for choropleth locationmode="USA-states"
    for col in ["make", "model", "city", "county"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.strip().str.upper()

    # ── drop rows with no usable location ──
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    print(f"  [EV Population] dropped {before - len(df)} rows missing coordinates")

    df = _clip_lat_lon(df)
    df = df.reset_index(drop=True)
    print(f"  [EV Population] OK {len(df):,} clean records")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2 · EV Charging Stations
# ══════════════════════════════════════════════════════════════════════════════

def load_ev_stations() -> pd.DataFrame:
    print("[EV Stations] loading …")
    df = pd.read_csv(EV_STATIONS_PATH, low_memory=False)

    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True)
                  .str.strip("_")
    )

    df = _drop_full_duplicates(df, "EV Stations")

    # ── coerce lat/lon ──
    for col in ["lat", "lon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.rename(columns={"lat": "latitude", "lon": "longitude"})

    # ── drop rows missing coordinates ──
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    print(f"  [EV Stations] dropped {before - len(df)} rows missing coordinates")

    df = _clip_lat_lon(df)

    # ── connector count ──
    if "num_connectors" in df.columns:
        df["num_connectors"] = pd.to_numeric(df["num_connectors"], errors="coerce").fillna(1).astype(int)
        df = df[df["num_connectors"] > 0]

    # ── keep only Operational stations for demand analysis ──
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.strip()
        operational = df[df["status"].str.lower() == "operational"]
        print(f"  [EV Stations] {len(operational):,} operational / {len(df):,} total")
        df = operational

    # ── country code upper ──
    if "country" in df.columns:
        df["country"] = df["country"].astype(str).str.strip().str.upper()

    # ── parse date ──
    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], utc=True, errors="coerce")

    df = df.reset_index(drop=True)
    print(f"  [EV Stations] OK {len(df):,} clean records")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3 · World Port Index
# ══════════════════════════════════════════════════════════════════════════════

def load_ports() -> pd.DataFrame:
    print("[World Port Index] loading …")
    df = pd.read_csv(PORT_INDEX_PATH, low_memory=False)

    # strip BOM from first column name
    df.columns = (
        df.columns.str.strip()
                  .str.lstrip("\ufeff")
                  .str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True)
                  .str.strip("_")
    )

    df = _drop_full_duplicates(df, "Ports")

    # ── lat/lon: prefer named columns, fall back to x/y ──
    for preferred, fallback in [("latitude", "y"), ("longitude", "x")]:
        if preferred not in df.columns and fallback in df.columns:
            df = df.rename(columns={fallback: preferred})

    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    print(f"  [Ports] dropped {before - len(df)} rows missing coordinates")

    df = _clip_lat_lon(df)

    # ── harbour size: keep V (Very Large), L (Large), M (Medium), S (Small) ──
    if "harborsize" in df.columns:
        df["harborsize"] = df["harborsize"].astype(str).str.strip().str.upper()
        df = df[df["harborsize"].isin(["V", "L", "M", "S"])]

    # ── standardise country ──
    if "country" in df.columns:
        df["country"] = _standardise_country(df["country"])

    # ── port name ──
    if "port_name" in df.columns:
        df["port_name"] = df["port_name"].astype(str).str.strip().str.title()

    df = df.reset_index(drop=True)
    print(f"  [Ports] OK {len(df):,} clean records")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4 · Mining Companies
# ══════════════════════════════════════════════════════════════════════════════

def load_mining_companies() -> pd.DataFrame:
    print("[Mining Companies] loading …")
    df = pd.read_csv(MINING_CO_PATH, low_memory=False)

    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True)
                  .str.strip("_")
    )

    df = _drop_full_duplicates(df, "Mining Companies")

    # ── commodity: explode multi-value cells ──
    if "commodity" in df.columns:
        df["commodity"] = df["commodity"].astype(str).str.strip()
        df = df.assign(commodity=df["commodity"].str.split(r",\s*")).explode("commodity")
        df["commodity"] = df["commodity"].str.strip().str.lower()

    # ── flag EV-relevant minerals ──
    if "commodity" in df.columns:
        df["ev_relevant"] = df["commodity"].isin(EV_MINERALS)

    # ── location cleanup ──
    if "location" in df.columns:
        df["location"] = df["location"].astype(str).str.strip()
        df = df[df["location"].notna() & (df["location"] != "nan")]

    df = df.reset_index(drop=True)
    print(f"  [Mining Companies] OK {len(df):,} clean records")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5 · World Mining Commodities (production by country/year)
# ══════════════════════════════════════════════════════════════════════════════

def load_mining_commodities() -> pd.DataFrame:
    print("[Mining Commodities] loading …")
    df = pd.read_csv(MINING_COMM_PATH, low_memory=False)

    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True)
                  .str.strip("_")
    )

    df = _drop_full_duplicates(df, "Mining Commodities")

    # ── melt year columns into long format ──
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", c)]
    id_cols   = [c for c in df.columns if c not in year_cols]
    df = df.melt(id_vars=id_cols, value_vars=year_cols,
                 var_name="year", value_name="production")

    df["year"]       = df["year"].astype(int)
    df["production"] = pd.to_numeric(df["production"], errors="coerce")

    # ── remove zero/negative production ──
    df = df[df["production"].gt(0)]

    # ── standardise mineral name ──
    if "mined_raw_mat" in df.columns:
        df["mineral"] = df["mined_raw_mat"].astype(str).str.strip().str.lower()
    elif "commodity" in df.columns:
        df["mineral"] = df["commodity"].astype(str).str.strip().str.lower()

    # ── flag EV-relevant ──
    df["ev_relevant"] = df["mineral"].isin(EV_MINERALS)

    # ── standardise country ──
    if "country" in df.columns:
        df["country"] = _standardise_country(df["country"])

    df = df.reset_index(drop=True)
    print(f"  [Mining Commodities] OK {len(df):,} clean records")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6 · Commodity Info (periodic table metadata)
# ══════════════════════════════════════════════════════════════════════════════

def load_commodity_info() -> pd.DataFrame:
    print("[Commodity Info] loading …")
    df = pd.read_excel(COMMODITY_INFO_PATH, sheet_name="Sheet1")

    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True)
                  .str.strip("_")
    )

    # ── drop fully empty rows ──
    df = df.dropna(how="all")

    # ── coerce numeric metadata ──
    for col in ["atomic_no", "periodic_table_group", "periodic_table_period"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── strip string fields ──
    for col in ["commodity", "chemical_composition", "material_sub", "form"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df = _drop_full_duplicates(df, "Commodity Info")

    # ── flag EV-critical minerals ──
    df["ev_critical"] = df["commodity"].str.lower().isin(EV_MINERALS)

    df = df.reset_index(drop=True)
    print(f"  [Commodity Info] OK {len(df):,} clean records")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# convenience: load everything at once
# ══════════════════════════════════════════════════════════════════════════════

def load_all() -> dict:
    print("=" * 60)
    print("Loading & cleaning all datasets …")
    print("=" * 60)
    datasets = {
        "ev_population":    load_ev_population(),
        "ev_stations":      load_ev_stations(),
        "ports":            load_ports(),
        "mining_companies": load_mining_companies(),
        "mining_commodities": load_mining_commodities(),
        "commodity_info":   load_commodity_info(),
    }
    print("=" * 60)
    print("All datasets loaded OK")
    print("=" * 60)
    return datasets
