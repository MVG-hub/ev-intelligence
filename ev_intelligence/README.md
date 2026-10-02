# ⚡ EV Intelligence Platform

**Global EV Market Expansion & Site Selection** — an end-to-end data intelligence system combining EDA, machine learning, geospatial analysis, and RAG-powered policy simulation.

---

## 📁 Project Structure

```
Kaggle Dashboards/
├── Electric_Vehicle_Population_Data.csv
├── ev_stations_2025.csv
├── World_Port_Index.csv
├── 116_world_mining_companies_clean.csv
├── world_mining_commodities_clean.csv
├── commodity_info.xlsx
│
└── ev_intelligence/
    ├── app.py                      ← Streamlit entry point
    ├── requirements.txt
    ├── policy_docs/                ← Drop .txt/.md policy files here
    │   ├── us_ev_tax_credit_2024.txt   (auto-seeded)
    │   ├── eu_ev_policy_2025.txt       (auto-seeded)
    │   ├── india_ev_policy_2025.txt    (auto-seeded)
    │   ├── china_ev_subsidy_2024.txt   (auto-seeded)
    │   └── global_tariff_scenarios.txt (auto-seeded)
    ├── chroma_store/               ← ChromaDB persistent vector DB (auto-created)
    ├── engine/
    │   ├── __init__.py
    │   ├── data_loader.py          ← Data ingestion & deep cleaning
    │   ├── ml_engine.py            ← EDA, forecasting, K-Means clustering
    │   └── rag_engine.py           ← ChromaDB indexing + tariff simulation
    └── pages/
        ├── 00_overview.py          ← KPI dashboard
        ├── 01_eda.py               ← EDA & market trends
        ├── 02_map_demand.py        ← Map 1: Demand & charging heatmap
        ├── 03_forecast.py          ← Ridge regression growth forecast
        ├── 04_clustering.py        ← K-Means plant site clustering
        ├── 05_map_supply.py        ← Map 2: Supply chain & port logistics
        └── 06_rag_policy.py        ← RAG search + tariff simulator
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
cd "ev_intelligence"
pip install -r requirements.txt
```

### 2. Run the app
```bash
# From the "Kaggle Dashboards" root directory:
streamlit run ev_intelligence/app.py
```

The app will open at **http://localhost:8501**

---

## 🔬 Modules

| Page | Description |
|------|-------------|
| 🏠 Overview | KPI cards + dataset summaries + system architecture diagram |
| 📊 EDA & Market Trends | BEV/PHEV trend, price vs range scatter, brand share, range evolution, mineral production |
| 🗺️ Map 1 · Demand & Charging | Dark-mode bubble map + density heatmap of all registrations + global charging stations |
| 🔮 Growth Forecast | Ridge regression forecasts per US state through 2028 + choropleth heatmap |
| 🏭 Plant Site Clustering | K-Means (configurable K) → GPS coordinates ranked by viability score |
| 🌊 Map 2 · Supply Chain | Plant sites + nearest ports (with route lines) + mineral source overlays |
| 📜 Policy & Tariff RAG | ChromaDB semantic search + interactive tariff simulator with scenario presets |

---

## 🤖 AI / ML Concepts Used

| Concept | Where |
|---------|-------|
| **Data Cleaning & Validation** | `engine/data_loader.py` — dedup, coercion, WKT parsing, range clipping |
| **Exploratory Data Analysis** | `engine/ml_engine.py` + `pages/01_eda.py` |
| **Time-Series Regression (Ridge)** | `engine/ml_engine.py → forecast_regional_growth()` |
| **Unsupervised ML (K-Means)** | `engine/ml_engine.py → run_kmeans()` — demand × infra × port proximity |
| **Viability Scoring** | Weighted multi-objective formula over cluster features |
| **RAG (Retrieval-Augmented Generation)** | `engine/rag_engine.py` — ChromaDB + sentence-transformer embeddings |
| **Geospatial Analysis** | Haversine distance, bubble maps, choropleth, density heatmaps |
| **Tariff Impact Simulation** | `engine/rag_engine.py → compute_tariff_impact()` |

---

## 📂 Adding Policy Documents

Drop any `.txt` or `.md` policy files into `ev_intelligence/policy_docs/` then click **♻️ Re-index Policy Documents** inside the app (Policy & Tariff RAG page). The vector store updates automatically.

---

## 🗃️ Data Sources

| Dataset | Source |
|---------|--------|
| Electric Vehicle Population Data | Kaggle / Washington State DOL |
| EV Charging Stations 2025 | Kaggle / Open Charge Map |
| World Port Index | NGA (National Geospatial-Intelligence Agency) |
| World Mining Companies | Kaggle |
| World Mining Commodities | Kaggle / USGS |
| Commodity Info (periodic table) | Compiled reference |
