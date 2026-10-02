# Dataset Setup

The large dataset files are **not included in this repository** (gitignored).  
Download them from Kaggle before running the app.

## Option A — Manual Download (Kaggle UI)

Download each file from Kaggle and place it in the **project root** (same folder as `ev_intelligence/`):

| File | Kaggle URL |
|------|-----------|
| `Electric_Vehicle_Population_Data.csv` | https://www.kaggle.com/datasets/ratikkakkar/electric-vehicle-population-data |
| `ev_stations_2025.csv` | https://www.kaggle.com/datasets/viveksanpat/ev-charging-stations-global |
| `World_Port_Index.csv` | https://www.kaggle.com/datasets/tridenttails/world-port-index |
| `116_world_mining_companies_clean.csv` | https://www.kaggle.com/datasets/bhanupratapbiswas/mining-companies |
| `world_mining_commodities_clean.csv` | https://www.kaggle.com/datasets/joebeachcapital/mineral-production-by-country |
| `commodity_info.xlsx` | Bundled — see `ev_intelligence/data/` |

## Option B — Kaggle API (automated)

1. Install the Kaggle CLI: `pip install kaggle`
2. Copy `kaggle.json.example` to `~/.kaggle/kaggle.json` and add your credentials
3. Run:

```bash
kaggle datasets download -d ratikkakkar/electric-vehicle-population-data --unzip
kaggle datasets download -d viveksanpat/ev-charging-stations-global --unzip
kaggle datasets download -d tridenttails/world-port-index --unzip
kaggle datasets download -d bhanupratapbiswas/mining-companies --unzip
kaggle datasets download -d joebeachcapital/mineral-production-by-country --unzip
```

## Expected directory structure after download

```
Kaggle Dashboards/
├── Electric_Vehicle_Population_Data.csv
├── ev_stations_2025.csv
├── World_Port_Index.csv
├── 116_world_mining_companies_clean.csv
├── world_mining_commodities_clean.csv
├── commodity_info.xlsx
└── ev_intelligence/
    ├── app.py
    └── ...
```
