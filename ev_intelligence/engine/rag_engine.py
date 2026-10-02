"""
rag_engine.py
-------------
RAG pipeline:
  • Indexes policy_docs/ folder into a local ChromaDB vector store
  • Accepts natural-language queries and returns top-k relevant chunks
  • Provides a tariff_impact() function that calculates how a given
    import duty / tax credit shifts a cluster's viability score
"""

import os
import glob
import hashlib
import textwrap

import chromadb
from chromadb.utils import embedding_functions

# ── paths ──────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(__file__)
POLICY_DIR  = os.path.normpath(os.path.join(_HERE, "..", "policy_docs"))
CHROMA_DIR  = os.path.normpath(os.path.join(_HERE, "..", "chroma_store"))

# ── Secrets / API keys loaded from environment (never hardcoded) ───────────────
# All keys are optional — the app works fully without them using built-in models.
# Set these in .streamlit/secrets.toml (local) or as environment variables (CI/CD).
def _get_secret(key: str, default: str = "") -> str:
    """Read a secret from st.secrets (Streamlit Cloud) or env var (local/CI)."""
    # Try Streamlit secrets first (available when deployed on Streamlit Cloud)
    try:
        import streamlit as st
        return st.secrets.get(key, os.environ.get(key, default))
    except Exception:
        return os.environ.get(key, default)

MAPBOX_TOKEN  = _get_secret("MAPBOX_TOKEN")   # optional — for premium map tiles
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY") # optional — for OpenAI embeddings

# Use the built-in sentence-transformer embedding function (no API key needed)
# Switch to OpenAI embeddings by setting OPENAI_API_KEY and uncommenting below:
# if OPENAI_API_KEY:
#     _EMBED_FN = embedding_functions.OpenAIEmbeddingFunction(
#         api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
#     )
# else:
_EMBED_FN = embedding_functions.DefaultEmbeddingFunction()

COLLECTION_NAME = "ev_policy_docs"


# ══════════════════════════════════════════════════════════════════════════════
# ChromaDB client (persistent)
# ══════════════════════════════════════════════════════════════════════════════

def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_EMBED_FN,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ══════════════════════════════════════════════════════════════════════════════
# Indexing
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """Split text into overlapping chunks of ~chunk_size characters."""
    words = text.split()
    chunks, current, count = [], [], 0
    for word in words:
        current.append(word)
        count += len(word) + 1
        if count >= chunk_size:
            chunks.append(" ".join(current))
            # keep overlap window
            overlap_words = current[-(overlap // 6):]
            current = overlap_words
            count = sum(len(w) + 1 for w in current)
    if current:
        chunks.append(" ".join(current))
    return chunks


def index_policy_docs(force_reindex: bool = False) -> int:
    """
    Walk policy_docs/ and index every .txt / .md file into ChromaDB.
    Returns total number of chunks indexed.

    If a document with the same content hash already exists it is skipped
    (idempotent), unless force_reindex=True.
    """
    os.makedirs(POLICY_DIR, exist_ok=True)
    collection = _get_collection()

    files = (
        glob.glob(os.path.join(POLICY_DIR, "**/*.txt"), recursive=True) +
        glob.glob(os.path.join(POLICY_DIR, "**/*.md"),  recursive=True)
    )

    if not files:
        # seed with built-in sample policies so the app always has content
        _seed_sample_policies()
        files = glob.glob(os.path.join(POLICY_DIR, "*.txt"))

    indexed = 0
    for fpath in files:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
            raw = fh.read().strip()
        if not raw:
            continue

        file_hash = hashlib.md5(raw.encode()).hexdigest()[:12]
        fname     = os.path.basename(fpath)

        chunks = _chunk_text(raw)
        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            doc_id = f"{file_hash}_{i}"
            # skip if already present and not force re-indexing
            if not force_reindex:
                existing = collection.get(ids=[doc_id])
                if existing["ids"]:
                    continue
            ids.append(doc_id)
            docs.append(chunk)
            metas.append({"source": fname, "chunk": i})

        if ids:
            collection.add(documents=docs, ids=ids, metadatas=metas)
            indexed += len(ids)

    return indexed


def _seed_sample_policies():
    """Write sample policy documents so the RAG pipeline always has data."""
    os.makedirs(POLICY_DIR, exist_ok=True)
    samples = {
        "us_ev_tax_credit_2024.txt": textwrap.dedent("""\
            US Federal EV Tax Credit (IRA 2024 Update)
            -------------------------------------------
            The Inflation Reduction Act provides a $7,500 federal tax credit for new
            battery electric vehicles (BEVs) meeting North American assembly requirements
            and critical mineral sourcing thresholds.  PHEVs with battery capacity >= 7 kWh
            qualify for up to $3,750.  Income caps apply: $150,000 (single) / $300,000
            (joint).  Used EV credit: up to $4,000 for vehicles <= $25,000.
            Critical mineral requirement: 40 % of battery minerals must be sourced from
            free-trade-agreement partners or domestic US mines.  This rises to 80 % by 2027.
            Import duty on Chinese-manufactured EVs: 100 % tariff effective 2024.
            Import duty on batteries from non-FTA countries: 25 % tariff.
        """),
        "eu_ev_policy_2025.txt": textwrap.dedent("""\
            EU Green Deal & EV Mandate 2025
            --------------------------------
            The European Union mandates a full phase-out of new internal-combustion-engine
            (ICE) vehicle sales by 2035.  Interim fleet CO2 targets: -55 % by 2030 vs 2021.
            EV purchase subsidies vary by member state: Germany offers EUR 3,000 (reduced
            from EUR 6,000), France offers EUR 5,000 bonus malus, Netherlands EUR 2,950.
            The EU Carbon Border Adjustment Mechanism (CBAM) imposes carbon-import levies
            on manufactured goods including EV components from high-emission jurisdictions.
            Anti-subsidy tariffs on Chinese EVs: provisional 17–38 % (2024), subject to
            WTO dispute review.
            Battery passport regulation effective 2026: all batteries > 2 kWh must carry
            supply-chain provenance data.
        """),
        "india_ev_policy_2025.txt": textwrap.dedent("""\
            India FAME-III & EV Import Duty Framework
            ------------------------------------------
            FAME-III (2024-2027) budget: INR 500 billion.  Subsidies target two-wheelers
            (INR 10,000/kWh), three-wheelers (INR 15,000/kWh), and electric buses.
            GST on EVs reduced to 5 % (from 28 % for ICE vehicles).
            Import duty on complete EVs (CBU): 100 % for vehicles >USD 40,000 MSRP;
            15 % concessional duty for manufacturers committing to INR 4,150 crore
            investment and 25 % localisation within 3 years.
            PLI (Production-Linked Incentive) scheme for Advanced Chemistry Cell batteries:
            INR 181 billion for 50 GWh domestic manufacturing capacity.
            Charging infrastructure: government target 2,636 charging stations on highways
            and expressways by 2025.
        """),
        "china_ev_subsidy_2024.txt": textwrap.dedent("""\
            China NEV Policy & Subsidy Landscape
            -------------------------------------
            China's national NEV (New Energy Vehicle) subsidy programme ended in Dec 2022.
            Post-subsidy support: purchase tax exemption (100 %) extended to end-2025.
            Local government subsidies persist in Beijing, Shanghai, Shenzhen.
            Export tariffs: China imposes 0 % export duty on EVs.  Retaliatory tariffs
            from EU (17-38 %), US (100 %), and Canada (100 %) reduce export competitiveness.
            Domestic EV penetration: 31 % of new car sales in 2024.
            Battery-grade lithium carbonate price: CNY 80,000–100,000/tonne (2024).
            Critical mineral export controls: graphite (2023), gallium, germanium,
            and antimony (2024) subject to export licence requirements.
        """),
        "global_tariff_scenarios.txt": textwrap.dedent("""\
            Global EV Tariff Scenario Summary
            ----------------------------------
            Scenario A – Status Quo (2024):
              US tariff on Chinese EVs: 100 %.  EU provisional tariff: 17-38 %.
              Lithium import duty (US): 0 %.  Cobalt import duty (US): 0 %.

            Scenario B – Trade War Escalation:
              US extends 25 % tariff to all non-USMCA EV components.
              EU raises definitive tariff on Chinese EVs to 45 %.
              India raises CBU import duty to 125 %.
              Impact: +12-18 % manufacturing cost increase for non-domestic producers.

            Scenario C – Green Trade Corridor:
              US-EU-Japan bilateral critical mineral agreement: 0 % duty on EV minerals.
              Mutual recognition of EV safety standards reduces homologation costs ~3 %.
              Impact: -8 % total vehicle cost for transatlantic manufacturers.

            Scenario D – Carbon Border Tax Full Implementation (2026):
              EU CBAM covers EV batteries and motors.
              Carbon price assumed: EUR 65/tonne CO2.
              Estimated surcharge on imports from high-coal-power grids: EUR 1,200–2,800/vehicle.
        """),
    }
    for fname, content in samples.items():
        fpath = os.path.join(POLICY_DIR, fname)
        if not os.path.exists(fpath):
            with open(fpath, "w", encoding="utf-8") as fh:
                fh.write(content)


# ══════════════════════════════════════════════════════════════════════════════
# Query
# ══════════════════════════════════════════════════════════════════════════════

def query_policies(query_text: str, top_k: int = 5) -> list[dict]:
    """
    Semantic search over indexed policy documents.

    Returns a list of dicts:
      { 'source': filename, 'chunk': chunk_index, 'text': str, 'distance': float }
    """
    collection = _get_collection()
    if collection.count() == 0:
        index_policy_docs()

    results = collection.query(
        query_texts=[query_text],
        n_results=min(top_k, max(1, collection.count())),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "source":   meta.get("source", "unknown"),
            "chunk":    meta.get("chunk", 0),
            "text":     doc,
            "distance": round(float(dist), 4),
        })
    return output


# ══════════════════════════════════════════════════════════════════════════════
# Tariff impact simulation
# ══════════════════════════════════════════════════════════════════════════════

TARIFF_PARAMS = {
    "import_duty_pct":      0.0,   # % tariff on imported EV components
    "local_tax_credit_usd": 0.0,   # USD tax credit per vehicle
    "carbon_levy_usd":      0.0,   # USD carbon border levy per vehicle
    "mineral_duty_pct":     0.0,   # % duty on imported battery minerals
}

# Baseline assumptions (USD per vehicle)
_BASELINE_COMPONENT_IMPORT_COST = 8_000
_BASELINE_MINERAL_COST          = 3_500
_AVERAGE_VEHICLE_PRICE          = 45_000


def compute_tariff_impact(
    base_viability: float,
    import_duty_pct: float     = 0.0,
    local_tax_credit_usd: float = 0.0,
    carbon_levy_usd: float     = 0.0,
    mineral_duty_pct: float    = 0.0,
) -> dict:
    """
    Simulate how tariff / tax-credit parameters shift a cluster's viability score.

    Returns a dict with:
      adjusted_viability, cost_increase_usd, cost_increase_pct,
      net_incentive_usd, policy_summary (natural-language string)
    """
    component_tariff_cost = _BASELINE_COMPONENT_IMPORT_COST * (import_duty_pct / 100)
    mineral_tariff_cost   = _BASELINE_MINERAL_COST          * (mineral_duty_pct / 100)
    total_cost_increase   = component_tariff_cost + mineral_tariff_cost + carbon_levy_usd
    net_cost_delta        = total_cost_increase - local_tax_credit_usd

    cost_increase_pct = (net_cost_delta / _AVERAGE_VEHICLE_PRICE) * 100

    # Viability penalty: each 1 % cost increase reduces score by 0.004
    viability_delta     = -0.004 * cost_increase_pct
    adjusted_viability  = max(0.0, min(1.0, round(base_viability + viability_delta, 4)))

    lines = []
    if import_duty_pct > 0:
        lines.append(f"Import duty of {import_duty_pct:.1f}% on components adds ${component_tariff_cost:,.0f}/vehicle.")
    if mineral_duty_pct > 0:
        lines.append(f"Mineral import duty of {mineral_duty_pct:.1f}% adds ${mineral_tariff_cost:,.0f}/vehicle.")
    if carbon_levy_usd > 0:
        lines.append(f"Carbon border levy adds ${carbon_levy_usd:,.0f}/vehicle.")
    if local_tax_credit_usd > 0:
        lines.append(f"Local tax credit of ${local_tax_credit_usd:,.0f}/vehicle offsets costs.")
    if not lines:
        lines.append("No active policy parameters — baseline viability applies.")

    net_label = "net increase" if net_cost_delta >= 0 else "net saving"
    lines.append(f"Net {net_label}: ${abs(net_cost_delta):,.0f}/vehicle ({abs(cost_increase_pct):.2f}%).")
    lines.append(f"Viability score: {base_viability:.4f} -> {adjusted_viability:.4f}.")

    return {
        "adjusted_viability":  adjusted_viability,
        "cost_increase_usd":   round(net_cost_delta, 2),
        "cost_increase_pct":   round(cost_increase_pct, 4),
        "net_incentive_usd":   round(local_tax_credit_usd - total_cost_increase, 2),
        "policy_summary":      " ".join(lines),
    }
