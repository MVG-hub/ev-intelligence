"""pages/06_rag_policy.py"""
import pandas as pd
import plotly.graph_objects as go
from engine.rag_engine import index_policy_docs, query_policies, compute_tariff_impact
from engine.ml_engine  import build_cluster_features, run_kmeans

ev    = data["ev_population"]
sta   = data["ev_stations"]
ports = data["ports"]

st.markdown("""
<div class="page-header">
  <h1>Policy &amp; Tariff RAG Simulation</h1>
  <p>Semantic search over government policy documents (ChromaDB) combined with a dynamic tariff impact calculator.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Initialising RAG vector store…")
def init_rag():
    return index_policy_docs()

n_indexed = init_rag()
st.sidebar.success(f"RAG ready — {n_indexed} chunks indexed" if n_indexed > 0 else "RAG ready (cached)")

@st.cache_data(show_spinner="Loading plant sites…")
def get_centroids(_ev, _sta, _ports):
    feat = build_cluster_features(_ev, _sta, _ports, grid_resolution=1.0)
    _, cents = run_kmeans(feat, n_clusters=7)
    return cents

centroids = get_centroids(ev, sta, ports)

DARK = dict(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#e6edf3",
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"))

tab1, tab2 = st.tabs(["Policy Search (RAG)", "Tariff Simulator"])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 · RAG Search
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Semantic Policy Document Search</div>', unsafe_allow_html=True)
    st.markdown(
        "Ask any question about EV tax credits, import duties, subsidies, or trade policy. "
        "Results are retrieved from the indexed policy corpus using vector similarity."
    )

    EXAMPLES = [
        "What is the US federal EV tax credit in 2024?",
        "What tariff does the EU impose on Chinese EVs?",
        "How does India FAME scheme support EV manufacturing?",
        "What are the critical mineral sourcing requirements for US IRA?",
        "What export controls has China placed on battery minerals?",
        "What is the carbon border adjustment mechanism for EVs?",
    ]

    col_q, col_e = st.columns([3,2])
    with col_q:
        query_input = st.text_input("Your question:", placeholder="e.g. What is the federal EV tax credit?")
    with col_e:
        example = st.selectbox("Or pick an example:", ["— select —"] + EXAMPLES)
        if example != "— select —":
            query_input = example

    c1, c2 = st.columns([1,4])
    with c1:
        top_k = st.slider("Results", 2, 8, 4)
    with c2:
        search_btn = st.button("Search Policy Documents", type="primary")

    if search_btn and query_input:
        with st.spinner("Searching vector store…"):
            results = query_policies(query_input, top_k=top_k)

        if results:
            st.success(f"Found {len(results)} relevant passages")
            for i, r in enumerate(results, 1):
                relevance = round((1 - r["distance"]) * 100, 1)
                with st.expander(f"Result {i} — {r['source']}  (relevance: {relevance}%)", expanded=(i==1)):
                    st.progress(min(1.0, max(0.0, 1 - r["distance"])))
                    st.markdown(f"""
                    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                                padding:1rem;font-size:0.85rem;color:#e6edf3;
                                font-family:monospace;white-space:pre-wrap;">{r['text']}</div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No results found. Try a different query.")

    st.markdown("---")
    st.markdown('<div class="section-title">Indexed Policy Documents</div>', unsafe_allow_html=True)
    import os, glob as _glob
    policy_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "policy_docs"))
    files = (
        _glob.glob(os.path.join(policy_dir, "**/*.txt"), recursive=True) +
        _glob.glob(os.path.join(policy_dir, "**/*.md"), recursive=True)
    )
    if files:
        cols = st.columns(3)
        for i, f in enumerate(sorted(files)):
            cols[i % 3].markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:6px;
                        padding:0.6rem 0.8rem;margin-bottom:0.5rem;font-size:0.8rem;color:#58a6ff;">
              {os.path.basename(f)}
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No documents found — they will be auto-seeded on next init.")

    st.markdown("""
    <div style="background:#161b22;border:1px solid #1f6feb;border-radius:8px;
                padding:0.8rem 1rem;margin-top:0.5rem;font-size:0.82rem;color:#8b949e;">
      <b style="color:#58a6ff;">Add your own policy docs:</b>
      Drop <code>.txt</code> or <code>.md</code> files into
      <code>ev_intelligence/policy_docs/</code> then click Re-index below.
    </div>
    """, unsafe_allow_html=True)
    if st.button("Re-index Policy Documents"):
        index_policy_docs(force_reindex=True)
        st.cache_resource.clear()
        st.success("Re-indexing complete — reload the page.")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 · Tariff Impact Simulator
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Dynamic Tariff &amp; Incentive Impact Simulator</div>', unsafe_allow_html=True)

    # ── Preset selector ───────────────────────────────────────────────────────
    PRESET_VALS = {
        "Custom":                                  None,
        "Scenario A — Status Quo (2024)":          (0.0,  0.0,     0.0,    7500.0),
        "Scenario B — Trade War Escalation":       (25.0, 10.0,    0.0,    3750.0),
        "Scenario C — Green Trade Corridor":       (0.0,  0.0,     0.0,   10000.0),
        "Scenario D — Carbon Border Tax (2026)":   (15.0, 5.0,  2000.0,   7500.0),
    }

    preset = st.selectbox("Load Scenario Preset:", list(PRESET_VALS.keys()))

    # When a named preset is chosen, use its values directly.
    # When Custom, read slider values.
    preset_tuple = PRESET_VALS.get(preset)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Cost Drivers")
        import_duty  = st.slider("Import Duty on Components (%)",       0.0, 150.0,
                                  float(preset_tuple[0]) if preset_tuple else 0.0, 2.5)
        mineral_duty = st.slider("Mineral Import Duty (%)",             0.0, 50.0,
                                  float(preset_tuple[1]) if preset_tuple else 0.0, 1.0)
        carbon_levy  = st.slider("Carbon Border Levy (USD/vehicle)",    0.0, 5000.0,
                                  float(preset_tuple[2]) if preset_tuple else 0.0, 100.0)
    with col2:
        st.markdown("#### Incentives")
        tax_credit   = st.slider("Local Tax Credit (USD/vehicle)",      0.0, 15000.0,
                                  float(preset_tuple[3]) if preset_tuple else 7500.0, 250.0)

        # Key assumptions callout
        st.markdown("""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                    padding:0.9rem;margin-top:0.5rem;font-size:0.78rem;color:#8b949e;line-height:1.6;">
          <b style="color:#e6edf3;">Model Assumptions</b><br>
          Component import cost: <b style="color:#58a6ff;">$8,000/vehicle</b><br>
          Mineral import cost: <b style="color:#58a6ff;">$3,500/vehicle</b><br>
          Avg vehicle price: <b style="color:#58a6ff;">$45,000</b><br>
          Viability penalty: <b style="color:#58a6ff;">0.004 per 1% cost increase</b>
        </div>
        """, unsafe_allow_html=True)

    # Use preset values directly when a named scenario is active;
    # otherwise use whatever the sliders show.
    if preset_tuple is not None:
        eff_import_duty  = float(preset_tuple[0])
        eff_mineral_duty = float(preset_tuple[1])
        eff_carbon_levy  = float(preset_tuple[2])
        eff_tax_credit   = float(preset_tuple[3])
    else:
        eff_import_duty  = import_duty
        eff_mineral_duty = mineral_duty
        eff_carbon_levy  = carbon_levy
        eff_tax_credit   = tax_credit

    st.markdown("---")

    # ── Compute results for all sites ────────────────────────────────────────
    result_rows = []
    for _, row in centroids.iterrows():
        impact = compute_tariff_impact(
            row["viability_score"],
            import_duty_pct      = eff_import_duty,
            local_tax_credit_usd = eff_tax_credit,
            carbon_levy_usd      = eff_carbon_levy,
            mineral_duty_pct     = eff_mineral_duty,
        )
        result_rows.append({
            "Site":               f"#{int(row['rank'])}",
            "GPS":                f"{row['latitude']:.2f}°, {row['longitude']:.2f}°",
            "Base Viability":     round(row["viability_score"], 4),
            "Adjusted Viability": impact["adjusted_viability"],
            "Delta":              round(impact["adjusted_viability"] - row["viability_score"], 4),
            "Net Cost (USD)":     f"${impact['cost_increase_usd']:+,.0f}",
            "Cost Impact (%)":    f"{impact['cost_increase_pct']:+.2f}%",
        })
    results_df = pd.DataFrame(result_rows)

    # ── Bar chart: before / after ────────────────────────────────────────────
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name="Base Viability", x=results_df["Site"],
        y=results_df["Base Viability"], marker_color="#30363d",
        hovertemplate="<b>%{x}</b><br>Base: %{y:.4f}<extra></extra>",
    ))
    fig2.add_trace(go.Bar(
        name="Adjusted Viability", x=results_df["Site"],
        y=results_df["Adjusted Viability"],
        marker_color=[("#3fb950" if v >= b else "#da3633")
                      for v, b in zip(results_df["Adjusted Viability"], results_df["Base Viability"])],
        hovertemplate="<b>%{x}</b><br>Adjusted: %{y:.4f}<extra></extra>",
    ))
    fig2.update_layout(
        **DARK, barmode="group", height=340,
        xaxis_title="Plant Site", yaxis_title="Viability Score",
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#161b22", bordercolor="#58a6ff",
                        font_size=13, font_color="#e6edf3"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Scenario label ────────────────────────────────────────────────────────
    if preset_tuple is not None:
        st.info(f"Showing results for preset: **{preset}** — sliders above show reference values only.")

    # ── Table ─────────────────────────────────────────────────────────────────
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # ── Narrative ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Policy Impact Narrative</div>', unsafe_allow_html=True)
    impact_top = compute_tariff_impact(
        centroids.iloc[0]["viability_score"],
        import_duty_pct=eff_import_duty, local_tax_credit_usd=eff_tax_credit,
        carbon_levy_usd=eff_carbon_levy, mineral_duty_pct=eff_mineral_duty,
    )
    st.markdown(f"""
    <div style="background:#161b22;border-left:4px solid #58a6ff;border-radius:0 8px 8px 0;
                padding:1rem 1.2rem;font-size:0.88rem;color:#e6edf3;line-height:1.7;">
      {impact_top['policy_summary']}
    </div>
    """, unsafe_allow_html=True)

    # ── RAG cross-reference ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">Relevant Policy Context (RAG)</div>', unsafe_allow_html=True)
    rag_q = f"import duty {eff_import_duty:.0f} percent tax credit {eff_tax_credit:.0f} EV manufacturing"
    rag_results = query_policies(rag_q, top_k=3)
    for r in rag_results:
        rel = round((1 - r["distance"]) * 100, 1)
        with st.expander(f"{r['source']}  (relevance: {rel}%)"):
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:6px;
                        padding:0.8rem;font-size:0.82rem;color:#e6edf3;
                        font-family:monospace;white-space:pre-wrap;">{r['text']}</div>
            """, unsafe_allow_html=True)
