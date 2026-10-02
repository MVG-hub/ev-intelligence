"""pages/03_forecast.py"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from engine.ml_engine import forecast_regional_growth

ev = data["ev_population"]

# DARK layout — no xaxis/yaxis keys here to avoid double-kwarg conflict
DARK = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font_color="#e6edf3",
)
AXIS = dict(gridcolor="#21262d", zerolinecolor="#21262d", color="#e6edf3")
HOVER = dict(hoverlabel=dict(bgcolor="#161b22", bordercolor="#58a6ff",
                              font_size=13, font_color="#e6edf3"))

st.markdown("""
<div class="page-header">
  <h1>Predictive Regional Growth Forecast</h1>
  <p>Ridge regression models trained per state to forecast EV registrations through 2028.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Running forecasting models…")
def get_forecast(_ev):
    return forecast_regional_growth(_ev, forecast_years=[2025,2026,2027,2028], min_records=5)

forecast_df = get_forecast(ev)

if forecast_df.empty:
    st.error("Forecast could not be generated.")
    st.stop()

all_states = sorted(forecast_df["state"].unique().tolist())
top_states = (
    forecast_df[~forecast_df["is_forecast"]]
    .groupby("state")["registrations"].sum()
    .nlargest(10).index.tolist()
)

with st.sidebar:
    st.markdown("### Forecast Controls")
    selected_states = st.multiselect("States / Regions", all_states, default=top_states[:8])

tab1, tab2, tab3 = st.tabs(["Forecast Lines", "Growth Heatmap", "Data Table"])

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Year-over-Year Registrations with 2025–2028 Forecasts</div>',
                unsafe_allow_html=True)

    states_to_show = selected_states if selected_states else top_states
    sub = forecast_df[forecast_df["state"].isin(states_to_show)]
    palette = px.colors.qualitative.Plotly

    fig = go.Figure()
    for i, state in enumerate(states_to_show):
        s     = sub[sub["state"] == state].sort_values("year")
        hist  = s[~s["is_forecast"]]
        fore  = s[s["is_forecast"]]
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=hist["year"], y=hist["registrations"],
            mode="lines+markers", name=state,
            line=dict(color=color, width=2), marker=dict(size=6),
            hovertemplate=f"<b>{state}</b><br>Year: %{{x}}<br>Regs: %{{y:,}}<extra></extra>",
        ))
        if not fore.empty:
            conn = pd.concat([hist.tail(1), fore])
            fig.add_trace(go.Scatter(
                x=conn["year"], y=conn["registrations"],
                mode="lines+markers", showlegend=False,
                line=dict(color=color, width=2, dash="dash"),
                marker=dict(symbol="circle-open", size=7),
                hovertemplate=f"<b>{state} (forecast)</b><br>Year: %{{x}}<br>Regs: %{{y:,}}<extra></extra>",
            ))

    fig.add_vrect(x0=2024.5, x1=2028.5, fillcolor="rgba(31,111,235,0.06)", line_width=0,
                  annotation_text="  Forecast Period", annotation_position="top left",
                  annotation_font_color="#58a6ff")
    fig.update_layout(
        **DARK, **HOVER,
        height=480, hovermode="x unified",
        xaxis_title="Year", yaxis_title="Registrations",
        xaxis=dict(dtick=1, **AXIS),
        yaxis=dict(**AXIS),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Predicted Growth: 2028 vs 2024</div>', unsafe_allow_html=True)

    pivot = forecast_df.pivot_table(index="state", columns="year",
                                     values="registrations", aggfunc="sum")
    pivot.columns = [int(float(c)) for c in pivot.columns]

    if 2024 in pivot.columns and 2028 in pivot.columns:
        pivot["growth_pct"] = ((pivot[2028] - pivot[2024]) / pivot[2024].clip(lower=1)) * 100
        pivot = pivot.reset_index().sort_values("growth_pct", ascending=False)
        top20 = pivot.head(20).copy()

        fig2 = px.bar(
            top20, x="state", y="growth_pct",
            color="growth_pct",
            color_continuous_scale=[[0,"#da3633"],[0.5,"#e3b341"],[1,"#3fb950"]],
            labels={"growth_pct":"Growth % (2024→2028)", "state":"State"},
            text=top20["growth_pct"].round(0).astype(int).astype(str) + "%",
        )
        fig2.update_traces(textposition="outside", textfont_color="#e6edf3",
                           hovertemplate="<b>%{x}</b><br>Growth: %{y:.1f}%<extra></extra>",
                           hoverlabel=dict(bgcolor="#161b22", bordercolor="#58a6ff",
                                           font_size=13, font_color="#e6edf3"))
        fig2.update_layout(
            **DARK, height=400, coloraxis_showscale=False, xaxis_tickangle=-30,
            xaxis=dict(**AXIS), yaxis=dict(**AXIS),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Choropleth — US states only (2-letter uppercase)
        us_states = pivot[pivot["state"].str.match(r"^[A-Z]{2}$")].copy()
        if not us_states.empty:
            fig3 = px.choropleth(
                us_states, locations="state", locationmode="USA-states",
                color="growth_pct",
                color_continuous_scale=[[0,"#da3633"],[0.5,"#e3b341"],[1,"#3fb950"]],
                scope="usa", labels={"growth_pct":"Growth %"},
                hover_data={"growth_pct": ":.1f"},
            )
            fig3.update_geos(bgcolor="#0d1117", landcolor="#1c2128",
                              showlakes=False, showcoastlines=True, coastlinecolor="#30363d")
            fig3.update_layout(
                **DARK, height=420, margin=dict(t=0,b=0,l=0,r=0),
                coloraxis_colorbar=dict(title="Growth %", tickfont_color="#e6edf3",
                                         title_font_color="#e6edf3"),
            )
            fig3.update_traces(
                hovertemplate="<b>%{location}</b><br>Growth: %{z:.1f}%<extra></extra>",
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No 2-letter US state codes found in forecast data.")
    else:
        years_avail = sorted([c for c in pivot.columns if isinstance(c, int)])
        st.info(f"Need 2024 and 2028 columns. Available: {years_avail}")

# ── Tab 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Full Forecast Table</div>', unsafe_allow_html=True)
    display = forecast_df.copy()
    display["Type"] = display["is_forecast"].map({True:"Forecast", False:"Historical"})
    display = display.drop(columns=["is_forecast"])[["state","year","registrations","Type"]]
    display.columns = ["State","Year","Registrations","Type"]
    st.dataframe(display.sort_values(["State","Year"]), use_container_width=True,
                 height=500, hide_index=True)
    st.download_button("Download CSV", display.to_csv(index=False).encode(),
                       "ev_forecast.csv", "text/csv")
