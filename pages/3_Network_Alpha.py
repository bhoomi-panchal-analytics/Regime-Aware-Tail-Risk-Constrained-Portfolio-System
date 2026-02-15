import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.load_data import load_all

st.title("Contagion Network & Survival vs Alpha Diagnostics")

# =========================================================
# LOAD DATA
# =========================================================

data = load_all()
contagion = data.get("contagion", pd.DataFrame())
market_data = data.get("market_data", pd.DataFrame())

# =========================================================
# DATE FILTER
# =========================================================

if not contagion.empty:
    min_date = contagion.index.min()
    max_date = contagion.index.max()

    start_date, end_date = st.slider(
        "Select Time Window",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(min_date.to_pydatetime(), max_date.to_pydatetime())
    )

    mask = (contagion.index >= start_date) & (contagion.index <= end_date)
    contagion_filtered = contagion.loc[mask]

else:
    contagion_filtered = pd.DataFrame()

# =========================================================
# CONTAGION INDEX PLOT
# =========================================================

st.subheader("1️⃣ Contagion Index Over Time")

if contagion_filtered.empty:
    st.error("Contagion data missing. Add contagion.csv to /data.")
else:
    fig = px.line(
        contagion_filtered,
        y=contagion_filtered.columns[0],
        title="Systemic Contagion Level",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# ROLLING CONTAGION VOLATILITY
# =========================================================

st.subheader("2️⃣ Rolling Contagion Volatility")

if not contagion_filtered.empty:
    rolling_vol = contagion_filtered.iloc[:, 0].rolling(30).std()

    fig2 = px.line(
        rolling_vol,
        title="30-Day Rolling Contagion Volatility"
    )
    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# DISTRIBUTION OF CONTAGION
# =========================================================

st.subheader("3️⃣ Contagion Distribution")

if not contagion_filtered.empty:
    fig3 = px.histogram(
        contagion_filtered,
        x=contagion_filtered.columns[0],
        nbins=50,
        title="Distribution of Contagion Levels"
    )
    st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# NETWORK SNAPSHOT (SIMPLIFIED MATRIX)
# =========================================================

st.subheader("4️⃣ Correlation Matrix (Market Data)")

if market_data.empty:
    st.warning("Market data missing. Cannot compute correlation network.")
else:
    market_filtered = market_data.loc[start_date:end_date]
    corr = market_filtered.pct_change().corr()

    fig4 = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Asset Correlation Matrix"
    )
    st.plotly_chart(fig4, use_container_width=True)

# =========================================================
# SURVIVAL VS ALPHA MATRIX
# =========================================================

st.subheader("5️⃣ Survival vs Alpha Matrix")

if market_data.empty:
    st.warning("Market data missing.")
else:
    returns = market_data.pct_change().dropna()

    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    drawdown = (returns.cumsum().min())

    metrics_df = pd.DataFrame({
        "Sharpe": sharpe,
        "Volatility": returns.std(),
        "Mean Return": returns.mean()
    })

    fig5 = px.scatter(
        metrics_df,
        x="Volatility",
        y="Sharpe",
        size="Mean Return",
        text=metrics_df.index,
        title="Alpha vs Risk Trade-off"
    )
    fig5.update_traces(textposition="top center")
    st.plotly_chart(fig5, use_container_width=True)

# =========================================================
# CONTAGION REGIME ZONES
# =========================================================

st.subheader("6️⃣ Contagion Stress Zones")

if not contagion_filtered.empty:
    threshold_high = contagion_filtered.iloc[:, 0].quantile(0.8)
    threshold_low = contagion_filtered.iloc[:, 0].quantile(0.2)

    stress_zones = pd.DataFrame({
        "Date": contagion_filtered.index,
        "Level": contagion_filtered.iloc[:, 0]
    })

    fig6 = go.Figure()

    fig6.add_trace(go.Scatter(
        x=stress_zones["Date"],
        y=stress_zones["Level"],
        mode="lines",
        name="Contagion"
    ))

    fig6.add_hline(y=threshold_high, line_color="red")
    fig6.add_hline(y=threshold_low, line_color="green")

    fig6.update_layout(title="High vs Low Contagion Regimes")
    st.plotly_chart(fig6, use_container_width=True)

# =========================================================
# NETWORK DENSITY PROXY
# =========================================================

st.subheader("7️⃣ Network Density Proxy")

if not market_data.empty:
    rolling_corr = market_data.pct_change().rolling(30).corr().mean(level=0)
    density = rolling_corr.mean(axis=1)

    fig7 = px.line(
        density,
        title="Average Rolling Correlation (Network Density Proxy)"
    )
    st.plotly_chart(fig7, use_container_width=True)

st.markdown("---")

st.markdown("""
### Interpretation Guide

• Rising contagion index → systemic stress increasing  
• High rolling correlation → diversification failing  
• Sharpe vs volatility scatter → survival vs alpha tradeoff  
• Upper stress zone → capital preservation mode  

This page demonstrates how contagion dynamics impact portfolio resilience.
""")
