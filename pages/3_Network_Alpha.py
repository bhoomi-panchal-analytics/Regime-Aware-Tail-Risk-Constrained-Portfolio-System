import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.load_data import load_all

st.title("Contagion Network & Survival vs Alpha")

# ======================================================
# LOAD DATA
# ======================================================

data = load_all()
market_data = data.get("market_data", pd.DataFrame())
contagion = data.get("contagion", pd.DataFrame())

if market_data.empty:
    st.error("market_data.csv missing inside /data folder.")
    st.stop()

# ======================================================
# DATE FILTER
# ======================================================

min_date = market_data.index.min()
max_date = market_data.index.max()

start_date, end_date = st.slider(
    "Select Analysis Window",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime())
)

market_filtered = market_data.loc[start_date:end_date]
returns = market_filtered.pct_change().dropna()

st.markdown("---")

# ======================================================
# 1️⃣ CORRELATION MATRIX
# ======================================================

st.subheader("1️⃣ Correlation Matrix")

corr = returns.corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Asset Correlation Structure"
)

st.plotly_chart(fig_corr, use_container_width=True)

# ======================================================
# 2️⃣ ROLLING CORRELATION DENSITY
# ======================================================

st.subheader("2️⃣ Rolling Network Density")

rolling_corr = returns.rolling(30).corr().groupby(level=0).mean()
density = rolling_corr.mean(axis=1)

fig_density = px.line(
    density,
    title="Average Rolling Correlation (Diversification Breakdown)"
)

st.plotly_chart(fig_density, use_container_width=True)

# ======================================================
# 3️⃣ CONTAGION INDEX (REAL OR PROXY)
# ======================================================

st.subheader("3️⃣ Contagion Index")

if not contagion.empty:
    contagion_filtered = contagion.loc[start_date:end_date]
    fig_contagion = px.line(
        contagion_filtered,
        y=contagion_filtered.columns[0],
        title="Observed Contagion Index"
    )
else:
    proxy = density / density.max()
    fig_contagion = px.line(
        proxy,
        title="Synthetic Contagion Proxy (Derived from Correlation)"
    )

st.plotly_chart(fig_contagion, use_container_width=True)

# ======================================================
# 4️⃣ SURVIVAL VS ALPHA MATRIX
# ======================================================

st.subheader("4️⃣ Survival vs Alpha Matrix")

sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
vol = returns.std()
mean_ret = returns.mean()

metrics_df = pd.DataFrame({
    "Sharpe Ratio": sharpe,
    "Volatility": vol,
    "Mean Return": mean_ret
})

fig_alpha = px.scatter(
    metrics_df,
    x="Volatility",
    y="Sharpe Ratio",
    size="Mean Return",
    text=metrics_df.index,
    title="Risk vs Return Tradeoff"
)

fig_alpha.update_traces(textposition="top center")
st.plotly_chart(fig_alpha, use_container_width=True)

# ======================================================
# 5️⃣ MAX DRAWDOWN
# ======================================================

st.subheader("5️⃣ Maximum Drawdown")

cum_returns = (1 + returns).cumprod()
rolling_max = cum_returns.cummax()
drawdown = (cum_returns - rolling_max) / rolling_max

fig_dd = px.line(
    drawdown,
    title="Asset Drawdown Profiles"
)

st.plotly_chart(fig_dd, use_container_width=True)

# ======================================================
# 6️⃣ TAIL RISK DIAGNOSTIC
# ======================================================

st.subheader("6️⃣ Tail Risk Diagnostic")

asset_choice = st.selectbox("Select Asset for Tail Analysis", returns.columns)

selected_returns = returns[asset_choice]

fig_hist = px.histogram(
    selected_returns,
    nbins=100,
    title=f"Return Distribution – {asset_choice}"
)

st.plotly_chart(fig_hist, use_container_width=True)

kurtosis_value = selected_returns.kurtosis()

st.metric("Excess Kurtosis", round(kurtosis_value, 2))

if kurtosis_value > 3:
    st.error("Fat Tail Risk Detected")
else:
    st.success("Relatively Thin Tail Distribution")

# ======================================================
# 7️⃣ SYSTEMIC RISK SCORE
# ======================================================

st.subheader("7️⃣ Systemic Risk Score")

risk_score = (
    sharpe.mean() * 0.3
    - vol.mean() * 0.3
    - abs(drawdown.min().mean()) * 0.4
)

st.metric("Composite Survival Score", round(risk_score, 3))

st.markdown("---")

st.markdown("""
### Interpretation Framework

• Rising correlation → diversification collapse  
• Deep drawdowns → capital survival threat  
• High kurtosis → crash probability elevated  
• Low Sharpe + high volatility → alpha illusion  

This page quantifies systemic fragility and portfolio survival capacity.
""")
