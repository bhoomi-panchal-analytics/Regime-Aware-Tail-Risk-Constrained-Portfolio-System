import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Simple Portfolio Allocation & Risk Diagnostics")

# =========================
# LOAD DATA
# =========================

from utils.load_data import load_all
data = load_all()

if "market_data_template" not in data:
    st.error("market_data_template.csv not found in /data.")
    st.stop()

df = data["market_data_template"].copy()

# =========================
# CLEAN DATA
# =========================

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date")

df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna(axis=1, how="all")
df = df.fillna(0)

if df.shape[1] < 2:
    st.error("Not enough asset columns available.")
    st.stop()

st.success("Market data loaded.")

# =========================
# USER INPUT WEIGHTS
# =========================

st.subheader("Portfolio Weight Allocation")

assets = df.columns.tolist()
weights = []

for asset in assets:
    w = st.slider(asset, 0.0, 1.0, 1.0 / len(assets), 0.01)
    weights.append(w)

weights = np.array(weights)

if weights.sum() == 0:
    st.error("Total weight cannot be zero.")
    st.stop()

weights = weights / weights.sum()

st.write("Normalized Weights:", np.round(weights, 3))

# =========================
# PORTFOLIO RETURNS
# =========================

portfolio_returns = df @ weights

# =========================
# CUMULATIVE PERFORMANCE
# =========================

st.subheader("Cumulative Growth of $1")

cum = (1 + portfolio_returns).cumprod()

fig1 = px.line(cum, template="plotly_dark")
st.plotly_chart(fig1, use_container_width=True)

# =========================
# RISK METRICS
# =========================

mean_return = portfolio_returns.mean()
volatility = portfolio_returns.std()
sharpe = mean_return / volatility if volatility != 0 else 0
max_drawdown = (cum / cum.cummax() - 1).min()

st.subheader("Risk Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean Return", round(mean_return, 4))
col2.metric("Volatility", round(volatility, 4))
col3.metric("Sharpe Ratio", round(sharpe, 3))
col4.metric("Max Drawdown", round(max_drawdown, 3))

# =========================
# RETURN DISTRIBUTION
# =========================

st.subheader("Return Distribution")

fig2 = px.histogram(portfolio_returns, nbins=50, template="plotly_dark")
st.plotly_chart(fig2, use_container_width=True)

# =========================
# TAIL RISK (5% VAR)
# =========================

var_5 = np.percentile(portfolio_returns, 5)

st.subheader("Tail Risk (5% VaR)")
st.metric("Value at Risk (5%)", round(var_5, 4))

# =========================
# CORRELATION HEATMAP
# =========================

st.subheader("Asset Correlation Matrix")

corr = df.corr()

fig3 = px.imshow(
    corr,
    zmin=-1,
    zmax=1,
    color_continuous_scale="RdBu",
    template="plotly_dark"
)

st.plotly_chart(fig3, use_container_width=True)

# =========================
# VOL CONTRIBUTION
# =========================

st.subheader("Volatility Contribution by Asset")

cov = df.cov()
port_var = weights.T @ cov.values @ weights
marginal_contrib = cov.values @ weights
risk_contrib = weights * marginal_contrib / port_var

fig4 = px.bar(
    x=assets,
    y=risk_contrib,
    template="plotly_dark"
)

st.plotly_chart(fig4, use_container_width=True)

# =========================
# INTERPRETATION
# =========================

st.markdown("""
### Interpretation

• High Sharpe → efficient risk-adjusted performance  
• Large drawdown → capital fragility  
• High VaR magnitude → fat left tail  
• Uneven risk contribution → concentration risk  

True diversification means balanced risk contribution — not equal weights.
""")
