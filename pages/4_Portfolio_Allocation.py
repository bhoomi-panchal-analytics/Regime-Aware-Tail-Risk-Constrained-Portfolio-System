import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("Portfolio Allocation & Tail Risk Diagnostics")

# ==========================
# Safe Data Loader
# ==========================

data = load_all()

assets = data.get("market_data", pd.DataFrame())

if assets.empty:
    st.warning("market_data not found. Attempting ETF fallback.")
    assets = data.get("etf_prices", pd.DataFrame())

if assets.empty:
    st.error("No usable asset data found. Upload ETF price CSVs inside /data folder.")
    st.stop()

assets = assets.apply(pd.to_numeric, errors="coerce").dropna()
returns = assets.pct_change().dropna()

# ==========================
# Sidebar Controls
# ==========================

st.sidebar.header("Investor Controls")

capital = st.sidebar.number_input("Capital", value=100000.0)
risk_aversion = st.sidebar.slider("Risk Aversion", 0.0, 1.0, 0.5)
tail_conf = st.sidebar.slider("Tail Confidence Level (%)", 90, 99, 95)

selected_assets = st.multiselect(
    "Select Assets",
    returns.columns,
    default=list(returns.columns[:4])
)

if len(selected_assets) < 2:
    st.warning("Select at least 2 assets.")
    st.stop()

returns = returns[selected_assets]

# ==========================
# Mean-Variance Allocation
# ==========================

mean_returns = returns.mean()
cov_matrix = returns.cov()

inv_cov = np.linalg.pinv(cov_matrix.values)
weights = inv_cov @ mean_returns.values
weights = weights / np.sum(weights)

weights = weights * (1 - risk_aversion)
weights = weights / np.sum(weights)

weights_df = pd.DataFrame(weights, index=selected_assets, columns=["Weight"])

st.subheader("Optimal Allocation")
st.dataframe(weights_df)

# ==========================
# Portfolio Performance
# ==========================

portfolio_returns = returns @ weights
growth = (1 + portfolio_returns).cumprod()

fig_growth = px.line(growth, template="plotly_dark", title="Cumulative Growth")
st.plotly_chart(fig_growth, use_container_width=True)

# ==========================
# Drawdown
# ==========================

rolling_max = growth.cummax()
drawdown = (growth - rolling_max) / rolling_max

fig_dd = px.line(drawdown, template="plotly_dark", title="Drawdown")
st.plotly_chart(fig_dd, use_container_width=True)

# ==========================
# Tail Risk Metrics
# ==========================

confidence = tail_conf / 100
var_level = np.percentile(portfolio_returns, (1 - confidence) * 100)
cvar = portfolio_returns[portfolio_returns <= var_level].mean()

col1, col2 = st.columns(2)
col1.metric("VaR", f"{var_level:.4f}")
col2.metric("CVaR", f"{cvar:.4f}")

# ==========================
# Fat Tail Visualization
# ==========================

st.subheader("Return Distribution & Fat Tail Check")

hist = px.histogram(
    portfolio_returns,
    nbins=50,
    template="plotly_dark",
    title="Return Distribution"
)

st.plotly_chart(hist, use_container_width=True)

# ==========================
# Kurtosis & Skew (No SciPy)
# ==========================

mean_r = portfolio_returns.mean()
std_r = portfolio_returns.std()

skew = ((portfolio_returns - mean_r) ** 3).mean() / (std_r ** 3)
kurt = ((portfolio_returns - mean_r) ** 4).mean() / (std_r ** 4)

col3, col4 = st.columns(2)
col3.metric("Skewness", f"{skew:.4f}")
col4.metric("Kurtosis", f"{kurt:.4f}")

# ==========================
# Normal vs Empirical Comparison
# ==========================

st.subheader("Normal vs Empirical Tail Comparison")

x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 1000)
normal_pdf = (1 / (std_r * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_r)/std_r)**2)

fig_pdf = go.Figure()

fig_pdf.add_trace(go.Scatter(
    x=x,
    y=normal_pdf,
    name="Normal Distribution"
))

fig_pdf.add_trace(go.Histogram(
    x=portfolio_returns,
    histnorm="probability density",
    name="Empirical",
    opacity=0.5
))

fig_pdf.update_layout(template="plotly_dark")
st.plotly_chart(fig_pdf, use_container_width=True)

# ==========================
# Rolling Tail Risk
# ==========================

st.subheader("Rolling CVaR")

rolling_cvar = portfolio_returns.rolling(60).apply(
    lambda x: x[x <= np.percentile(x, 5)].mean()
)

fig_roll = px.line(rolling_cvar, template="plotly_dark")
st.plotly_chart(fig_roll, use_container_width=True)

# ==========================
# Diversification
# ==========================

weighted_vol = np.sum(weights * np.sqrt(np.diag(cov_matrix)))
port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
div_ratio = weighted_vol / port_vol

st.subheader("Diversification Ratio")
st.metric("Diversification Ratio", f"{div_ratio:.4f}")

# ==========================
# Correlation Heatmap
# ==========================

fig_corr = px.imshow(
    returns.corr(),
    color_continuous_scale="RdBu",
    template="plotly_dark",
    title="Correlation Matrix"
)

st.plotly_chart(fig_corr, use_container_width=True)

# ==========================
# Interpretation
# ==========================

st.markdown("""
### Capital Preservation Interpretation

• Kurtosis > 3 → Fat tails present  
• Negative skew → Downside asymmetry  
• CVaR captures expected extreme loss  
• Diversification ratio measures structural robustness  
• Rolling CVaR tracks stress regime shifts  

Tail risk, not variance, defines survival.
""")
