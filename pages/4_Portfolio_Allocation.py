import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("Portfolio Allocation & Tail Risk Engine")

# ==========================
# Load Data Safely
# ==========================

data = load_all()

assets = data.get("market_data")

# Fallback if market_data missing
if assets is None or assets.empty:
    st.warning("market_data not found. Attempting fallback to ETF prices.")
    possible = ["SPY", "TLT", "GLD", "DBC", "UUP", "SHY"]
    fallback = {}
    for k in possible:
        if k in data:
            fallback[k] = data[k]
    if fallback:
        assets = pd.DataFrame(fallback)
    else:
        st.error("No usable asset data found.")
        st.stop()

assets = assets.apply(pd.to_numeric, errors="coerce").dropna()
returns = assets.pct_change().dropna()

# ==========================
# Sidebar Controls
# ==========================

st.sidebar.header("Portfolio Settings")

capital = st.sidebar.number_input("Initial Capital", value=100000.0)
risk_aversion = st.sidebar.slider("Risk Aversion", 0.0, 1.0, 0.5)
confidence_level = st.sidebar.slider("VaR Confidence Level", 0.90, 0.99, 0.95)

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
# Core Mean-Variance
# ==========================

mean_returns = returns.mean()
cov_matrix = returns.cov()

inv_cov = np.linalg.pinv(cov_matrix.values)
weights = inv_cov @ mean_returns.values
weights = weights / np.sum(weights)

weights = weights * (1 - risk_aversion)
weights = weights / np.sum(weights)

weights_df = pd.DataFrame(weights, index=selected_assets, columns=["Weight"])

st.subheader("Optimal Weights")
st.dataframe(weights_df)

portfolio_returns = returns @ weights
portfolio_growth = capital * (1 + portfolio_returns).cumprod()

# ==========================
# Basic Metrics
# ==========================

port_return = portfolio_returns.mean()
port_vol = portfolio_returns.std()
sharpe = port_return / port_vol

col1, col2, col3 = st.columns(3)
col1.metric("Expected Return", f"{port_return:.4f}")
col2.metric("Volatility", f"{port_vol:.4f}")
col3.metric("Sharpe Ratio", f"{sharpe:.4f}")

# ==========================
# Cumulative Growth
# ==========================

fig_growth = px.line(
    portfolio_growth,
    title="Cumulative Portfolio Growth",
    template="plotly_dark"
)
st.plotly_chart(fig_growth, use_container_width=True)

# ==========================
# Drawdown
# ==========================

rolling_max = portfolio_growth.cummax()
drawdown = (portfolio_growth - rolling_max) / rolling_max

fig_dd = px.line(drawdown, title="Drawdown Profile", template="plotly_dark")
st.plotly_chart(fig_dd, use_container_width=True)

# ==========================
# Tail Risk Analysis
# ==========================

st.subheader("Tail Risk Analysis")

# Historical VaR
var = np.quantile(portfolio_returns, 1 - confidence_level)

# CVaR
cvar = portfolio_returns[portfolio_returns <= var].mean()

col1, col2 = st.columns(2)
col1.metric("Value at Risk", f"{var:.4f}")
col2.metric("Conditional VaR", f"{cvar:.4f}")

# ==========================
# Fat vs Thin Tail Diagnostics
# ==========================

st.subheader("Distribution & Tail Diagnostics")

# Skewness
skewness = portfolio_returns.skew()

# Kurtosis
kurt = portfolio_returns.kurtosis()

col1, col2 = st.columns(2)
col1.metric("Skewness", f"{skewness:.4f}")
col2.metric("Kurtosis", f"{kurt:.4f}")

fig_hist = px.histogram(
    portfolio_returns,
    nbins=60,
    title="Return Distribution",
    template="plotly_dark"
)
st.plotly_chart(fig_hist, use_container_width=True)

# ==========================
# QQ-Style Comparison
# ==========================

sorted_returns = np.sort(portfolio_returns)
normal_sample = np.sort(np.random.normal(port_return, port_vol, len(portfolio_returns)))

fig_qq = go.Figure()

fig_qq.add_trace(go.Scatter(
    x=normal_sample,
    y=sorted_returns,
    mode="markers",
    name="Empirical vs Normal"
))

fig_qq.update_layout(
    title="QQ-Style Tail Comparison",
    template="plotly_dark"
)

st.plotly_chart(fig_qq, use_container_width=True)

# ==========================
# Rolling CVaR
# ==========================

rolling_cvar = portfolio_returns.rolling(60).apply(
    lambda x: np.mean(x[x <= np.quantile(x, 1 - confidence_level)])
)

fig_rcvar = px.line(
    rolling_cvar,
    title="Rolling CVaR (60-day)",
    template="plotly_dark"
)

st.plotly_chart(fig_rcvar, use_container_width=True)

# ==========================
# Diversification Ratio
# ==========================

weighted_vol = np.sum(weights * np.sqrt(np.diag(cov_matrix)))
div_ratio = weighted_vol / port_vol

st.subheader("Diversification Strength")
st.metric("Diversification Ratio", f"{div_ratio:.4f}")

# ==========================
# Risk Contribution
# ==========================

marginal = cov_matrix @ weights
risk_contrib = weights * marginal / port_vol

fig_rc = px.bar(
    x=selected_assets,
    y=risk_contrib,
    title="Risk Contribution by Asset",
    template="plotly_dark"
)
st.plotly_chart(fig_rc, use_container_width=True)

st.markdown("""
### Interpretation

• VaR and CVaR quantify downside exposure  
• Kurtosis > 3 indicates fat tails  
• Skewness reveals asymmetry  
• Rolling CVaR tracks regime-dependent tail risk  
• Diversification ratio measures structural robustness  
""")
