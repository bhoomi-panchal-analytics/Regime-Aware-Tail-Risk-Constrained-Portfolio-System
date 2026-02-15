import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("Dynamic Portfolio Allocation & Capital Preservation Engine")

# ==========================
# Load Data
# ==========================

data = load_all()

assets = data.get("market_data", pd.DataFrame())

if assets.empty:
    st.error("Market data missing.")
    st.stop()

assets = assets.apply(pd.to_numeric, errors="coerce").dropna()

returns = assets.pct_change().dropna()

# ==========================
# Investor Profile (Sidebar)
# ==========================

st.sidebar.header("Investor Profile")

capital = st.sidebar.number_input("Capital", value=100000.0)
risk_aversion = st.sidebar.slider("Risk Aversion (0 = Aggressive, 1 = Defensive)", 0.0, 1.0, 0.5)
rebalance_freq = st.sidebar.selectbox("Rebalance Frequency", ["Monthly", "Quarterly", "Yearly"])

# ==========================
# Asset Selection
# ==========================

selected_assets = st.multiselect(
    "Select Assets",
    options=returns.columns,
    default=returns.columns[:4]
)

if len(selected_assets) < 2:
    st.warning("Select at least 2 assets.")
    st.stop()

returns = returns[selected_assets]

# ==========================
# Basic Mean-Variance (Closed Form)
# ==========================

mean_returns = returns.mean()
cov_matrix = returns.cov()

inv_cov = np.linalg.pinv(cov_matrix.values)
ones = np.ones(len(mean_returns))

# Tangency portfolio weights
weights = inv_cov @ mean_returns.values
weights = weights / np.sum(weights)

# Risk scaling based on risk_aversion
weights = weights * (1 - risk_aversion)
weights = weights / np.sum(weights)

weights_df = pd.DataFrame(weights, index=selected_assets, columns=["Weight"])

# ==========================
# Portfolio Metrics
# ==========================

port_return = np.dot(weights, mean_returns)
port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
sharpe = port_return / port_vol

st.subheader("Optimal Weights")
st.dataframe(weights_df)

st.subheader("Portfolio Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Expected Return", f"{port_return:.4f}")
col2.metric("Volatility", f"{port_vol:.4f}")
col3.metric("Sharpe Ratio", f"{sharpe:.4f}")

# ==========================
# Risk Contribution
# ==========================

marginal_risk = cov_matrix @ weights
risk_contribution = weights * marginal_risk / port_vol

risk_df = pd.DataFrame(risk_contribution, index=selected_assets, columns=["Risk Contribution"])

fig_rc = px.bar(risk_df, title="Risk Contribution by Asset", template="plotly_dark")
st.plotly_chart(fig_rc, use_container_width=True)

# ==========================
# Portfolio Growth Simulation
# ==========================

portfolio_returns = returns @ weights
portfolio_growth = (1 + portfolio_returns).cumprod()

fig_growth = px.line(
    portfolio_growth,
    title="Simulated Portfolio Growth",
    template="plotly_dark"
)

st.plotly_chart(fig_growth, use_container_width=True)

# ==========================
# Drawdown Analysis
# ==========================

rolling_max = portfolio_growth.cummax()
drawdown = (portfolio_growth - rolling_max) / rolling_max

fig_dd = px.line(drawdown, title="Drawdown Profile", template="plotly_dark")
st.plotly_chart(fig_dd, use_container_width=True)

# ==========================
# Diversification Ratio
# ==========================

weighted_vol = np.sum(weights * np.sqrt(np.diag(cov_matrix)))
div_ratio = weighted_vol / port_vol

st.subheader("Diversification Ratio")
st.metric("Diversification Ratio", f"{div_ratio:.4f}")

# ==========================
# Correlation Heatmap
# ==========================

fig_corr = px.imshow(
    returns.corr(),
    color_continuous_scale="RdBu",
    title="Correlation Matrix",
    template="plotly_dark"
)

st.plotly_chart(fig_corr, use_container_width=True)

# ==========================
# Rolling Volatility
# ==========================

rolling_vol = portfolio_returns.rolling(30).std()

fig_rollvol = px.line(
    rolling_vol,
    title="Rolling 30-Day Volatility",
    template="plotly_dark"
)

st.plotly_chart(fig_rollvol, use_container_width=True)

# ==========================
# Fundamental Proxy (Return vs Risk Scatter)
# ==========================

scatter_df = pd.DataFrame({
    "Return": mean_returns,
    "Volatility": np.sqrt(np.diag(cov_matrix))
})

fig_scatter = px.scatter(
    scatter_df,
    x="Volatility",
    y="Return",
    text=scatter_df.index,
    title="Risk vs Return Profile",
    template="plotly_dark"
)

st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================
# Interpretation
# ==========================

st.markdown("""
### Capital Preservation Interpretation

• Higher risk_aversion reduces exposure scaling  
• Diversification ratio measures structural robustness  
• Risk contribution highlights concentration risk  
• Drawdown profile evaluates survival capacity  
• Rolling volatility confirms clustering  

This page demonstrates optimization under covariance structure,
not return speculation.
""")
