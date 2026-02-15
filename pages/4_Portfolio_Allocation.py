import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("Advanced Portfolio Allocation & Capital Preservation Engine")

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
# Sidebar Controls
# ==========================

st.sidebar.header("Portfolio Configuration")

capital = st.sidebar.number_input("Capital", value=100000.0)
risk_aversion = st.sidebar.slider("Risk Aversion (0-1)", 0.0, 1.0, 0.5)
forecast_horizon = st.sidebar.slider("Monte Carlo Horizon (Days)", 30, 252, 90)

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
# Core Mean-Variance Engine
# ==========================

mean_returns = returns.mean()
cov_matrix = returns.cov()

inv_cov = np.linalg.pinv(cov_matrix.values)
weights = inv_cov @ mean_returns.values
weights = weights / np.sum(weights)

# Risk scaling
weights = weights * (1 - risk_aversion)
weights = weights / np.sum(weights)

weights_df = pd.DataFrame(weights, index=selected_assets, columns=["Weight"])

# ==========================
# Portfolio Metrics
# ==========================

port_return = np.dot(weights, mean_returns)
port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
sharpe = port_return / port_vol

st.subheader("Optimal Allocation")
st.dataframe(weights_df)

col1, col2, col3 = st.columns(3)
col1.metric("Expected Return", f"{port_return:.4f}")
col2.metric("Volatility", f"{port_vol:.4f}")
col3.metric("Sharpe Ratio", f"{sharpe:.4f}")

# ==========================
# 1️⃣ Efficient Frontier
# ==========================

st.subheader("Efficient Frontier")

frontier_returns = []
frontier_vol = []

for a in np.linspace(0, 1, 50):
    w = inv_cov @ (mean_returns.values * a)
    w = w / np.sum(w)
    r = np.dot(w, mean_returns)
    v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    frontier_returns.append(r)
    frontier_vol.append(v)

fig_frontier = px.scatter(
    x=frontier_vol,
    y=frontier_returns,
    labels={"x":"Volatility","y":"Return"},
    template="plotly_dark"
)
st.plotly_chart(fig_frontier, use_container_width=True)

# ==========================
# 2️⃣ Risk Contribution
# ==========================

marginal = cov_matrix @ weights
risk_contrib = weights * marginal / port_vol

fig_rc = px.bar(
    x=selected_assets,
    y=risk_contrib,
    title="Risk Contribution",
    template="plotly_dark"
)
st.plotly_chart(fig_rc, use_container_width=True)

# ==========================
# 3️⃣ Cumulative Growth
# ==========================

portfolio_returns = returns @ weights
growth = (1 + portfolio_returns).cumprod()

fig_growth = px.line(
    growth,
    title="Cumulative Growth",
    template="plotly_dark"
)
st.plotly_chart(fig_growth, use_container_width=True)

# ==========================
# 4️⃣ Drawdown
# ==========================

rolling_max = growth.cummax()
drawdown = (growth - rolling_max) / rolling_max

fig_dd = px.line(drawdown, title="Drawdown", template="plotly_dark")
st.plotly_chart(fig_dd, use_container_width=True)

# ==========================
# 5️⃣ Rolling Sharpe
# ==========================

rolling_sharpe = portfolio_returns.rolling(60).mean() / portfolio_returns.rolling(60).std()

fig_rs = px.line(rolling_sharpe, title="Rolling Sharpe", template="plotly_dark")
st.plotly_chart(fig_rs, use_container_width=True)

# ==========================
# 6️⃣ Rolling Volatility
# ==========================

rolling_vol = portfolio_returns.rolling(30).std()

fig_rv = px.line(rolling_vol, title="Rolling Volatility", template="plotly_dark")
st.plotly_chart(fig_rv, use_container_width=True)

# ==========================
# 7️⃣ Correlation Heatmap
# ==========================

fig_corr = px.imshow(
    returns.corr(),
    color_continuous_scale="RdBu",
    template="plotly_dark",
    title="Correlation Matrix"
)
st.plotly_chart(fig_corr, use_container_width=True)

# ==========================
# 8️⃣ Diversification Ratio Gauge
# ==========================

weighted_vol = np.sum(weights * np.sqrt(np.diag(cov_matrix)))
div_ratio = weighted_vol / port_vol

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=div_ratio,
    title={'text':"Diversification Ratio"},
    gauge={'axis': {'range': [0, 3]}}
))

fig_gauge.update_layout(template="plotly_dark")
st.plotly_chart(fig_gauge, use_container_width=True)

# ==========================
# 9️⃣ Monte Carlo Simulation
# ==========================

st.subheader("Monte Carlo Forward Projection")

simulations = 500
sim_results = []

for _ in range(simulations):
    simulated_returns = np.random.normal(
        port_return,
        port_vol,
        forecast_horizon
    )
    sim_path = capital * np.cumprod(1 + simulated_returns)
    sim_results.append(sim_path)

sim_results = np.array(sim_results)

fig_mc = go.Figure()

for i in range(50):
    fig_mc.add_trace(go.Scatter(
        y=sim_results[i],
        mode="lines",
        line=dict(width=1),
        showlegend=False
    ))

fig_mc.update_layout(
    template="plotly_dark",
    title="Monte Carlo Paths"
)

st.plotly_chart(fig_mc, use_container_width=True)

# ==========================
# 🔟 Capital Distribution at Horizon
# ==========================

final_values = sim_results[:, -1]

fig_hist = px.histogram(
    final_values,
    nbins=50,
    title="Distribution of Ending Capital",
    template="plotly_dark"
)

st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("""
### Capital Preservation Diagnostics

• Efficient frontier shows structural risk-return trade-off  
• Risk contribution exposes concentration risk  
• Drawdown confirms survival behavior  
• Diversification ratio measures structural robustness  
• Monte Carlo simulation estimates probabilistic future outcomes  
""")
