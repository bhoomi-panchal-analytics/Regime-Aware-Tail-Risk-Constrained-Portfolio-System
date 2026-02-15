import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("Capital Preservation & Tail-Risk Optimized Portfolio Engine")

data = load_all()
assets = data.get("market_data", pd.DataFrame())

if assets.empty:
    st.error("No usable asset data found. Upload ETF price CSVs inside /data folder.")
    st.stop()

assets = assets.apply(pd.to_numeric, errors="coerce").dropna()
returns = assets.pct_change().dropna()

# ------------------------
# User Controls
# ------------------------

st.sidebar.header("Investor Configuration")

capital = st.sidebar.number_input("Capital", value=100000.0)
risk_aversion = st.sidebar.slider("Risk Aversion", 0.0, 1.0, 0.5)
tail_conf = st.sidebar.slider("Tail Confidence Level (%)", 90, 99, 95)
horizon = st.sidebar.slider("Projection Days", 30, 252, 90)

selected_assets = st.multiselect(
    "Select Assets",
    returns.columns,
    default=list(returns.columns[:4])
)

if len(selected_assets) < 2:
    st.warning("Select at least 2 assets.")
    st.stop()

returns = returns[selected_assets]

# ------------------------
# Mean-Variance Engine
# ------------------------

mean_returns = returns.mean()
cov = returns.cov()

inv_cov = np.linalg.pinv(cov.values)
weights = inv_cov @ mean_returns.values
weights = weights / np.sum(weights)

weights = weights * (1 - risk_aversion)
weights = weights / np.sum(weights)

# ------------------------
# Portfolio Metrics
# ------------------------

port_ret = np.dot(weights, mean_returns)
port_vol = np.sqrt(weights.T @ cov.values @ weights)

portfolio_returns = returns @ weights
growth = (1 + portfolio_returns).cumprod()

st.subheader("Optimal Allocation")
st.dataframe(pd.DataFrame(weights, index=selected_assets, columns=["Weight"]))

# ------------------------
# 1. Growth Chart
# ------------------------

fig_growth = px.line(growth, template="plotly_dark", title="Cumulative Growth")
st.plotly_chart(fig_growth, use_container_width=True)

# ------------------------
# 2. Drawdown
# ------------------------

roll_max = growth.cummax()
drawdown = (growth - roll_max) / roll_max

fig_dd = px.line(drawdown, template="plotly_dark", title="Drawdown")
st.plotly_chart(fig_dd, use_container_width=True)

# ------------------------
# 3. Rolling Volatility
# ------------------------

rolling_vol = portfolio_returns.rolling(30).std()
fig_rv = px.line(rolling_vol, template="plotly_dark", title="Rolling Volatility")
st.plotly_chart(fig_rv, use_container_width=True)

# ------------------------
# 4. Tail Risk (VaR & CVaR)
# ------------------------

alpha = 1 - tail_conf/100

VaR = np.quantile(portfolio_returns, alpha)
CVaR = portfolio_returns[portfolio_returns <= VaR].mean()

st.subheader("Tail Risk Metrics")
col1, col2 = st.columns(2)
col1.metric("VaR", f"{VaR:.4f}")
col2.metric("CVaR", f"{CVaR:.4f}")

# ------------------------
# 5. Fat vs Thin Tail Analysis
# ------------------------

mean = portfolio_returns.mean()
std = portfolio_returns.std()

skew = ((portfolio_returns - mean)**3).mean() / std**3
kurtosis = ((portfolio_returns - mean)**4).mean() / std**4

st.subheader("Distribution Shape")
col3, col4 = st.columns(2)
col3.metric("Skewness", f"{skew:.4f}")
col4.metric("Kurtosis", f"{kurtosis:.4f}")

fig_hist = px.histogram(portfolio_returns, nbins=60,
                        template="plotly_dark",
                        title="Return Distribution")
st.plotly_chart(fig_hist, use_container_width=True)

# ------------------------
# 6. Q-Q Plot (Fat Tail Check)
# ------------------------

theoretical = np.sort(np.random.normal(mean, std, len(portfolio_returns)))
empirical = np.sort(portfolio_returns)

fig_qq = go.Figure()
fig_qq.add_trace(go.Scatter(
    x=theoretical,
    y=empirical,
    mode="markers"
))
fig_qq.update_layout(template="plotly_dark",
                     title="Q-Q Plot (Normal vs Empirical)")
st.plotly_chart(fig_qq, use_container_width=True)

# ------------------------
# 7. Monte Carlo Simulation
# ------------------------

simulations = 300
paths = []

for _ in range(simulations):
    sim_returns = np.random.normal(port_ret, port_vol, horizon)
    sim_path = capital * np.cumprod(1 + sim_returns)
    paths.append(sim_path)

paths = np.array(paths)

fig_mc = go.Figure()
for i in range(50):
    fig_mc.add_trace(go.Scatter(y=paths[i], mode="lines",
                                line=dict(width=1), showlegend=False))
fig_mc.update_layout(template="plotly_dark",
                     title="Monte Carlo Projection")
st.plotly_chart(fig_mc, use_container_width=True)

# ------------------------
# 8. Ending Capital Distribution
# ------------------------

final_vals = paths[:, -1]

fig_end = px.histogram(final_vals, nbins=50,
                       template="plotly_dark",
                       title="Ending Capital Distribution")
st.plotly_chart(fig_end, use_container_width=True)

# ------------------------
# 9. Correlation Heatmap
# ------------------------

fig_corr = px.imshow(returns.corr(),
                     color_continuous_scale="RdBu",
                     template="plotly_dark",
                     title="Correlation Matrix")
st.plotly_chart(fig_corr, use_container_width=True)

# ------------------------
# 10. Risk Contribution
# ------------------------

marginal = cov @ weights
risk_contrib = weights * marginal / port_vol

fig_rc = px.bar(x=selected_assets, y=risk_contrib,
                template="plotly_dark",
                title="Risk Contribution")
st.plotly_chart(fig_rc, use_container_width=True)

st.markdown("""
### Interpretation

• VaR and CVaR quantify left-tail exposure  
• Skewness indicates asymmetry  
• Kurtosis > 3 implies fat tails  
• Q-Q deviation confirms non-normality  
• Monte Carlo projects survival range  
• Risk contribution exposes concentration  
""")
