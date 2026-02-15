import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.load_data import load_all
from scipy.stats import kurtosis, skew, t

st.set_page_config(layout="wide")
st.title("Advanced Portfolio Allocation & Tail Risk Engine")

# ==========================
# LOAD DATA (ROBUST)
# ==========================

data = load_all()

# Auto-detect market-like dataset
assets = None
for key in data:
    df = data[key]
    if isinstance(df, pd.DataFrame) and df.shape[1] >= 3:
        assets = df
        break

if assets is None:
    st.error("No suitable market dataset found.")
    st.stop()

assets = assets.apply(pd.to_numeric, errors="coerce").dropna()

# ==========================
# Timeline Selection
# ==========================

start_date = st.date_input("Start Date", assets.index.min())
end_date = st.date_input("End Date", assets.index.max())

assets = assets.loc[
    (assets.index >= pd.to_datetime(start_date)) &
    (assets.index <= pd.to_datetime(end_date))
]

returns = assets.pct_change().dropna()

# ==========================
# Sidebar Controls
# ==========================

st.sidebar.header("Investor Configuration")

capital = st.sidebar.number_input("Capital", value=100000.0)
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
# MEAN VARIANCE CORE
# ==========================

mean_returns = returns.mean()
cov_matrix = returns.cov()

inv_cov = np.linalg.pinv(cov_matrix.values)
weights = inv_cov @ mean_returns.values
weights = weights / np.sum(weights)
weights *= (1 - risk_aversion)
weights /= np.sum(weights)

portfolio_returns = returns @ weights

# ==========================
# METRICS
# ==========================

port_return = np.dot(weights, mean_returns)
port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
sharpe = port_return / port_vol

st.subheader("Portfolio Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Expected Return", f"{port_return:.4f}")
col2.metric("Volatility", f"{port_vol:.4f}")
col3.metric("Sharpe", f"{sharpe:.4f}")

# ==========================
# 1️⃣ Weight Distribution
# ==========================

fig_weights = px.pie(
    names=selected_assets,
    values=weights,
    title="Portfolio Weights",
    template="plotly_dark"
)
st.plotly_chart(fig_weights, use_container_width=True)

# ==========================
# 2️⃣ Cumulative Growth
# ==========================

growth = (1 + portfolio_returns).cumprod()
fig_growth = px.line(growth, template="plotly_dark", title="Cumulative Growth")
st.plotly_chart(fig_growth, use_container_width=True)

# ==========================
# 3️⃣ Drawdown
# ==========================

rolling_max = growth.cummax()
drawdown = (growth - rolling_max) / rolling_max
fig_dd = px.line(drawdown, template="plotly_dark", title="Drawdown")
st.plotly_chart(fig_dd, use_container_width=True)

# ==========================
# 4️⃣ Rolling Volatility
# ==========================

rolling_vol = portfolio_returns.rolling(30).std()
fig_vol = px.line(rolling_vol, template="plotly_dark", title="Rolling Volatility")
st.plotly_chart(fig_vol, use_container_width=True)

# ==========================
# 5️⃣ Rolling Sharpe
# ==========================

rolling_sharpe = portfolio_returns.rolling(60).mean() / portfolio_returns.rolling(60).std()
fig_rs = px.line(rolling_sharpe, template="plotly_dark", title="Rolling Sharpe")
st.plotly_chart(fig_rs, use_container_width=True)

# ==========================
# 6️⃣ Correlation Heatmap
# ==========================

fig_corr = px.imshow(
    returns.corr(),
    template="plotly_dark",
    color_continuous_scale="RdBu",
    title="Correlation Matrix"
)
st.plotly_chart(fig_corr, use_container_width=True)

# ==========================
# 7️⃣ Tail Risk — VaR & CVaR
# ==========================

var = np.percentile(portfolio_returns, (1-confidence_level)*100)
cvar = portfolio_returns[portfolio_returns <= var].mean()

st.subheader("Tail Risk Metrics")
col4, col5 = st.columns(2)
col4.metric("Value at Risk", f"{var:.4f}")
col5.metric("Conditional VaR", f"{cvar:.4f}")

# ==========================
# 8️⃣ Return Distribution
# ==========================

fig_hist = px.histogram(
    portfolio_returns,
    nbins=50,
    template="plotly_dark",
    title="Return Distribution"
)
st.plotly_chart(fig_hist, use_container_width=True)

# ==========================
# 9️⃣ Kurtosis & Skew
# ==========================

kurt = kurtosis(portfolio_returns)
skw = skew(portfolio_returns)

st.subheader("Distribution Shape")
col6, col7 = st.columns(2)
col6.metric("Kurtosis (Tail Fatness)", f"{kurt:.4f}")
col7.metric("Skewness", f"{skw:.4f}")

# ==========================
# 🔟 Normal vs Fat Tail Comparison
# ==========================

x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 100)
normal_pdf = (1/(port_vol*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-port_return)/port_vol)**2)
t_pdf = t.pdf((x-port_return)/port_vol, df=4)

fig_tail = go.Figure()
fig_tail.add_trace(go.Scatter(x=x, y=normal_pdf, name="Normal"))
fig_tail.add_trace(go.Scatter(x=x, y=t_pdf, name="Fat Tail (t-dist)"))

fig_tail.update_layout(template="plotly_dark", title="Thin vs Fat Tail Comparison")
st.plotly_chart(fig_tail, use_container_width=True)

# ==========================
# 11️⃣ Monte Carlo Simulation
# ==========================

st.subheader("Monte Carlo Projection")

simulations = 300
horizon = 60
paths = []

for _ in range(simulations):
    sim = np.random.normal(port_return, port_vol, horizon)
    paths.append(capital * np.cumprod(1 + sim))

paths = np.array(paths)

fig_mc = go.Figure()
for i in range(50):
    fig_mc.add_trace(go.Scatter(y=paths[i], mode="lines", showlegend=False))

fig_mc.update_layout(template="plotly_dark", title="Monte Carlo Paths")
st.plotly_chart(fig_mc, use_container_width=True)

# ==========================
# 12️⃣ Ending Capital Distribution
# ==========================

fig_end = px.histogram(paths[:, -1], nbins=40,
                       template="plotly_dark",
                       title="Distribution of Ending Capital")
st.plotly_chart(fig_end, use_container_width=True)

st.markdown("""
### Capital Preservation Insights

• VaR and CVaR quantify downside risk  
• Kurtosis measures tail heaviness  
• Fat-tail overlay reveals non-Gaussian risk  
• Monte Carlo projects probabilistic capital outcomes  
• Diversification reduces but does not eliminate tail exposure  
""")
