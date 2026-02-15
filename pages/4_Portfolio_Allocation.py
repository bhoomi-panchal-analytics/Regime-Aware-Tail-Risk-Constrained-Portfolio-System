import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide")
st.title("Regime-Aware Portfolio Allocation & Tail Risk Engine")

# ==========================
# DATA LOADING (ROBUST)
# ==========================

def load_asset_data():
    possible_files = [
        "data/market_data.csv",
        "data/SPY.csv",
        "data/TLT.csv",
        "data/GLD.csv",
        "data/DBC.csv",
        "data/UUP.csv",
        "data/SHY.csv"
    ]
    
    price_data = {}

    for file in possible_files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file, index_col=0, parse_dates=True)
                if "Adj Close" in df.columns:
                    price_data[file.split("/")[-1].split(".")[0]] = df["Adj Close"]
                elif df.shape[1] == 1:
                    price_data[file.split("/")[-1].split(".")[0]] = df.iloc[:, 0]
            except:
                continue

    if len(price_data) == 0:
        return pd.DataFrame()

    return pd.DataFrame(price_data).dropna()

assets = load_asset_data()

if assets.empty:
    st.error("No usable asset data found. Upload ETF price CSVs inside /data folder.")
    st.stop()

returns = assets.pct_change().dropna()

# ==========================
# USER CONTROLS
# ==========================

st.sidebar.header("Portfolio Configuration")

capital = st.sidebar.number_input("Capital", value=100000.0)
risk_aversion = st.sidebar.slider("Risk Aversion", 0.0, 1.0, 0.5)
confidence_level = st.sidebar.slider("Tail Risk Confidence (CVaR)", 0.90, 0.99, 0.95)

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
# MEAN-VARIANCE CORE
# ==========================

mean_returns = returns.mean()
cov_matrix = returns.cov()

inv_cov = np.linalg.pinv(cov_matrix.values)
weights = inv_cov @ mean_returns.values
weights = weights / np.sum(weights)

weights *= (1 - risk_aversion)
weights = weights / np.sum(weights)

weights_df = pd.DataFrame(weights, index=selected_assets, columns=["Weight"])
st.subheader("Optimal Weights")
st.dataframe(weights_df)

portfolio_returns = returns @ weights

# ==========================
# METRICS
# ==========================

port_return = portfolio_returns.mean()
port_vol = portfolio_returns.std()
sharpe = port_return / port_vol

col1, col2, col3 = st.columns(3)
col1.metric("Expected Return", f"{port_return:.4f}")
col2.metric("Volatility", f"{port_vol:.4f}")
col3.metric("Sharpe Ratio", f"{sharpe:.4f}")

# ==========================
# TAIL RISK METRICS
# ==========================

var_level = np.quantile(portfolio_returns, 1 - confidence_level)
cvar = portfolio_returns[portfolio_returns <= var_level].mean()

skewness = portfolio_returns.skew()
kurt = portfolio_returns.kurt()

st.subheader("Tail Risk Diagnostics")

col4, col5, col6 = st.columns(3)
col4.metric("VaR", f"{var_level:.4f}")
col5.metric("CVaR", f"{cvar:.4f}")
col6.metric("Kurtosis", f"{kurt:.4f}")

# ==========================
# FAT vs THIN TAIL CHART
# ==========================

st.subheader("Return Distribution vs Normal")

x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 100)
normal_dist = (1/(port_vol*np.sqrt(2*np.pi))) * np.exp(-(x-port_return)**2/(2*port_vol**2))

fig_tail = go.Figure()

fig_tail.add_trace(go.Histogram(
    x=portfolio_returns,
    histnorm='probability density',
    name="Actual Returns",
    opacity=0.6
))

fig_tail.add_trace(go.Scatter(
    x=x,
    y=normal_dist,
    mode="lines",
    name="Normal Distribution"
))

fig_tail.update_layout(template="plotly_dark")
st.plotly_chart(fig_tail, use_container_width=True)

# ==========================
# CUMULATIVE GROWTH
# ==========================

growth = (1 + portfolio_returns).cumprod()

fig_growth = px.line(growth, title="Cumulative Growth", template="plotly_dark")
st.plotly_chart(fig_growth, use_container_width=True)

# ==========================
# DRAWDOWN
# ==========================

rolling_max = growth.cummax()
drawdown = (growth - rolling_max) / rolling_max

fig_dd = px.line(drawdown, title="Drawdown Curve", template="plotly_dark")
st.plotly_chart(fig_dd, use_container_width=True)

# ==========================
# ROLLING VOLATILITY
# ==========================

rolling_vol = portfolio_returns.rolling(30).std()

fig_rv = px.line(rolling_vol, title="Rolling Volatility", template="plotly_dark")
st.plotly_chart(fig_rv, use_container_width=True)

# ==========================
# CORRELATION HEATMAP
# ==========================

fig_corr = px.imshow(
    returns.corr(),
    color_continuous_scale="RdBu",
    template="plotly_dark",
    title="Correlation Matrix"
)

st.plotly_chart(fig_corr, use_container_width=True)

# ==========================
# MONTE CARLO SIMULATION
# ==========================

st.subheader("Monte Carlo Projection")

simulations = 300
horizon = 90
sim_paths = []

for _ in range(simulations):
    sim_returns = np.random.normal(port_return, port_vol, horizon)
    sim_path = capital * np.cumprod(1 + sim_returns)
    sim_paths.append(sim_path)

sim_paths = np.array(sim_paths)

fig_mc = go.Figure()

for i in range(30):
    fig_mc.add_trace(go.Scatter(
        y=sim_paths[i],
        mode='lines',
        showlegend=False
    ))

fig_mc.update_layout(template="plotly_dark")
st.plotly_chart(fig_mc, use_container_width=True)

st.markdown("""
### Interpretation

• Kurtosis > 3 implies fat tails  
• CVaR captures expected extreme loss  
• Distribution vs normal reveals tail thickness  
• Drawdown measures survival capacity  
• Monte Carlo shows probabilistic capital evolution  

This engine emphasizes capital preservation, not naive return maximization.
""")
