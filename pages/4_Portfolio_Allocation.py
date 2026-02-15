import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("Portfolio Allocation & Tail Risk Engine")

# ==========================
# LOAD DATA (ROBUST)
# ==========================

data = load_all()

assets = None
max_cols = 0

for key in data:
    df = data[key]
    if isinstance(df, pd.DataFrame) and df.shape[1] > max_cols:
        assets = df
        max_cols = df.shape[1]

if assets is None:
    st.error("No suitable market dataset found.")
    st.stop()

assets = assets.apply(pd.to_numeric, errors="coerce").dropna()

# ==========================
# TIMELINE FILTER
# ==========================

min_date = assets.index.min()
max_date = assets.index.max()

if pd.isna(min_date) or pd.isna(max_date):
    st.error("Invalid date index.")
    st.stop()

start_date = st.date_input("Start Date", min_date.date())
end_date = st.date_input("End Date", max_date.date())

end_date = st.date_input("End Date", assets.index.max())

assets = assets.loc[
    (assets.index >= pd.to_datetime(start_date)) &
    (assets.index <= pd.to_datetime(end_date))
]

returns = assets.pct_change().dropna()

# ==========================
# SIDEBAR CONTROLS
# ==========================

st.sidebar.header("Investor Configuration")

capital = st.sidebar.number_input("Capital", value=100000.0)
risk_aversion = st.sidebar.slider("Risk Aversion", 0.0, 1.0, 0.5)
confidence_level = st.sidebar.slider("VaR Confidence", 0.90, 0.99, 0.95)

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
weights /= np.sum(weights)

weights *= (1 - risk_aversion)
weights /= np.sum(weights)

portfolio_returns = returns @ weights

# ==========================
# BASIC METRICS
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
# 1️⃣ WEIGHTS PIE
# ==========================

fig_weights = px.pie(
    names=selected_assets,
    values=weights,
    title="Portfolio Weights",
    template="plotly_dark"
)
st.plotly_chart(fig_weights, use_container_width=True)

# ==========================
# 2️⃣ CUMULATIVE GROWTH
# ==========================

growth = (1 + portfolio_returns).cumprod()

fig_growth = px.line(growth, template="plotly_dark",
                     title="Cumulative Growth")
st.plotly_chart(fig_growth, use_container_width=True)

# ==========================
# 3️⃣ DRAWDOWN
# ==========================

rolling_max = growth.cummax()
drawdown = (growth - rolling_max) / rolling_max

fig_dd = px.line(drawdown, template="plotly_dark",
                 title="Drawdown Profile")
st.plotly_chart(fig_dd, use_container_width=True)

# ==========================
# 4️⃣ ROLLING VOL
# ==========================

rolling_vol = portfolio_returns.rolling(30).std()

fig_vol = px.line(rolling_vol, template="plotly_dark",
                  title="Rolling Volatility")
st.plotly_chart(fig_vol, use_container_width=True)

# ==========================
# 5️⃣ CORRELATION HEATMAP
# ==========================

fig_corr = px.imshow(
    returns.corr(),
    template="plotly_dark",
    color_continuous_scale="RdBu",
    title="Correlation Matrix"
)
st.plotly_chart(fig_corr, use_container_width=True)

# ==========================
# 6️⃣ TAIL RISK (VaR & CVaR)
# ==========================

var = np.percentile(portfolio_returns, (1-confidence_level)*100)
cvar = portfolio_returns[portfolio_returns <= var].mean()

st.subheader("Tail Risk Metrics")

col4, col5 = st.columns(2)
col4.metric("Value at Risk", f"{var:.4f}")
col5.metric("Conditional VaR", f"{cvar:.4f}")

# ==========================
# 7️⃣ DISTRIBUTION
# ==========================

fig_hist = px.histogram(
    portfolio_returns,
    nbins=50,
    template="plotly_dark",
    title="Return Distribution"
)
st.plotly_chart(fig_hist, use_container_width=True)

# ==========================
# 8️⃣ KURTOSIS & SKEW (MANUAL)
# ==========================

mean_r = np.mean(portfolio_returns)
std_r = np.std(portfolio_returns)

skewness = np.mean(((portfolio_returns - mean_r)/std_r)**3)
kurt = np.mean(((portfolio_returns - mean_r)/std_r)**4)

st.subheader("Distribution Shape")

col6, col7 = st.columns(2)
col6.metric("Skewness", f"{skewness:.4f}")
col7.metric("Kurtosis", f"{kurt:.4f}")

# ==========================
# 9️⃣ FAT VS THIN TAIL
# ==========================

x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 100)
normal_pdf = (1/(std_r*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean_r)/std_r)**2)

fig_tail = go.Figure()
fig_tail.add_trace(go.Scatter(x=x, y=normal_pdf, name="Normal Curve"))
fig_tail.update_layout(template="plotly_dark",
                       title="Normal Distribution Overlay")
st.plotly_chart(fig_tail, use_container_width=True)

# ==========================
# 🔟 MONTE CARLO
# ==========================

st.subheader("Monte Carlo Simulation")

simulations = 200
horizon = 60
paths = []

for _ in range(simulations):
    sim = np.random.normal(port_return, port_vol, horizon)
    paths.append(capital * np.cumprod(1 + sim))

paths = np.array(paths)

fig_mc = go.Figure()
for i in range(30):
    fig_mc.add_trace(go.Scatter(y=paths[i],
                                mode="lines",
                                showlegend=False))

fig_mc.update_layout(template="plotly_dark",
                     title="Monte Carlo Paths")
st.plotly_chart(fig_mc, use_container_width=True)

# ==========================
# 11️⃣ ENDING CAPITAL HISTOGRAM
# ==========================

fig_end = px.histogram(paths[:, -1],
                       nbins=40,
                       template="plotly_dark",
                       title="Ending Capital Distribution")

st.plotly_chart(fig_end, use_container_width=True)

st.markdown("""
### Interpretation

• Kurtosis > 3 suggests fat tails  
• CVaR captures extreme downside risk  
• Monte Carlo shows probabilistic capital paths  
• Diversification reduces but does not eliminate tail exposure  
""")
