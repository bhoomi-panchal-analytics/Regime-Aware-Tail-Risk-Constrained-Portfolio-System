import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.load_data import load_all
from scipy.optimize import minimize

st.set_page_config(layout="wide")
st.title("Portfolio Allocation & Capital Preservation Engine")

# ====================================
# LOAD DATA
# ====================================

data = load_all()

if "market" not in data or data["macro"].empty:
    st.warning("Market return data missing.")
    st.stop()

# For demo assume you have market_prices.csv with asset returns
try:
    market = pd.read_csv("data/market_prices.csv", index_col=0, parse_dates=True)
except:
    st.warning("market_prices.csv missing.")
    st.stop()

returns = market.pct_change().dropna()

assets = list(returns.columns)

selected_assets = st.multiselect(
    "Select Assets",
    assets,
    default=assets[:4]
)

if len(selected_assets) < 2:
    st.warning("Select at least two assets.")
    st.stop()

returns = returns[selected_assets]

# ====================================
# OPTIMIZATION: MIN VARIANCE
# ====================================

cov_matrix = returns.cov()
mean_returns = returns.mean()

def portfolio_vol(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = tuple((0,1) for _ in selected_assets)

init_guess = np.ones(len(selected_assets)) / len(selected_assets)

opt = minimize(portfolio_vol, init_guess,
               method='SLSQP',
               bounds=bounds,
               constraints=constraints)

weights = opt.x

weights_df = pd.DataFrame({
    "Asset": selected_assets,
    "Weight": weights
})

st.subheader("Optimized Allocation (Min Variance)")
st.dataframe(weights_df)

# ====================================
# PIE CHART
# ====================================

fig_pie = px.pie(weights_df,
                 names="Asset",
                 values="Weight",
                 title="Capital Allocation Breakdown")

st.plotly_chart(fig_pie, use_container_width=True)

# ====================================
# RISK CONTRIBUTION
# ====================================

portfolio_variance = portfolio_vol(weights)**2
marginal_contrib = np.dot(cov_matrix, weights)
risk_contrib = weights * marginal_contrib / portfolio_variance

risk_df = pd.DataFrame({
    "Asset": selected_assets,
    "Risk Contribution": risk_contrib
})

fig_bar = px.bar(risk_df,
                 x="Asset",
                 y="Risk Contribution",
                 title="Risk Contribution by Asset")

st.plotly_chart(fig_bar, use_container_width=True)

# ====================================
# DIVERSIFICATION RATIO
# ====================================

weighted_vol = np.dot(weights, np.sqrt(np.diag(cov_matrix)))
portfolio_volatility = portfolio_vol(weights)
div_ratio = weighted_vol / portfolio_volatility

st.metric("Diversification Ratio", round(div_ratio, 2))

# ====================================
# HISTORICAL DRAWDOWN
# ====================================

portfolio_returns = returns.dot(weights)
cum = (1 + portfolio_returns).cumprod()
peak = cum.cummax()
drawdown = (cum - peak) / peak

fig_dd = px.line(drawdown,
                 title="Historical Drawdown")

st.plotly_chart(fig_dd, use_container_width=True)

# ====================================
# CVaR
# ====================================

cvar = portfolio_returns.quantile(0.05)

st.metric("Historical CVaR (5%)", round(cvar, 4))

# ====================================
# PROBABILISTIC NEAR-FUTURE SIMULATION
# ====================================

simulations = []
for _ in range(1000):
    sim = np.random.multivariate_normal(mean_returns, cov_matrix)
    simulations.append(np.dot(weights, sim))

simulations = np.array(simulations)

fig_sim = px.histogram(simulations,
                       nbins=40,
                       title="Probabilistic 1-Period Forward Return Distribution")

st.plotly_chart(fig_sim, use_container_width=True)

st.markdown("""
### Interpretation
- Allocation minimizes variance under constraints.
- Risk contribution shows which asset dominates portfolio volatility.
- Diversification ratio > 1.5 indicates meaningful diversification.
- CVaR measures left-tail exposure.
- Forward simulation reflects distribution, not prediction.
""")
