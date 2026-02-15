import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("Contagion Network & Diversification Diagnostics")

# =====================================================
# LOAD SPECIFIC ETF FILES
# =====================================================

data = load_all()

required_assets = ["SPY", "TLT", "GLD", "DBC", "UUP", "SHY"]

asset_frames = []

for asset in required_assets:
    for key in data.keys():
        if asset in key:
            df = data[key]
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.dropna()
            df = df.iloc[:, 0]  # use first column
            df.name = asset
            asset_frames.append(df)

if len(asset_frames) < 3:
    st.error("Not enough ETF market data found. Ensure SPY, TLT, GLD etc are in /data.")
    st.stop()

# Merge into one dataframe
assets = pd.concat(asset_frames, axis=1).dropna()

assets.index = pd.to_datetime(assets.index)
assets = assets.sort_index()

# =====================================================
# DATE FILTER
# =====================================================

min_date = assets.index.min()
max_date = assets.index.max()

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input("Start Date", min_date.date())

with col2:
    end_date = st.date_input("End Date", max_date.date())

assets = assets.loc[
    (assets.index >= pd.to_datetime(start_date)) &
    (assets.index <= pd.to_datetime(end_date))
]

if len(assets) < 100:
    st.warning("Not enough data after filtering.")
    st.stop()

# =====================================================
# RETURNS
# =====================================================

returns = assets.pct_change().dropna()

# =====================================================
# ROLLING CONTAGION INDEX
# =====================================================

st.subheader("Rolling Contagion Index")

window = st.slider("Rolling Window (days)", 30, 150, 60)

density = []

for i in range(window, len(returns)):
    corr = returns.iloc[i-window:i].corr()
    upper = corr.abs().values[np.triu_indices_from(corr, k=1)]
    density.append(np.mean(upper))

density_series = pd.Series(
    density,
    index=returns.index[window:]
)

fig_density = px.line(
    density_series,
    template="plotly_dark",
    title="Average Absolute Correlation"
)

st.plotly_chart(fig_density, use_container_width=True)

# =====================================================
# CURRENT CORRELATION MATRIX
# =====================================================

st.subheader("Current Correlation Heatmap")

corr_matrix = returns.iloc[-window:].corr()

fig_heat = px.imshow(
    corr_matrix,
    text_auto=True,
    zmin=-1,
    zmax=1,
    color_continuous_scale="RdBu"
)

fig_heat.update_layout(template="plotly_dark")

st.plotly_chart(fig_heat, use_container_width=True)

# =====================================================
# CENTRALITY
# =====================================================

st.subheader("Systemic Centrality")

centrality = corr_matrix.abs().mean().sort_values(ascending=False)

fig_cent = px.bar(
    centrality,
    template="plotly_dark",
    title="Mean Absolute Correlation by Asset"
)

st.plotly_chart(fig_cent, use_container_width=True)

# =====================================================
# DIVERSIFICATION RATIO
# =====================================================

st.subheader("Diversification Ratio")

vol = returns.std()
cov = returns.cov()

weights = np.ones(len(vol)) / len(vol)

portfolio_vol = np.sqrt(weights @ cov.values @ weights)
weighted_vol = weights @ vol.values

div_ratio = weighted_vol / portfolio_vol

st.metric("Diversification Ratio", f"{div_ratio:.2f}")

# =====================================================
# EIGENVALUE CONCENTRATION
# =====================================================

st.subheader("Eigenvalue Spectrum")

eigvals = np.linalg.eigvals(corr_matrix.values)
eigvals = np.real(eigvals)

fig_eig = px.bar(
    eigvals,
    template="plotly_dark",
    title="Eigenvalue Distribution"
)

st.plotly_chart(fig_eig, use_container_width=True)

concentration = eigvals.max() / eigvals.sum()
st.metric("Systemic Concentration Ratio", f"{concentration:.2f}")

# =====================================================
# CORRELATION DISTRIBUTION
# =====================================================

st.subheader("Correlation Distribution")

upper_vals = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]

fig_hist = px.histogram(
    upper_vals,
    nbins=30,
    template="plotly_dark",
    title="Distribution of Pairwise Correlations"
)

st.plotly_chart(fig_hist, use_container_width=True)

# =====================================================
# REGIME
# =====================================================

current_density = density_series.iloc[-1]

if current_density > density_series.quantile(0.75):
    regime = "High Contagion"
elif current_density < density_series.quantile(0.25):
    regime = "Low Contagion"
else:
    regime = "Medium Contagion"

st.metric("Current Contagion Regime", regime)
