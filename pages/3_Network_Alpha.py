import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Contagion Network & Diversification Diagnostics")

# =====================================================
# LOAD DATA
# =====================================================

from utils.load_data import load_all

data = load_all()

if "market_data_template" not in data:
    st.error("market_data_template.csv missing in /data folder.")
    st.stop()

assets = data["market_data_template"].copy()

# =====================================================
# CLEAN INDEX SAFELY
# =====================================================

assets.index = pd.to_datetime(assets.index, errors="coerce")

# Remove invalid timestamps
assets = assets[~assets.index.isna()]

if assets.empty:
    st.error("No valid datetime index found in dataset.")
    st.stop()

assets = assets.sort_index()

# Ensure numeric
assets = assets.apply(pd.to_numeric, errors="coerce")

# Drop rows where all assets are NaN
assets = assets.dropna(how="all")

if assets.shape[1] < 3:
    st.error("Need at least 3 assets for network diagnostics.")
    st.stop()

# =====================================================
# SAFE DATE RANGE
# =====================================================

min_date = assets.index.min()
max_date = assets.index.max()

if pd.isna(min_date) or pd.isna(max_date):
    st.error("Invalid date range in dataset.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input("Start Date", value=min_date.date())

with col2:
    end_date = st.date_input("End Date", value=max_date.date())

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

assets = assets.loc[
    (assets.index >= pd.to_datetime(start_date)) &
    (assets.index <= pd.to_datetime(end_date))
]

if len(assets) < 100:
    st.warning("Not enough observations after filtering.")
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

if len(returns) <= window:
    st.warning("Window too large for selected date range.")
    st.stop()

density = []

for i in range(window, len(returns)):
    corr = returns.iloc[i-window:i].corr()
    upper = corr.abs().values[np.triu_indices_from(corr, k=1)]
    density.append(np.nanmean(upper))

density_series = pd.Series(
    density,
    index=returns.index[window:]
)

fig_density = px.line(
    density_series,
    template="plotly_dark",
    title="Average Absolute Correlation (Contagion Index)"
)

st.plotly_chart(fig_density, use_container_width=True)

# =====================================================
# CURRENT CORRELATION HEATMAP
# =====================================================

st.subheader("Current Correlation Heatmap")

corr_matrix = returns.iloc[-window:].corr()

# Remove NaNs before eigen operations
corr_matrix = corr_matrix.fillna(0)

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
    title="Mean Absolute Correlation per Asset"
)

st.plotly_chart(fig_cent, use_container_width=True)

# =====================================================
# DIVERSIFICATION RATIO
# =====================================================

st.subheader("Diversification Ratio")

vol = returns.std()
cov = returns.cov().fillna(0)

weights = np.ones(len(vol)) / len(vol)

portfolio_vol = np.sqrt(weights @ cov.values @ weights)
weighted_vol = weights @ vol.values

div_ratio = weighted_vol / portfolio_vol if portfolio_vol != 0 else 0

st.metric("Diversification Ratio", f"{div_ratio:.2f}")

# =====================================================
# EIGENVALUE ANALYSIS (SAFE)
# =====================================================

st.subheader("Eigenvalue Spectrum")

try:
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

except:
    st.warning("Eigenvalue calculation unstable due to matrix conditioning.")

# =====================================================
# CORRELATION DISTRIBUTION
# =====================================================

st.subheader("Correlation Distribution")

upper_vals = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]

fig_hist = px.histogram(
    upper_vals,
    nbins=30,
    template="plotly_dark",
    title="Pairwise Correlation Distribution"
)

st.plotly_chart(fig_hist, use_container_width=True)
