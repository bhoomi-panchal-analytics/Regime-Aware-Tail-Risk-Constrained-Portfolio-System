import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("Systemic Contagion & Network Diagnostics")

# =====================================================
# LOAD MARKET DATA SAFELY
# =====================================================

data = load_all()

assets = None

for key, df in data.items():
    if isinstance(df, pd.DataFrame) and df.shape[1] >= 3:
        numeric = df.apply(pd.to_numeric, errors="coerce")
        if numeric.dropna().shape[0] > 100:
            assets = numeric
            break

if assets is None:
    st.error("No suitable multi-asset dataset found.")
    st.stop()

assets.index = pd.to_datetime(assets.index, errors="coerce")
assets = assets[~assets.index.isna()].sort_index()
assets = assets.ffill().dropna()

if assets.shape[1] < 3:
    st.error("Need at least 3 assets for network analysis.")
    st.stop()

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

if len(assets) < 60:
    st.warning("Not enough data in selected window.")
    st.stop()

# =====================================================
# RETURNS
# =====================================================

returns = assets.pct_change().dropna()

# =====================================================
# ROLLING NETWORK DENSITY
# =====================================================

st.subheader("Rolling Network Density (Contagion Index)")

window = st.slider("Rolling Window", 30, 180, 60)

density = []

for i in range(window, len(returns)):
    corr = returns.iloc[i-window:i].corr()

    if corr.isnull().values.any():
        density.append(np.nan)
        continue

    avg_corr = corr.abs().values[np.triu_indices_from(corr, k=1)].mean()
    density.append(avg_corr)

density_series = pd.Series(
    density,
    index=returns.index[window:]
).dropna()

fig_density = px.line(
    density_series,
    template="plotly_dark"
)

st.plotly_chart(fig_density, use_container_width=True)

# =====================================================
# CURRENT CORRELATION STRUCTURE
# =====================================================

st.subheader("Current Correlation Matrix")

corr_matrix = returns.iloc[-window:].corr()

corr_matrix = corr_matrix.replace([np.inf, -np.inf], np.nan)
corr_matrix = corr_matrix.fillna(0)

fig_corr = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1
)

st.plotly_chart(fig_corr, use_container_width=True)

# =====================================================
# CENTRALITY SCORE
# =====================================================

st.subheader("Asset Centrality Score")

centrality = corr_matrix.abs().mean()
fig_cent = px.bar(
    centrality,
    template="plotly_dark"
)

st.plotly_chart(fig_cent, use_container_width=True)

# =====================================================
# DIVERSIFICATION RATIO
# =====================================================

st.subheader("Diversification Ratio")

vol = returns.std()
cov = returns.cov()

try:
    weights = np.ones(len(vol)) / len(vol)
    portfolio_vol = np.sqrt(weights @ cov.values @ weights)
    weighted_vol = weights @ vol.values

    diversification_ratio = weighted_vol / portfolio_vol
except:
    diversification_ratio = np.nan

st.metric("Diversification Ratio", 
          "Stable" if np.isfinite(diversification_ratio) else "Unstable")

# =====================================================
# EIGENVALUE CONCENTRATION
# =====================================================

st.subheader("Eigenvalue Concentration")

try:
    eigenvalues = np.linalg.eigvals(corr_matrix.values)
    eigenvalues = np.real(eigenvalues)

    fig_eig = px.bar(
        eigenvalues,
        template="plotly_dark"
    )

    st.plotly_chart(fig_eig, use_container_width=True)

    concentration = eigenvalues.max() / eigenvalues.sum()
    st.metric("Systemic Concentration Ratio", f"{concentration:.2f}")

except:
    st.warning("Eigenvalue decomposition unstable in this window.")

# =====================================================
# REGIME CLASSIFICATION
# =====================================================

st.subheader("Contagion Regime Classification")

high_threshold = density_series.quantile(0.75)
low_threshold = density_series.quantile(0.25)

regime = pd.cut(
    density_series,
    bins=[-np.inf, low_threshold, high_threshold, np.inf],
    labels=["Low", "Medium", "High"]
)

fig_regime = px.histogram(
    regime,
    template="plotly_dark"
)

st.plotly_chart(fig_regime, use_container_width=True)

# =====================================================
# INTERPRETATION
# =====================================================

st.markdown("""
### Interpretation Guide

• Rising density → correlation tightening  
• High centrality → systemic risk concentration  
• High first eigenvalue → market factor dominance  
• Low diversification ratio → capital fragility  

When density spikes, diversification disappears.
""")
