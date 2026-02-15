import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("Contagion Network & Diversification Diagnostics")

# =====================================================
# LOAD MULTI-ASSET DATA
# =====================================================

data = load_all()

assets = None

for key, df in data.items():
    if isinstance(df, pd.DataFrame) and df.shape[1] >= 3:
        df = df.apply(pd.to_numeric, errors="coerce")
        if df.dropna().shape[0] > 200:
            assets = df
            break

if assets is None:
    st.error("No suitable multi-asset dataset detected.")
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

if len(assets) < 100:
    st.warning("Insufficient data in selected window.")
    st.stop()

# =====================================================
# RETURNS
# =====================================================

returns = assets.pct_change().dropna()

if returns.shape[0] < 50:
    st.warning("Not enough return observations.")
    st.stop()

# =====================================================
# ROLLING CONTAGION INDEX
# =====================================================

st.subheader("Rolling Contagion Index")

window = st.slider("Rolling Window (days)", 30, 150, 60)

if len(returns) <= window:
    st.warning("Rolling window too large for dataset.")
    st.stop()

density = []

for i in range(window, len(returns)):
    corr = returns.iloc[i-window:i].corr()

    if corr.isnull().values.any():
        density.append(np.nan)
        continue

    upper = corr.abs().values[np.triu_indices_from(corr, k=1)]
    density.append(np.mean(upper))

density_series = pd.Series(
    density,
    index=returns.index[window:]
).dropna()

if density_series.empty:
    st.warning("Contagion index empty.")
    st.stop()

fig_density = px.line(
    density_series,
    template="plotly_dark",
    title="Average Absolute Correlation Over Time"
)

st.plotly_chart(fig_density, use_container_width=True)

# =====================================================
# CURRENT CORRELATION HEATMAP
# =====================================================

st.subheader("Current Correlation Heatmap")

corr_matrix = returns.iloc[-window:].corr()
corr_matrix = corr_matrix.replace([np.inf, -np.inf], 0)
corr_matrix = corr_matrix.fillna(0)

fig_heat = px.imshow(
    corr_matrix,
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    text_auto=True
)

fig_heat.update_layout(template="plotly_dark")

st.plotly_chart(fig_heat, use_container_width=True)

# =====================================================
# ASSET CENTRALITY BAR
# =====================================================

st.subheader("Systemic Centrality (Mean |Correlation|)")

centrality = corr_matrix.abs().mean().sort_values(ascending=False)

fig_cent = px.bar(
    centrality,
    template="plotly_dark",
    title="Asset Centrality Ranking"
)

st.plotly_chart(fig_cent, use_container_width=True)

# =====================================================
# DIVERSIFICATION RATIO
# =====================================================

st.subheader("Diversification Ratio")

vol = returns.std()
cov = returns.cov()

weights = np.ones(len(vol)) / len(vol)

try:
    portfolio_vol = np.sqrt(weights @ cov.values @ weights)
    weighted_vol = weights @ vol.values

    div_ratio = weighted_vol / portfolio_vol

    st.metric("Diversification Ratio", f"{div_ratio:.2f}")

except:
    st.metric("Diversification Ratio", "Unstable")

# =====================================================
# EIGENVALUE CONCENTRATION
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
    st.warning("Eigenvalue decomposition unstable.")

# =====================================================
# REGIME CLASSIFICATION
# =====================================================

st.subheader("Contagion Regime")

low = density_series.quantile(0.25)
high = density_series.quantile(0.75)

current_density = density_series.iloc[-1]

if current_density <= low:
    regime = "Low Contagion"
elif current_density >= high:
    regime = "High Contagion"
else:
    regime = "Medium Contagion"

st.metric("Current Regime", regime)

# =====================================================
# DISTRIBUTION OF CORRELATION
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
# INTERPRETATION TEXT
# =====================================================

st.markdown("""
### How to Read This Page

• Rising contagion index → tightening correlations  
• High centrality → systemic dominance  
• High first eigenvalue → one-factor market behavior  
• Low diversification ratio → fragile portfolio  

When contagion rises, diversification collapses.
""")
