import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Systemic Contagion & Network Diagnostics")

# ======================================================
# LOAD DATA SAFELY
# ======================================================

from utils.load_data import load_all
data = load_all()

if "market_data_template" not in data:
    st.error("market_data_template.csv missing.")
    st.stop()

raw = data["market_data_template"].copy()

# ======================================================
# SAFE DATETIME PROCESSING
# ======================================================

if "Date" in raw.columns:
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"])
    raw = raw.set_index("Date")

raw.index = pd.to_datetime(raw.index, errors="coerce")
raw = raw[~raw.index.isna()]
raw = raw.sort_index()

if raw.empty:
    st.error("Dataset empty after datetime cleaning.")
    st.stop()

# ======================================================
# NUMERIC CLEANING
# ======================================================

raw = raw.apply(pd.to_numeric, errors="coerce")

# Drop columns with more than 50% NaN
valid_cols = raw.columns[raw.isna().mean() < 0.5]
df = raw[valid_cols].copy()

if df.shape[1] < 2:
    st.error("Need at least two valid asset columns.")
    st.stop()

df = df.fillna(method="ffill").fillna(0)

# ======================================================
# DATE RANGE (NO BOUND ERRORS)
# ======================================================

min_date = df.index.min()
max_date = df.index.max()

if pd.isna(min_date) or pd.isna(max_date):
    st.error("Invalid date bounds.")
    st.stop()

start_date = st.date_input("Start Date", min_date.date())
end_date = st.date_input("End Date", max_date.date())

if start_date >= end_date:
    st.error("Invalid date range.")
    st.stop()

filtered = df.loc[str(start_date):str(end_date)]

if len(filtered) < 60:
    st.warning("Short sample period. Results may be unstable.")

# ======================================================
# 1️⃣ Rolling Contagion Index
# ======================================================

st.subheader("1. Rolling Contagion Index")

window = st.slider("Rolling Window", 30, 150, 60)

rolling_density = []

for i in range(window, len(filtered)):
    corr = filtered.iloc[i-window:i].corr()
    upper = corr.values[np.triu_indices_from(corr, k=1)]
    rolling_density.append(np.nanmean(np.abs(upper)))

rolling_density = pd.Series(
    rolling_density,
    index=filtered.index[window:]
)

fig1 = px.line(
    rolling_density,
    template="plotly_dark",
    title="Average Absolute Correlation (Contagion Proxy)"
)

st.plotly_chart(fig1, use_container_width=True)

# ======================================================
# 2️⃣ Correlation Heatmap
# ======================================================

st.subheader("2. Correlation Heatmap (Current Window)")

corr_matrix = filtered.iloc[-window:].corr().fillna(0)

fig2 = px.imshow(
    corr_matrix,
    zmin=-1,
    zmax=1,
    color_continuous_scale="RdBu",
    text_auto=True
)

fig2.update_layout(template="plotly_dark")
st.plotly_chart(fig2, use_container_width=True)

# ======================================================
# 3️⃣ Correlation Distribution
# ======================================================

st.subheader("3. Correlation Distribution")

upper_vals = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]

fig3 = px.histogram(
    upper_vals,
    nbins=25,
    template="plotly_dark",
    title="Pairwise Correlation Histogram"
)

st.plotly_chart(fig3, use_container_width=True)

# ======================================================
# 4️⃣ Systemic Centrality
# ======================================================

st.subheader("4. Systemic Centrality Score")

centrality = corr_matrix.abs().mean().sort_values(ascending=False)

fig4 = px.bar(
    centrality,
    template="plotly_dark",
    title="Average Absolute Correlation by Asset"
)

st.plotly_chart(fig4, use_container_width=True)

# ======================================================
# 5️⃣ Diversification Ratio
# ======================================================

st.subheader("5. Diversification Ratio")

vol = filtered.std()
cov = filtered.cov()

weights = np.ones(len(vol)) / len(vol)

portfolio_vol = np.sqrt(weights @ cov.values @ weights)
weighted_vol = weights @ vol.values

div_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 0

st.metric("Diversification Ratio", round(div_ratio, 2))

# ======================================================
# 6️⃣ Eigenvalue Spectrum
# ======================================================

st.subheader("6. Eigenvalue Spectrum")

try:
    eigvals = np.linalg.eigvals(corr_matrix.values)
    eigvals = np.real(eigvals)

    fig6 = px.bar(
        eigvals,
        template="plotly_dark",
        title="Eigenvalue Magnitudes"
    )

    st.plotly_chart(fig6, use_container_width=True)

    concentration = eigvals.max() / eigvals.sum()
    st.metric("Systemic Concentration", round(concentration, 2))

except:
    st.warning("Eigen decomposition unstable.")

# ======================================================
# 7️⃣ Rolling Volatility Overlay
# ======================================================

st.subheader("7. Average Rolling Volatility")

rolling_vol = filtered.rolling(window).std().mean(axis=1)

fig7 = px.line(
    rolling_vol,
    template="plotly_dark",
    title="Cross-Asset Rolling Volatility"
)

st.plotly_chart(fig7, use_container_width=True)

# ======================================================
# 8️⃣ Network Stress Classification
# ======================================================

st.subheader("8. Network Stress Regime")

if not rolling_density.empty:
    latest = rolling_density.iloc[-1]
    q25 = rolling_density.quantile(0.25)
    q75 = rolling_density.quantile(0.75)

    if latest > q75:
        regime = "High Contagion"
    elif latest < q25:
        regime = "Low Contagion"
    else:
        regime = "Moderate Contagion"

    st.metric("Current Regime", regime)

st.markdown("""
### Interpretation

• Rising contagion index → diversification breakdown  
• Dominant eigenvalue spike → systemic factor dominance  
• Falling diversification ratio → hidden fragility  
• Volatility + correlation rising together → crisis regime
""")
