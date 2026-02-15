import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Systemic Contagion & Network Diagnostics")

# ======================================================
# LOAD DATA
# ======================================================

from utils.load_data import load_all
data = load_all()

if "market_data_template" not in data:
    st.error("market_data_template.csv not found.")
    st.stop()

df = data["market_data_template"].copy()

# ======================================================
# SAFE CLEANING
# ======================================================

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date")

df.index = pd.to_datetime(df.index, errors="coerce")
df = df[~df.index.isna()]
df = df.sort_index()

# Convert to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Drop columns with too many NaN
df = df.loc[:, df.isna().mean() < 0.5]

# Forward fill then zero fill
df = df.ffill().fillna(0)

if df.shape[1] < 2:
    st.error("Not enough valid asset columns.")
    st.stop()

if len(df) < 100:
    st.warning("Small dataset. Results may be unstable.")

st.success("Dataset loaded and cleaned successfully.")

# ======================================================
# ROLLING WINDOW
# ======================================================

window = st.slider("Rolling Window (days)", 30, 180, 60)

# ======================================================
# 1️⃣ Rolling Contagion Index
# ======================================================

st.subheader("1. Rolling Contagion Index")

rolling_density = []

for i in range(window, len(df)):
    corr = df.iloc[i-window:i].corr()
    upper = corr.values[np.triu_indices_from(corr, k=1)]
    rolling_density.append(np.nanmean(np.abs(upper)))

rolling_density = pd.Series(
    rolling_density,
    index=df.index[window:]
)

fig1 = px.line(
    rolling_density,
    template="plotly_dark",
    title="Average Absolute Correlation Over Time"
)

st.plotly_chart(fig1, use_container_width=True)

# ======================================================
# 2️⃣ Current Correlation Heatmap
# ======================================================

st.subheader("2. Current Correlation Structure")

corr_matrix = df.iloc[-window:].corr().fillna(0)

fig2 = px.imshow(
    corr_matrix,
    zmin=-1,
    zmax=1,
    color_continuous_scale="RdBu",
    text_auto=False
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
    nbins=30,
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

vol = df.std()
cov = df.cov()

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
# 7️⃣ Rolling Volatility
# ======================================================

st.subheader("7. Cross-Asset Rolling Volatility")

rolling_vol = df.rolling(window).std().mean(axis=1)

fig7 = px.line(
    rolling_vol,
    template="plotly_dark",
    title="Average Rolling Volatility"
)

st.plotly_chart(fig7, use_container_width=True)

# ======================================================
# 8️⃣ Regime Classification
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

    st.metric("Current Network Regime", regime)

st.markdown("""
### Interpretation

• Rising contagion → diversification collapse  
• Large first eigenvalue → dominant systemic factor  
• Falling diversification ratio → hidden concentration  
• Volatility + correlation spike → crisis regime
""")
