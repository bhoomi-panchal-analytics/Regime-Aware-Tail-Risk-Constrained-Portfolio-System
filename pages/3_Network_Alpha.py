import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
# DATETIME SAFETY
# ======================================================

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date")
else:
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

df = df.sort_index()

if df.empty:
    st.error("Dataset empty after datetime parsing.")
    st.stop()

# ======================================================
# NUMERIC CLEANING
# ======================================================

df = df.apply(pd.to_numeric, errors="coerce")

# Remove columns with excessive missing values
null_ratio = df.isna().mean()
df = df.loc[:, null_ratio < 0.5]

if df.shape[1] < 2:
    st.error("Not enough valid asset columns.")
    st.stop()

df = df.fillna(method="ffill").fillna(0)

assets = df.copy()

# ======================================================
# SAFE DATE RANGE
# ======================================================

date_list = assets.index.tolist()

date_range = st.select_slider(
    "Select Date Range",
    options=date_list,
    value=(date_list[0], date_list[-1])
)

filtered = assets.loc[date_range[0]:date_range[1]]

if len(filtered) < 100:
    st.warning("Short time window — statistical stability reduced.")

# ======================================================
# ROLLING CONTAGION INDEX
# ======================================================

st.subheader("1. Rolling Contagion Index")

window = st.slider("Rolling Window (Days)", 30, 150, 60)

density_values = []

for i in range(window, len(filtered)):
    corr = filtered.iloc[i-window:i].corr()
    upper = corr.values[np.triu_indices_from(corr, k=1)]
    density_values.append(np.nanmean(np.abs(upper)))

density_series = pd.Series(
    density_values,
    index=filtered.index[window:]
)

fig1 = px.line(
    density_series,
    template="plotly_dark",
    title="Average Absolute Correlation Over Time"
)

st.plotly_chart(fig1, use_container_width=True)

# ======================================================
# CORRELATION HEATMAP
# ======================================================

st.subheader("2. Current Correlation Matrix")

corr_matrix = filtered.iloc[-window:].corr().fillna(0)

fig2 = px.imshow(
    corr_matrix,
    text_auto=True,
    zmin=-1,
    zmax=1,
    color_continuous_scale="RdBu"
)

fig2.update_layout(template="plotly_dark")
st.plotly_chart(fig2, use_container_width=True)

# ======================================================
# CORRELATION DISTRIBUTION
# ======================================================

st.subheader("3. Correlation Distribution")

upper_vals = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]

fig3 = px.histogram(
    upper_vals,
    nbins=30,
    template="plotly_dark",
    title="Pairwise Correlation Distribution"
)

st.plotly_chart(fig3, use_container_width=True)

# ======================================================
# CENTRALITY BAR CHART
# ======================================================

st.subheader("4. Systemic Centrality Score")

centrality = corr_matrix.abs().mean().sort_values(ascending=False)

fig4 = px.bar(
    centrality,
    template="plotly_dark",
    title="Mean Absolute Correlation by Asset"
)

st.plotly_chart(fig4, use_container_width=True)

# ======================================================
# DIVERSIFICATION RATIO
# ======================================================

st.subheader("5. Diversification Ratio")

vol = filtered.std()
cov = filtered.cov()

weights = np.ones(len(vol)) / len(vol)

portfolio_vol = np.sqrt(weights @ cov.values @ weights)
weighted_vol = weights @ vol.values

div_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 0

st.metric("Diversification Ratio", f"{div_ratio:.2f}")

# ======================================================
# EIGENVALUE SPECTRUM
# ======================================================

st.subheader("6. Eigenvalue Spectrum")

try:
    eigvals = np.linalg.eigvals(corr_matrix.values)
    eigvals = np.real(eigvals)

    fig6 = px.bar(
        eigvals,
        template="plotly_dark",
        title="Eigenvalue Distribution"
    )

    st.plotly_chart(fig6, use_container_width=True)

    concentration = eigvals.max() / eigvals.sum()
    st.metric("Systemic Concentration Ratio", f"{concentration:.2f}")

except:
    st.warning("Eigenvalue decomposition unstable.")

# ======================================================
# NETWORK STRESS GAUGE
# ======================================================

st.subheader("7. Network Stress Gauge")

current_density = density_series.iloc[-1]

low = density_series.quantile(0.25)
high = density_series.quantile(0.75)

if current_density > high:
    regime = "High Contagion"
elif current_density < low:
    regime = "Low Contagion"
else:
    regime = "Moderate Contagion"

st.metric("Current Contagion Regime", regime)

st.markdown("""
### Analytical Interpretation

• High rolling density → systemic coupling  
• Large dominant eigenvalue → single risk factor dominance  
• Low diversification ratio → hidden fragility  

Correlation convergence often precedes liquidity stress.
""")
