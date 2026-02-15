import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Systemic Contagion & Network Diagnostics")

# =====================================================
# LOAD DATA
# =====================================================

from utils.load_data import load_all
data = load_all()

if "market_data_template" not in data:
    st.error("market_data_template.csv not found.")
    st.stop()

df = data["market_data_template"].copy()

# =====================================================
# ENSURE DATE INDEX
# =====================================================

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date")
else:
    df = df.reset_index()
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df = df.dropna(subset=[df.columns[0]])
    df = df.set_index(df.columns[0])

df = df.sort_index()

if df.empty:
    st.error("No valid datetime values found.")
    st.stop()

# =====================================================
# HANDLE NULL RETURNS
# =====================================================

numeric_cols = df.columns

df = df.apply(pd.to_numeric, errors="coerce")

null_ratio = df.isna().mean().mean()

if null_ratio > 0.8:
    st.warning("Most values are missing. Replacing None with 0 for demonstration.")
    df = df.fillna(0)
else:
    df = df.fillna(method="ffill").fillna(0)

assets = df.copy()

# =====================================================
# SAFE RANGE SELECTION
# =====================================================

if len(assets) < 120:
    st.warning("Limited observations available.")

index_list = assets.index.to_list()

if len(index_list) < 2:
    st.error("Not enough data points.")
    st.stop()

date_range = st.select_slider(
    "Date Range",
    options=index_list,
    value=(index_list[0], index_list[-1])
)

filtered = assets.loc[date_range[0]:date_range[1]]

returns = filtered.copy()

# =====================================================
# ROLLING CONTAGION INDEX
# =====================================================

st.subheader("Rolling Contagion Index")

window = st.slider("Rolling Window (days)", 30, 150, 60)

if len(returns) <= window:
    st.warning("Window too large.")
    st.stop()

density = []

for i in range(window, len(returns)):
    corr = returns.iloc[i-window:i].corr().fillna(0)
    upper_vals = corr.abs().values[np.triu_indices_from(corr, k=1)]
    density.append(np.mean(upper_vals))

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

st.subheader("Current Correlation Matrix")

corr_matrix = returns.iloc[-window:].corr().fillna(0)

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
# SYSTEMIC CENTRALITY
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
cov = returns.cov().fillna(0)

weights = np.ones(len(vol)) / len(vol)

portfolio_vol = np.sqrt(weights @ cov.values @ weights)
weighted_vol = weights @ vol.values

div_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 0

st.metric("Diversification Ratio", f"{div_ratio:.2f}")

# =====================================================
# EIGENVALUE SPECTRUM
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
    st.warning("Eigenvalue calculation unstable.")

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

# =====================================================
# CONTAGION REGIME CLASSIFICATION
# =====================================================

current_density = density_series.iloc[-1]

low = density_series.quantile(0.25)
high = density_series.quantile(0.75)

if current_density > high:
    regime = "High Contagion"
elif current_density < low:
    regime = "Low Contagion"
else:
    regime = "Medium Contagion"

st.metric("Current Contagion Regime", regime)

st.markdown("""
### Interpretation

• Rising contagion → correlation compression  
• Large eigenvalue concentration → systemic dominance  
• Low diversification ratio → fragile allocation  

High contagion often precedes drawdowns.
""")
