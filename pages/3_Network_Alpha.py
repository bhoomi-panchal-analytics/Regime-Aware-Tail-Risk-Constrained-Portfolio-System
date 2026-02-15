import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =====================================================
# SAFE DATE RANGE HANDLING
# =====================================================

if assets.index.dtype != "datetime64[ns]":
    st.error("Index is not datetime. Check CSV structure.")
    st.stop()

if assets.index.isna().all():
    st.error("All datetime values are invalid (NaT).")
    st.stop()

min_date = assets.index.min()
max_date = assets.index.max()

if pd.isna(min_date) or pd.isna(max_date):
    st.error("Datetime bounds invalid after parsing.")
    st.stop()

# Convert to Python date objects explicitly
min_date_py = min_date.to_pydatetime().date()
max_date_py = max_date.to_pydatetime().date()

start_date = st.date_input(
    "Start Date",
    value=min_date_py,
    min_value=min_date_py,
    max_value=max_date_py
)

end_date = st.date_input(
    "End Date",
    value=max_date_py,
    min_value=min_date_py,
    max_value=max_date_py
)

if start_date >= end_date:
    st.warning("Start date must be earlier than end date.")
    st.stop()

st.set_page_config(layout="wide")
st.title("Systemic Contagion & Network Diagnostics")

# =====================================================
# LOAD DATA
# =====================================================

from utils.load_data import load_all

data = load_all()

if "market_data_template" not in data:
    st.error("market_data_template.csv not found in /data.")
    st.stop()

raw_df = data["market_data_template"].copy()

# =====================================================
# FORCE FIRST COLUMN AS DATE INDEX (DETERMINISTIC)
# =====================================================

df = raw_df.copy()

# Always reset index to avoid double parsing
df = df.reset_index()

# Assume first column contains dates
date_col = df.columns[0]

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# Drop invalid dates
df = df.dropna(subset=[date_col])

if df.empty:
    st.error("No valid datetime values found in first column.")
    st.stop()

df = df.set_index(date_col)
df = df.sort_index()

# Keep only numeric columns
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna(how="all")

if df.shape[1] < 3:
    st.error("At least 3 numeric asset columns required.")
    st.stop()

assets = df.copy()

# =====================================================
# DATE RANGE FILTER (SAFE)
# =====================================================

min_date = assets.index.min()
max_date = assets.index.max()

start_date = st.date_input("Start Date", value=min_date)
end_date = st.date_input("End Date", value=max_date)

if start_date >= end_date:
    st.warning("Invalid date range selected.")
    st.stop()

filtered = assets.loc[
    (assets.index >= pd.to_datetime(start_date)) &
    (assets.index <= pd.to_datetime(end_date))
]

if len(filtered) < 100:
    st.warning("Not enough data in selected range.")
    st.stop()

# =====================================================
# RETURNS
# =====================================================

returns = filtered.pct_change().dropna()

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
    upper = corr.abs().values[np.triu_indices_from(corr, k=1)]
    density.append(np.mean(upper))

density_series = pd.Series(
    density,
    index=returns.index[window:]
)

fig_density = px.line(
    density_series,
    template="plotly_dark",
    title="Average Absolute Correlation (Systemic Contagion)"
)

st.plotly_chart(fig_density, use_container_width=True)

# =====================================================
# CURRENT CORRELATION MATRIX
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
    st.warning("Eigenvalue computation unstable.")

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

• Rising contagion index = tightening correlations  
• High centrality = systemic dominance  
• Large first eigenvalue = market behaving as single factor  
• Low diversification ratio = fragile allocation  

Diversification fails when correlations compress.
""")
