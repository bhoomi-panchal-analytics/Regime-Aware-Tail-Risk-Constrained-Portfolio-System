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
    st.error("market_data_template.csv not found in /data.")
    st.stop()

raw_df = data["market_data_template"].copy()

# =====================================================
# FORCE DATETIME INDEX (NO ASSUMPTIONS)
# =====================================================

df = raw_df.copy()

# Case 1: Already datetime index
if not isinstance(df.index, pd.DatetimeIndex):

    # Try to find a date column
    date_column = None

    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > len(df) * 0.7:
                date_column = col
                break
        except:
            continue

    if date_column is None:
        st.error("No valid date column detected.")
        st.stop()

    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column])
    df = df.set_index(date_column)

# Now we guarantee datetime index
df.index = pd.to_datetime(df.index, errors="coerce")
df = df[~df.index.isna()]

if df.empty:
    st.error("Datetime index conversion failed — dataset empty.")
    st.stop()

df = df.sort_index()

# =====================================================
# ENSURE NUMERIC ASSETS
# =====================================================

df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna(how="all")

if df.shape[1] < 3:
    st.error("Need at least 3 asset columns.")
    st.stop()

assets = df.copy()

# =====================================================
# SAFE DATE FILTER
# =====================================================

min_date = assets.index.min()
max_date = assets.index.max()

if pd.isna(min_date) or pd.isna(max_date):
    st.error("Invalid datetime bounds.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input(
        "Start Date",
        value=min_date.date(),
        min_value=min_date.date(),
        max_value=max_date.date()
    )

with col2:
    end_date = st.date_input(
        "End Date",
        value=max_date.date(),
        min_value=min_date.date(),
        max_value=max_date.date()
    )

if start_date >= end_date:
    st.warning("Start date must be earlier than end date.")
    st.stop()

assets = assets.loc[
    (assets.index >= pd.to_datetime(start_date)) &
    (assets.index <= pd.to_datetime(end_date))
]

if len(assets) < 120:
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

if len(returns) <= window:
    st.warning("Window too large for selected range.")
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
    title="Average Absolute Correlation (Contagion Index)"
)

st.plotly_chart(fig_density, use_container_width=True)

# =====================================================
# CORRELATION HEATMAP
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
cov = returns.cov().fillna(0)

weights = np.ones(len(vol)) / len(vol)

portfolio_vol = np.sqrt(weights @ cov.values @ weights)
weighted_vol = weights @ vol.values

div_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 0

st.metric("Diversification Ratio", f"{div_ratio:.2f}")

# =====================================================
# EIGENVALUE ANALYSIS
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
