import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("MS-GARCH vs VIX — Institutional Volatility Diagnostics")

# =====================================================
# LOAD DATA
# =====================================================

data = load_all()

garch = None
vix = None

for key, df in data.items():
    if isinstance(df, pd.DataFrame):
        numeric = df.apply(pd.to_numeric, errors="coerce").dropna()
        if numeric.empty:
            continue

        if numeric.mean().mean() < 2:
            garch = numeric
        if numeric.max().max() > 5:
            vix = numeric

if garch is None or vix is None:
    st.error("Could not detect both MS-GARCH and VIX datasets.")
    st.stop()

# =====================================================
# CLEAN INDEX
# =====================================================

garch.index = pd.to_datetime(garch.index, errors="coerce")
vix.index = pd.to_datetime(vix.index, errors="coerce")

garch = garch[~garch.index.isna()].sort_index()
vix = vix[~vix.index.isna()].sort_index()

garch = garch.resample("B").mean().ffill()
vix = vix.resample("B").mean().ffill()

combined = pd.concat([garch, vix], axis=1).dropna()

if combined.empty:
    st.error("No overlapping data after synchronization.")
    st.stop()

garch_col = combined.columns[0]
vix_col = combined.columns[-1]

# =====================================================
# DATE FILTER
# =====================================================

min_date = combined.index.min()
max_date = combined.index.max()

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input("Start Date", min_date.date())

with col2:
    end_date = st.date_input("End Date", max_date.date())

combined = combined.loc[
    (combined.index >= pd.to_datetime(start_date)) &
    (combined.index <= pd.to_datetime(end_date))
]

if combined.empty:
    st.warning("No data in selected range.")
    st.stop()

# =====================================================
# MAIN COMPARISON
# =====================================================

st.subheader("Volatility Forecast vs Implied Vol")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=combined.index,
    y=combined[garch_col],
    name="MS-GARCH Forecast",
    line=dict(width=2)
))

fig.add_trace(go.Scatter(
    x=combined.index,
    y=combined[vix_col],
    name="VIX",
    line=dict(width=2)
))

fig.add_vrect(x0="2008-01-01", x1="2009-06-01",
              fillcolor="red", opacity=0.15, line_width=0)

fig.add_vrect(x0="2020-02-01", x1="2020-06-01",
              fillcolor="orange", opacity=0.15, line_width=0)

fig.update_layout(template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# VOL RISK PREMIUM
# =====================================================

st.subheader("Volatility Risk Premium")

spread = combined[vix_col] - combined[garch_col]

fig_spread = go.Figure()
fig_spread.add_trace(go.Scatter(
    x=combined.index,
    y=spread,
    name="VIX - Forecast"
))
fig_spread.update_layout(template="plotly_dark")

st.plotly_chart(fig_spread, use_container_width=True)

# =====================================================
# VOLATILITY CLUSTERING (MANUAL OLS)
# =====================================================

st.subheader("Volatility Clustering Test")

vol = combined[garch_col]
lag = vol.shift(1)

cluster_df = pd.concat([lag, vol], axis=1).dropna()
cluster_df.columns = ["Lagged", "Current"]

x = cluster_df["Lagged"].values
y = cluster_df["Current"].values

if len(x) > 10:

    beta = np.cov(x, y)[0, 1] / np.var(x)
    alpha = y.mean() - beta * x.mean()

    reg_line = alpha + beta * x

    fig_cluster = go.Figure()

    fig_cluster.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="Observed"
    ))

    fig_cluster.add_trace(go.Scatter(
        x=x,
        y=reg_line,
        mode="lines",
        name="OLS Fit"
    ))

    fig_cluster.update_layout(template="plotly_dark")

    st.plotly_chart(fig_cluster, use_container_width=True)

    st.metric("Volatility Persistence (β)", f"{beta:.4f}")

else:
    st.warning("Not enough data for clustering test.")

# =====================================================
# AUTOCORRELATION
# =====================================================

st.subheader("Volatility Autocorrelation")

autocorr = vol.autocorr(lag=1)
st.metric("Lag-1 Autocorrelation", f"{autocorr:.4f}")

# =====================================================
# EXTREME EVENTS
# =====================================================

st.subheader("Extreme Volatility Events (Top 5%)")

threshold = vol.quantile(0.95)
extreme = vol[vol > threshold]

fig_extreme = go.Figure()
fig_extreme.add_trace(go.Scatter(
    x=extreme.index,
    y=extreme,
    mode="markers",
    name="Extreme Events"
))
fig_extreme.update_layout(template="plotly_dark")

st.plotly_chart(fig_extreme, use_container_width=True)
