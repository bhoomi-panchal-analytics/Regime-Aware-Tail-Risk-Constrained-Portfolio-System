import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("MS-GARCH vs VIX — Advanced Volatility Diagnostics")

# =====================================================
# LOAD DATA SAFELY
# =====================================================

data = load_all()

garch = None
vix = None

# Detect GARCH (low magnitude volatility)
for key, df in data.items():
    if isinstance(df, pd.DataFrame) and df.shape[1] >= 1:
        numeric = df.apply(pd.to_numeric, errors="coerce").dropna()
        if numeric.empty:
            continue

        max_val = numeric.max().max()
        mean_val = numeric.mean().mean()

        # heuristic detection
        if 0 < mean_val < 2:
            garch = numeric
        if max_val > 5:
            vix = numeric

if garch is None or vix is None:
    st.error("Unable to auto-detect MS-GARCH or VIX dataset.")
    st.stop()

# =====================================================
# CLEAN & ALIGN
# =====================================================

garch.index = pd.to_datetime(garch.index, errors="coerce")
vix.index = pd.to_datetime(vix.index, errors="coerce")

garch = garch[~garch.index.isna()]
vix = vix[~vix.index.isna()]

garch = garch.sort_index()
vix = vix.sort_index()

combined = garch.join(vix, how="inner")

if combined.empty:
    st.error("No overlapping dates between MS-GARCH and VIX.")
    st.stop()

combined = combined.dropna()

garch_col = combined.columns[0]
vix_col = combined.columns[-1]

# =====================================================
# TIMELINE FILTER
# =====================================================

min_date = combined.index.min()
max_date = combined.index.max()

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input("Start Date", min_date.date())

with col2:
    end_date = st.date_input("End Date", max_date.date())

mask = (combined.index >= pd.to_datetime(start_date)) & \
       (combined.index <= pd.to_datetime(end_date))

combined = combined.loc[mask]

if combined.empty:
    st.warning("No data in selected period.")
    st.stop()

# =====================================================
# 1️⃣ MAIN VOLATILITY COMPARISON
# =====================================================

st.subheader("Forecast vs Implied Volatility")

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
    name="VIX (Implied Vol)",
    line=dict(width=2)
))

# Crisis shading
fig.add_vrect(x0="2008-01-01", x1="2009-06-01",
              fillcolor="red", opacity=0.2, line_width=0)

fig.add_vrect(x0="2020-02-01", x1="2020-06-01",
              fillcolor="orange", opacity=0.2, line_width=0)

fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 2️⃣ VOLATILITY CLUSTERING SCATTER
# =====================================================

st.subheader("Volatility Clustering (Lag Test)")

vol = combined[garch_col]
lag = vol.shift(1)

cluster_df = pd.concat([lag, vol], axis=1).dropna()
cluster_df.columns = ["Lagged", "Current"]

fig_cluster = px.scatter(
    cluster_df,
    x="Lagged",
    y="Current",
    template="plotly_dark",
    trendline="ols"
)

st.plotly_chart(fig_cluster, use_container_width=True)

# =====================================================
# 3️⃣ ROLLING VOL COMPARISON
# =====================================================

st.subheader("Rolling 30-Day Volatility")

rolling_garch = combined[garch_col].rolling(30).mean()
rolling_vix = combined[vix_col].rolling(30).mean()

fig_roll = go.Figure()
fig_roll.add_trace(go.Scatter(x=combined.index, y=rolling_garch,
                              name="Rolling GARCH"))
fig_roll.add_trace(go.Scatter(x=combined.index, y=rolling_vix,
                              name="Rolling VIX"))

fig_roll.update_layout(template="plotly_dark")
st.plotly_chart(fig_roll, use_container_width=True)

# =====================================================
# 4️⃣ VOLATILITY RISK PREMIUM
# =====================================================

st.subheader("Volatility Risk Premium (VIX − Forecast)")

spread = combined[vix_col] - combined[garch_col]

fig_spread = px.line(
    spread,
    template="plotly_dark"
)

st.plotly_chart(fig_spread, use_container_width=True)

# =====================================================
# 5️⃣ REGIME CLASSIFICATION
# =====================================================

st.subheader("Volatility Regime Distribution")

low = vol.quantile(0.25)
high = vol.quantile(0.75)

regime = pd.cut(
    vol,
    bins=[-np.inf, low, high, np.inf],
    labels=["Low", "Medium", "High"]
)

fig_regime = px.histogram(
    regime,
    template="plotly_dark"
)

st.plotly_chart(fig_regime, use_container_width=True)

# =====================================================
# 6️⃣ EXTREME SPIKES
# =====================================================

st.subheader("Top 5% Volatility Events")

threshold = vol.quantile(0.95)
extreme = vol[vol > threshold]

fig_extreme = px.scatter(
    x=extreme.index,
    y=extreme,
    template="plotly_dark"
)

st.plotly_chart(fig_extreme, use_container_width=True)

# =====================================================
# 7️⃣ PERSISTENCE METRIC
# =====================================================

st.subheader("Volatility Persistence")

persistence = vol.autocorr(lag=1)
st.metric("Lag-1 Autocorrelation", f"{persistence:.4f}")

# =====================================================
# 8️⃣ DISTRIBUTION ANALYSIS
# =====================================================

st.subheader("Volatility Distribution")

fig_dist = px.histogram(
    vol,
    nbins=50,
    template="plotly_dark"
)

st.plotly_chart(fig_dist, use_container_width=True)

# =====================================================
# 9️⃣ VOL ACCELERATION
# =====================================================

st.subheader("Volatility Acceleration")

acceleration = vol.diff()

fig_acc = px.line(
    acceleration,
    template="plotly_dark"
)

st.plotly_chart(fig_acc, use_container_width=True)

# =====================================================
# 🔟 SUMMARY INTERPRETATION
# =====================================================

st.markdown("""
### Interpretation Guide

• Clustering confirms heteroskedasticity  
• Spread measures volatility risk premium  
• Persistence shows regime memory  
• Extreme spikes align with systemic crises  
• Rolling trends confirm model responsiveness  

MS-GARCH adapts slowly.  
VIX anticipates forward stress.
""")
