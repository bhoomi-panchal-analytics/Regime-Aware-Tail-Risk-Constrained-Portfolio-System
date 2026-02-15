import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("MS-GARCH Volatility Intelligence Dashboard")

# ==============================
# LOAD & VALIDATE DATA
# ==============================

data = load_all()

garch = data.get("garch", pd.DataFrame())
vix = data.get("vix", pd.DataFrame())

if garch.empty or vix.empty:
    st.error("Volatility data missing. Ensure garch.csv and vix.csv exist inside /data folder.")
    st.stop()

# Force datetime index
garch.index = pd.to_datetime(garch.index)
vix.index = pd.to_datetime(vix.index)

# Align datasets safely
combined = garch.join(vix, how="inner")

if combined.empty:
    st.error("No overlapping dates between GARCH and VIX.")
    st.stop()

garch_col = combined.columns[0]
vix_col = combined.columns[1]

# ==============================
# USER CONTROLS
# ==============================

col1, col2 = st.columns(2)

start_date = col1.date_input("Start Date", combined.index.min())
end_date = col2.date_input("End Date", combined.index.max())

combined = combined.loc[
    (combined.index >= pd.to_datetime(start_date)) &
    (combined.index <= pd.to_datetime(end_date))
]

if combined.empty:
    st.warning("No data in selected period.")
    st.stop()

smooth = st.slider("Smoothing Window", 1, 60, 15)
crisis_overlay = st.checkbox("Enable Crisis Shading", value=True)

# ==============================
# MAIN VOLATILITY PANEL
# ==============================

st.subheader("Forecast vs Market Implied Volatility")

garch_s = combined[garch_col].rolling(smooth).mean()
vix_s = combined[vix_col].rolling(smooth).mean()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=combined.index,
    y=garch_s,
    name="MS-GARCH Forecast",
    line=dict(width=2)
))

fig.add_trace(go.Scatter(
    x=combined.index,
    y=vix_s,
    name="VIX",
    line=dict(width=2)
))

# Crisis shading
if crisis_overlay:
    fig.add_vrect(x0="2008-09-01", x1="2009-06-01", fillcolor="red", opacity=0.1)
    fig.add_vrect(x0="2020-02-01", x1="2020-06-01", fillcolor="orange", opacity=0.1)

fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig, use_container_width=True)

# ==============================
# FORECAST ERROR BAND
# ==============================

st.subheader("Forecast Error Diagnostics")

error = combined[garch_col] - combined[vix_col]

upper = error.mean() + 2 * error.std()
lower = error.mean() - 2 * error.std()

fig_error = go.Figure()

fig_error.add_trace(go.Scatter(x=combined.index, y=error, name="Error"))
fig_error.add_hline(y=upper, line_dash="dash")
fig_error.add_hline(y=lower, line_dash="dash")

fig_error.update_layout(template="plotly_dark")
st.plotly_chart(fig_error, use_container_width=True)

# ==============================
# ROLLING CORRELATION
# ==============================

st.subheader("Rolling Correlation (Model vs Market)")

rolling_corr = combined[garch_col].rolling(60).corr(combined[vix_col])

fig_corr = px.line(rolling_corr, template="plotly_dark")
st.plotly_chart(fig_corr, use_container_width=True)

# ==============================
# VOLATILITY PERCENTILE GAUGE
# ==============================

current_vol = combined[garch_col].iloc[-1]
percentile = (combined[garch_col] < current_vol).mean() * 100

st.subheader("Current Volatility Percentile")

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=percentile,
    title={'text':"Percentile"},
    gauge={'axis': {'range': [0,100]}}
))

fig_gauge.update_layout(template="plotly_dark")
st.plotly_chart(fig_gauge, use_container_width=True)

# ==============================
# REGIME CLASSIFICATION
# ==============================

st.subheader("Volatility Regime Distribution")

low = combined[garch_col].quantile(0.33)
high = combined[garch_col].quantile(0.66)

regimes = pd.cut(
    combined[garch_col],
    bins=[-np.inf, low, high, np.inf],
    labels=["Low", "Medium", "High"]
)

counts = regimes.value_counts()

fig_regime = px.bar(counts, template="plotly_dark")
st.plotly_chart(fig_regime, use_container_width=True)

# ==============================
# VOL OF VOL
# ==============================

st.subheader("Volatility of Volatility")

vol_of_vol = combined[garch_col].rolling(30).std()

fig_vov = px.line(vol_of_vol, template="plotly_dark")
st.plotly_chart(fig_vov, use_container_width=True)

# ==============================
# VOL MOMENTUM
# ==============================

st.subheader("Volatility Momentum")

vol_momentum = combined[garch_col].diff()

fig_mom = px.line(vol_momentum, template="plotly_dark")
st.plotly_chart(fig_mom, use_container_width=True)

# ==============================
# STRESS SPIKE DETECTOR
# ==============================

st.subheader("Extreme Volatility Spikes")

threshold = combined[garch_col].mean() + 2 * combined[garch_col].std()
spikes = combined[garch_col] > threshold

fig_spike = go.Figure()

fig_spike.add_trace(go.Scatter(
    x=combined.index,
    y=combined[garch_col],
    mode="lines",
    name="Volatility"
))

fig_spike.add_trace(go.Scatter(
    x=combined.index[spikes],
    y=combined[garch_col][spikes],
    mode="markers",
    name="Spikes",
    marker=dict(color="red", size=6)
))

fig_spike.update_layout(template="plotly_dark")
st.plotly_chart(fig_spike, use_container_width=True)

# ==============================
# INTERPRETATION PANEL
# ==============================

st.markdown("""
## Structural Interpretation

• Forecast alignment indicates model calibration quality  
• Rolling correlation measures model-market consistency  
• Volatility percentile shows stress positioning  
• Regime classification quantifies structural states  
• Vol-of-vol captures instability acceleration  
• Momentum detects regime transitions  
• Spike detection flags crisis emergence  

Volatility is not random — it is structured state behavior.
""")
