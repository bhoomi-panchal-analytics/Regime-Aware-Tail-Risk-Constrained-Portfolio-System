import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("MS-GARCH Volatility Regime Diagnostics")

# ==========================
# LOAD DATA SAFELY
# ==========================

data = load_all()

garch = data.get("garch", pd.DataFrame())
vix = data.get("vix", pd.DataFrame())

if garch.empty or vix.empty:
    st.error("Volatility data missing. Ensure garch and vix CSV files exist.")
    st.stop()

garch.index = pd.to_datetime(garch.index)
vix.index = pd.to_datetime(vix.index)

# Align
combined = garch.join(vix, how="inner")

if combined.empty:
    st.error("No overlapping dates between GARCH and VIX.")
    st.stop()

garch_col = combined.columns[0]
vix_col = combined.columns[1]

# ==========================
# USER CONTROLS
# ==========================

colA, colB = st.columns(2)

start_date = colA.date_input("Start Date", combined.index.min())
end_date = colB.date_input("End Date", combined.index.max())

combined = combined.loc[
    (combined.index >= pd.to_datetime(start_date)) &
    (combined.index <= pd.to_datetime(end_date))
]

if combined.empty:
    st.warning("No data in selected range.")
    st.stop()

smooth_window = st.slider("Smoothing Window (days)", 1, 60, 20)

# ==========================
# MAIN VOLATILITY COMPARISON
# ==========================

st.subheader("MS-GARCH Forecast vs Market Implied Volatility")

rolling_garch = combined[garch_col].rolling(smooth_window).mean()
rolling_vix = combined[vix_col].rolling(smooth_window).mean()

fig_main = go.Figure()

fig_main.add_trace(go.Scatter(
    x=combined.index,
    y=rolling_garch,
    name="MS-GARCH (Smoothed)",
    line=dict(width=2)
))

fig_main.add_trace(go.Scatter(
    x=combined.index,
    y=rolling_vix,
    name="VIX (Smoothed)",
    line=dict(width=2)
))

fig_main.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig_main, use_container_width=True)

# ==========================
# VOLATILITY CLUSTERING TEST
# ==========================

st.subheader("Volatility Clustering (Lag Relationship)")

lag_vol = combined[garch_col].shift(1)
current_vol = combined[garch_col]

fig_cluster = px.scatter(
    x=lag_vol,
    y=current_vol,
    labels={"x":"Lagged Volatility", "y":"Current Volatility"},
    template="plotly_dark"
)

st.plotly_chart(fig_cluster, use_container_width=True)

# ==========================
# FORECAST ERROR ANALYSIS
# ==========================

st.subheader("Forecast Error (Model vs Market)")

forecast_error = combined[garch_col] - combined[vix_col]

fig_error = px.line(
    forecast_error,
    template="plotly_dark",
    title="Forecast Error Time Series"
)

st.plotly_chart(fig_error, use_container_width=True)

# ==========================
# VOLATILITY PERCENTILE
# ==========================

current_vol = combined[garch_col].iloc[-1]
vol_percentile = (combined[garch_col] < current_vol).mean()

st.subheader("Current Volatility Percentile")

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=vol_percentile * 100,
    title={'text':"Percentile"},
    gauge={'axis': {'range': [0,100]}}
))

fig_gauge.update_layout(template="plotly_dark")
st.plotly_chart(fig_gauge, use_container_width=True)

# ==========================
# REGIME BANDS
# ==========================

st.subheader("Volatility Regime Bands")

low = combined[garch_col].quantile(0.33)
high = combined[garch_col].quantile(0.66)

regime_band = pd.cut(
    combined[garch_col],
    bins=[-np.inf, low, high, np.inf],
    labels=["Low", "Medium", "High"]
)

band_counts = regime_band.value_counts()

fig_band = px.bar(
    band_counts,
    template="plotly_dark",
    title="Volatility Regime Distribution"
)

st.plotly_chart(fig_band, use_container_width=True)

# ==========================
# ROLLING VOL OF VOL
# ==========================

st.subheader("Volatility of Volatility")

vol_of_vol = combined[garch_col].rolling(30).std()

fig_vov = px.line(vol_of_vol, template="plotly_dark")
st.plotly_chart(fig_vov, use_container_width=True)

# ==========================
# INTERPRETATION
# ==========================

st.markdown("""
### Interpretation Framework

• Clustering scatter confirms persistence  
• Forecast error highlights model lag vs implied market expectation  
• Percentile gauge shows stress positioning  
• Regime bands quantify structural volatility states  
• Vol-of-vol indicates instability in risk regime  

Volatility is not noise — it is structural information.
""")
