import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("MS-GARCH Volatility vs VIX Diagnostics")

data = load_all()

garch = data.get("garch", pd.DataFrame())
vix = data.get("vix", pd.DataFrame())

if garch.empty or vix.empty:
    st.error("GARCH or VIX data missing.")
    st.stop()

# --------------------------
# Ensure datetime index
# --------------------------

garch.index = pd.to_datetime(garch.index)
vix.index = pd.to_datetime(vix.index)

# --------------------------
# Align both datasets
# --------------------------

combined = garch.join(vix, how="inner")

if combined.empty:
    st.error("No overlapping dates between GARCH and VIX.")
    st.stop()

garch_col = combined.columns[0]
vix_col = combined.columns[1]

# --------------------------
# Timeline Selector
# --------------------------

start_date = st.date_input("Start Date", combined.index.min())
end_date = st.date_input("End Date", combined.index.max())

combined = combined.loc[
    (combined.index >= pd.to_datetime(start_date)) &
    (combined.index <= pd.to_datetime(end_date))
]

if combined.empty:
    st.warning("No data in selected date range.")
    st.stop()

# --------------------------
# MS-GARCH vs VIX Plot
# --------------------------

st.subheader("Forecasted Volatility vs Market Implied Volatility")

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

fig.update_layout(
    template="plotly_dark",
    height=500,
    legend=dict(x=0.01, y=0.99)
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Volatility Clustering Check
# --------------------------

st.subheader("Volatility Clustering (Lag vs Current)")

vol = combined[garch_col]

lag_vol = vol.shift(1).dropna()
curr_vol = vol.iloc[1:]

fig_cluster = px.scatter(
    x=lag_vol,
    y=curr_vol,
    labels={"x": "Lagged Volatility", "y": "Current Volatility"},
    template="plotly_dark"
)

st.plotly_chart(fig_cluster, use_container_width=True)

# --------------------------
# Rolling Comparison
# --------------------------

st.subheader("Rolling Volatility Comparison")

rolling_vix = combined[vix_col].rolling(20).mean()
rolling_garch = combined[garch_col].rolling(20).mean()

fig_roll = go.Figure()

fig_roll.add_trace(go.Scatter(
    x=combined.index,
    y=rolling_garch,
    name="Rolling GARCH"
))

fig_roll.add_trace(go.Scatter(
    x=combined.index,
    y=rolling_vix,
    name="Rolling VIX"
))

fig_roll.update_layout(template="plotly_dark")
st.plotly_chart(fig_roll, use_container_width=True)

# --------------------------
# Volatility Regime Zones
# --------------------------

st.subheader("Volatility Regime Zones")

high_vol_threshold = vol.quantile(0.75)
low_vol_threshold = vol.quantile(0.25)

regime_zone = pd.cut(
    vol,
    bins=[-np.inf, low_vol_threshold, high_vol_threshold, np.inf],
    labels=["Low Vol", "Medium Vol", "High Vol"]
)

zone_counts = regime_zone.value_counts()

fig_zone = px.bar(
    zone_counts,
    title="Volatility Regime Distribution",
    template="plotly_dark"
)

st.plotly_chart(fig_zone, use_container_width=True)

st.markdown("""
### Interpretation

• MS-GARCH captures volatility clustering if lag-vol scatter shows structure  
• Rolling comparison shows forecast adaptability  
• Regime distribution quantifies persistence of stress environments  
• Inner-join alignment guarantees no index mismatch errors  
""")
