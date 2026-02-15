import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("MS-GARCH Volatility Analytics")

data = load_all()

garch = data["garch"]
vix = data["vix"]

if garch.empty or vix.empty:
    st.error("Volatility or VIX data missing.")
    st.stop()

garch_col = garch.columns[0]
vix_col = vix.columns[0]

# Timeline selection
start_date = st.date_input("Start Date", garch.index.min())
end_date = st.date_input("End Date", garch.index.max())

mask = (garch.index >= pd.to_datetime(start_date)) & \
       (garch.index <= pd.to_datetime(end_date))

garch_f = garch.loc[mask]
vix_f = vix.loc[mask]

# ==========================
# MS-GARCH vs VIX
# ==========================

st.subheader("MS-GARCH Forecast vs Market Volatility")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=garch_f.index,
    y=garch_f[garch_col],
    name="MS-GARCH Forecast",
    line=dict(width=2)
))

fig.add_trace(go.Scatter(
    x=vix_f.index,
    y=vix_f[vix_col],
    name="VIX",
    line=dict(width=2)
))

fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig, use_container_width=True)

# ==========================
# Volatility Clustering
# ==========================

st.subheader("Volatility Clustering")

fig_cluster = px.scatter(
    x=garch_f[garch_col][:-1],
    y=garch_f[garch_col][1:],
    labels={"x":"Lag Vol", "y":"Current Vol"},
    title="Volatility Autocorrelation"
)

fig_cluster.update_layout(template="plotly_dark")
st.plotly_chart(fig_cluster, use_container_width=True)

# ==========================
# Rolling Realized Vol
# ==========================

st.subheader("Rolling Realized vs Forecast")

realized = vix_f[vix_col].rolling(20).mean()

fig_roll = go.Figure()

fig_roll.add_trace(go.Scatter(
    x=realized.index,
    y=realized,
    name="Rolling Realized"
))

fig_roll.add_trace(go.Scatter(
    x=garch_f.index,
    y=garch_f[garch_col],
    name="GARCH Forecast"
))

fig_roll.update_layout(template="plotly_dark")
st.plotly_chart(fig_roll, use_container_width=True)

# ==========================
# Volatility Distribution
# ==========================

st.subheader("Volatility Distribution")

fig_hist = px.histogram(
    garch_f[garch_col],
    nbins=50,
    title="Volatility Distribution"
)

fig_hist.update_layout(template="plotly_dark")
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("""
### Interpretation

- MS-GARCH captures clustering behavior.
- Autocorrelation scatter confirms volatility persistence.
- Rolling comparison shows adaptive forecasting.
- Distribution highlights tail risk structure.
""")
