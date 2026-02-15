import streamlit as st
import plotly.graph_objects as go
from utils.load_data import load_all

st.title("MS-GARCH Volatility vs VIX")

data = load_all()

garch = data["garch"]
vix = data["vix"]

# --- SAFETY CHECKS ---
if garch.empty:
    st.error("GARCH data missing or empty.")
    st.stop()

if vix.empty:
    st.error("VIX data missing or empty.")
    st.stop()

# Automatically select first column (no hardcoding)
garch_col = garch.columns[0]
vix_col = vix.columns[0]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=garch.index,
    y=garch[garch_col],
    name="MS-GARCH Forecast",
    line=dict(color="blue")
))

fig.add_trace(go.Scatter(
    x=vix.index,
    y=vix[vix_col],
    name="VIX",
    line=dict(color="red")
))

fig.update_layout(
    title="Volatility Adaptation During Crisis",
    xaxis_title="Date",
    yaxis_title="Volatility"
)

st.plotly_chart(fig, use_container_width=True)
