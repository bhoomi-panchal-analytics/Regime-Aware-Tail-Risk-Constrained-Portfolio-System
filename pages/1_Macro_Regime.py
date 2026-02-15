import streamlit as st
import plotly.express as px
import pandas as pd
from utils.load_data import load_all
data = load_all()
regime_probs = data["regime_probs"]

if regime_probs.empty:
    st.warning("Regime probabilities file missing.")
    st.stop()


import plotly.express as px

if regime_probs.empty:
    st.warning("Regime data empty.")
    st.stop()

fig = px.imshow(
    regime_probs.T.values,
    aspect="auto",
    color_continuous_scale="RdBu_r"
)

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Regime"
)

st.plotly_chart(fig, use_container_width=True)


st.markdown("### Crisis Highlights")

crises = {
    "2008 Crisis": ("2008-01-01", "2009-06-01"),
    "2012 Euro Stress": ("2012-01-01", "2012-12-01"),
    "2017 Calm Growth": ("2017-01-01", "2017-12-01"),
    "2020 COVID": ("2020-02-01", "2020-06-01"),
}

for name, (start, end) in crises.items():
    st.write("Data shape:", regime_probs.shape)
    st.write(regime_probs.head())

