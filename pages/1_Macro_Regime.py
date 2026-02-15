import streamlit as st
import plotly.express as px
from utils.load_data import load_all

st.title("Macro Regime Probability Heatmap")

data = load_all()
regime_probs = data["regime_probs"]

if regime_probs.empty:
    st.warning("Regime probabilities file missing or empty.")
    st.stop()

# Ensure numeric
regime_probs = regime_probs.apply(pd.to_numeric, errors='coerce')

st.write("Shape:", regime_probs.shape)

fig = px.imshow(
    regime_probs.T,
    aspect="auto",
    color_continuous_scale="RdBu_r",
    labels=dict(x="Time", y="Regime", color="Probability")
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Regime"
)

st.plotly_chart(fig, use_container_width=True)
