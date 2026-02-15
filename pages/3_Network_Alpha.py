import streamlit as st
import plotly.express as px
from utils.load_data import load_all

st.title("Contagion & Survival vs Alpha")

data = load_all()
contagion = data["contagion"]
metrics = data["metrics"]

if contagion.empty:
    st.warning("Contagion data missing.")
    st.stop()

st.subheader("Systemic Contagion Over Time")

fig = px.line(
    contagion,
    x=contagion.index,
    y=contagion.columns[0],
    title="Contagion Index"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Survival vs Alpha Matrix")

if not metrics.empty:
    st.dataframe(metrics)
else:
    st.warning("Portfolio metrics missing.")
