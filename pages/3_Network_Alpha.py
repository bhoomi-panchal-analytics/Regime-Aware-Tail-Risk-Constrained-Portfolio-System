import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Simple Contagion & Correlation Analysis")

# =========================
# LOAD DATA
# =========================

from utils.load_data import load_all
data = load_all()

if "market_data_template" not in data:
    st.error("market_data_template.csv not found in /data.")
    st.stop()

df = data["market_data_template"].copy()

# =========================
# CLEAN DATA
# =========================

# Use Date column if present
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date")

# Convert everything numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Drop columns that are all empty
df = df.dropna(axis=1, how="all")

# Fill remaining missing with 0
df = df.fillna(0)

if df.shape[1] < 2:
    st.error("Not enough asset columns available.")
    st.stop()

st.success("Data loaded successfully.")

# =========================
# CORRELATION MATRIX
# =========================

st.subheader("Correlation Heatmap")

corr = df.corr()

fig = px.imshow(
    corr,
    zmin=-1,
    zmax=1,
    color_continuous_scale="RdBu",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# AVERAGE CORRELATION
# =========================

upper = corr.values[np.triu_indices_from(corr, k=1)]
avg_corr = np.mean(np.abs(upper))

st.subheader("Average Absolute Correlation")
st.metric("Contagion Level", round(avg_corr, 3))

# =========================
# BAR CHART OF CORRELATION STRENGTH
# =========================

st.subheader("Average Correlation per Asset")

asset_corr = corr.abs().mean().sort_values(ascending=False)

fig2 = px.bar(
    asset_corr,
    template="plotly_dark"
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
### What This Means

• High average correlation → diversification weak  
• Low correlation → portfolio more resilient  
• Assets with highest average correlation → systemic drivers  
""")
