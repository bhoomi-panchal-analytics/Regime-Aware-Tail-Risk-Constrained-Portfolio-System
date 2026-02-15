import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("Advanced Macro Regime Diagnostics")

# ==========================
# LOAD DATA
# ==========================

data = load_all()

regime_probs = None

# Auto-detect probability dataset
for key in data:
    df = data[key]
    if isinstance(df, pd.DataFrame):
        if df.shape[1] >= 2:
            # Check if values between 0 and 1
            if df.max().max() <= 1.1 and df.min().min() >= -0.1:
                regime_probs = df
                break

if regime_probs is None:
    st.error("No regime probability dataset detected.")
    st.stop()

# ==========================
# CLEAN & VALIDATE
# ==========================

regime_probs = regime_probs.apply(pd.to_numeric, errors="coerce")
regime_probs = regime_probs.dropna()
regime_probs.index = pd.to_datetime(regime_probs.index, errors="coerce")
regime_probs = regime_probs[~regime_probs.index.isna()]
regime_probs = regime_probs.sort_index()

if regime_probs.empty:
    st.error("Regime probability dataset empty after cleaning.")
    st.stop()

# Normalize rows (in case not exactly summing to 1)
row_sums = regime_probs.sum(axis=1)
regime_probs = regime_probs.div(row_sums, axis=0)

# ==========================
# TIMELINE
# ==========================

min_date = regime_probs.index.min()
max_date = regime_probs.index.max()

start_date = st.date_input("Start Date", min_date.date())
end_date = st.date_input("End Date", max_date.date())

regime_probs = regime_probs.loc[
    (regime_probs.index >= pd.to_datetime(start_date)) &
    (regime_probs.index <= pd.to_datetime(end_date))
]

if regime_probs.empty:
    st.warning("No data in selected range.")
    st.stop()

# ==========================
# 1️⃣ HEATMAP
# ==========================

st.subheader("Regime Probability Heatmap")

fig_heat = px.imshow(
    regime_probs.T,
    aspect="auto",
    color_continuous_scale="RdBu_r",
    labels=dict(x="Time", y="Regime", color="Probability"),
    template="plotly_dark"
)

st.plotly_chart(fig_heat, use_container_width=True)

# ==========================
# 2️⃣ DOMINANT REGIME OVER TIME
# ==========================

st.subheader("Dominant Regime Timeline")

dominant_regime = regime_probs.idxmax(axis=1)

fig_dom = px.line(
    dominant_regime,
    template="plotly_dark"
)

st.plotly_chart(fig_dom, use_container_width=True)

# ==========================
# 3️⃣ REGIME CONFIDENCE
# ==========================

st.subheader("Rolling Regime Confidence")

confidence = regime_probs.max(axis=1).rolling(30).mean()

fig_conf = px.line(confidence,
                   template="plotly_dark",
                   title="30-Day Rolling Confidence")

st.plotly_chart(fig_conf, use_container_width=True)

# ==========================
# 4️⃣ REGIME DISTRIBUTION
# ==========================

st.subheader("Regime Distribution")

distribution = dominant_regime.value_counts()

fig_dist = px.bar(distribution,
                  template="plotly_dark")

st.plotly_chart(fig_dist, use_container_width=True)

# ==========================
# 5️⃣ TRANSITION MATRIX
# ==========================

st.subheader("Transition Matrix")

transitions = pd.crosstab(
    dominant_regime.shift(1),
    dominant_regime
)

transition_prob = transitions.div(transitions.sum(axis=1), axis=0)

fig_trans = px.imshow(
    transition_prob,
    template="plotly_dark",
    color_continuous_scale="Blues"
)

st.plotly_chart(fig_trans, use_container_width=True)

# ==========================
# 6️⃣ PERSISTENCE METRIC
# ==========================

st.subheader("Regime Persistence")

persistence = np.diag(transition_prob.fillna(0))

for i, p in enumerate(persistence):
    st.metric(f"Regime {i} Persistence", f"{p:.2f}")

# ==========================
# 7️⃣ CRISIS HIGHLIGHT BANDS
# ==========================

st.subheader("Crisis Highlighted Timeline")

fig_crisis = go.Figure()

for col in regime_probs.columns:
    fig_crisis.add_trace(go.Scatter(
        x=regime_probs.index,
        y=regime_probs[col],
        name=f"Regime {col}"
    ))

# Add crisis shading
fig_crisis.add_vrect(
    x0="2008-01-01", x1="2009-06-01",
    fillcolor="red", opacity=0.2, line_width=0
)

fig_crisis.add_vrect(
    x0="2020-02-01", x1="2020-06-01",
    fillcolor="orange", opacity=0.2, line_width=0
)

fig_crisis.update_layout(template="plotly_dark")

st.plotly_chart(fig_crisis, use_container_width=True)

st.markdown("""
### Interpretation

• Heatmap shows regime probability clustering  
• Dominant regime identifies structural phase  
• Rolling confidence measures regime clarity  
• Transition matrix quantifies persistence  
• Crisis bands validate regime switching behavior  
""")
