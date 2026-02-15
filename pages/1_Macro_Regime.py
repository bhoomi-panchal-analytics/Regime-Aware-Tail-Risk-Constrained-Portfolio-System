import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("Advanced Macro Regime Diagnostics")

data = load_all()
regime_probs = data["regime_probs"]

if regime_probs.empty:
    st.error("Regime probabilities missing.")
    st.stop()

regime_probs = regime_probs.apply(pd.to_numeric, errors="coerce").dropna()

# ==========================
# Timeline selector
# ==========================

min_date = assets.index.min()
max_date = assets.index.max()

if pd.isna(min_date) or pd.isna(max_date):
    st.error("Invalid date index.")
    st.stop()

start_date = st.date_input("Start Date", min_date.date())
end_date = st.date_input("End Date", max_date.date())

end_date = st.date_input("End Date", regime_probs.index.max())

mask = (regime_probs.index >= pd.to_datetime(start_date)) & \
       (regime_probs.index <= pd.to_datetime(end_date))

regime_filtered = regime_probs.loc[mask]

# ==========================
# Heatmap
# ==========================

st.subheader("Regime Probability Heatmap")

fig_heat = go.Figure(
    go.Heatmap(
        z=regime_filtered.T.values,
        x=regime_filtered.index,
        y=regime_filtered.columns,
        colorscale="RdBu_r",
        zmin=0,
        zmax=1,
        colorbar=dict(title="Probability")
    )
)

fig_heat.update_layout(height=500, template="plotly_dark")

st.plotly_chart(fig_heat, use_container_width=True)

# ==========================
# Dominant Regime Over Time
# ==========================

st.subheader("Dominant Regime Path")

dominant = regime_filtered.idxmax(axis=1)

fig_dom = px.line(
    x=regime_filtered.index,
    y=dominant.astype("category").cat.codes,
    labels={"x": "Date", "y": "Regime State"},
)

fig_dom.update_layout(template="plotly_dark")
st.plotly_chart(fig_dom, use_container_width=True)

# ==========================
# Regime Persistence
# ==========================

st.subheader("Regime Persistence Duration")

durations = dominant.groupby((dominant != dominant.shift()).cumsum()).agg(['first','size'])
fig_persist = px.bar(
    durations,
    y="size",
    title="Regime Duration Blocks"
)

fig_persist.update_layout(template="plotly_dark")
st.plotly_chart(fig_persist, use_container_width=True)

# ==========================
# Transition Matrix
# ==========================

st.subheader("Estimated Transition Matrix")

states = dominant.astype("category").cat.codes
trans_matrix = pd.crosstab(states[:-1], states[1:], normalize="index")

fig_trans = px.imshow(trans_matrix,
                      color_continuous_scale="Blues",
                      title="Regime Transition Probabilities")

fig_trans.update_layout(template="plotly_dark")
st.plotly_chart(fig_trans, use_container_width=True)

# ==========================
# Regime Entropy
# ==========================

st.subheader("Rolling Regime Uncertainty (Entropy)")

entropy = -(regime_filtered * np.log(regime_filtered + 1e-9)).sum(axis=1)

fig_entropy = px.line(
    entropy,
    title="Regime Probability Entropy"
)

fig_entropy.update_layout(template="plotly_dark")
st.plotly_chart(fig_entropy, use_container_width=True)

st.markdown("""
### Interpretation

- Heatmap shows regime dominance and persistence.
- Transition matrix quantifies regime switching intensity.
- Entropy indicates uncertainty — spikes imply regime ambiguity.
- Long duration blocks indicate structural regime persistence.
""")
