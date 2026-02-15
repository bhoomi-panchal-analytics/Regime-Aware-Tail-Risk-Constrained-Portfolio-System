import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("Macro Regime Probability Heatmap")

# ===============================
# LOAD DATA
# ===============================

data = load_all()
regime_probs = data["regime_probs"]

if regime_probs.empty:
    st.error("Regime probabilities file missing or empty.")
    st.stop()

# Ensure numeric values
regime_probs = regime_probs.apply(pd.to_numeric, errors="coerce")

# Drop completely empty rows
regime_probs = regime_probs.dropna(how="all")

if regime_probs.shape[0] == 0:
    st.error("Regime file exists but contains no usable data.")
    st.stop()

st.markdown(f"**Data Range:** {regime_probs.index.min().date()} → {regime_probs.index.max().date()}")
st.markdown(f"**Shape:** {regime_probs.shape[0]} observations × {regime_probs.shape[1]} regimes")

# ===============================
# BUILD HEATMAP
# ===============================

fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        z=regime_probs.T.values,
        x=regime_probs.index,
        y=regime_probs.columns,
        colorscale="RdBu_r",
        zmin=0,
        zmax=1,
        colorbar=dict(title="Probability"),
        hovertemplate=
        "Date: %{x}<br>" +
        "Regime: %{y}<br>" +
        "Probability: %{z:.3f}<extra></extra>"
    )
)

# ===============================
# ADD CRISIS SHADING
# ===============================

def add_crisis_shading(fig, start, end, label, color):
    fig.add_vrect(
        x0=start,
        x1=end,
        fillcolor=color,
        opacity=0.2,
        line_width=0,
        annotation_text=label,
        annotation_position="top left"
    )

# 2008 Global Financial Crisis
add_crisis_shading(fig, "2008-01-01", "2009-06-01", "2008 GFC", "red")

# 2012 Euro Stress
add_crisis_shading(fig, "2012-01-01", "2012-12-31", "2012 Euro Crisis", "orange")

# 2017 Calm Growth
add_crisis_shading(fig, "2017-01-01", "2017-12-31", "2017 Calm Expansion", "green")

# 2020 COVID Shock
add_crisis_shading(fig, "2020-02-01", "2020-06-30", "2020 COVID Shock", "purple")

# ===============================
# LAYOUT STYLING
# ===============================

fig.update_layout(
    title="Latent Macro Regime Probability Structure (HMM Output)",
    xaxis_title="Time",
    yaxis_title="Regime State",
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
    template="plotly_dark",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# ===============================
# ADD INTERPRETATION BLOCK
# ===============================

st.markdown("### Interpretation")

st.markdown("""
- Deep red bands during **2008 and 2020** should show dominance of crisis regime.
- 2017 should display stable dominance of growth regime.
- Regime transitions should appear as horizontal probability shifts.
- Persistent blocks validate HMM regime stickiness.
""")
