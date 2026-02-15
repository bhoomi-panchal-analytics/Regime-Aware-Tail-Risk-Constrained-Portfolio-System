import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("Advanced Contagion & Systemic Risk Diagnostics")

# ==========================================
# LOAD DATA SAFELY
# ==========================================

data = load_all()

assets = None

# Auto-detect asset return matrix (multi-column)
for key in data:
    df = data[key]
    if isinstance(df, pd.DataFrame) and df.shape[1] >= 3:
        assets = df.copy()
        break

if assets is None:
    st.error("No multi-asset market dataset detected.")
    st.stop()

assets.index = pd.to_datetime(assets.index, errors="coerce")
assets = assets[~assets.index.isna()]
assets = assets.sort_index()

# ==========================================
# TIMELINE FILTER
# ==========================================

min_date = assets.index.min()
max_date = assets.index.max()

start_date = st.date_input("Start Date", min_date.date())
end_date = st.date_input("End Date", max_date.date())

assets = assets.loc[
    (assets.index >= pd.to_datetime(start_date)) &
    (assets.index <= pd.to_datetime(end_date))
]

if assets.empty:
    st.warning("No data in selected range.")
    st.stop()

returns = assets.pct_change().dropna()

# ==========================================
# 1️⃣ ROLLING CORRELATION NETWORK DENSITY
# ==========================================

st.subheader("Rolling Network Density (Contagion Index)")

window = st.slider("Rolling Window", 20, 120, 60)

rolling_density = []

for i in range(window, len(returns)):
    corr = returns.iloc[i-window:i].corr()
    avg_corr = np.mean(np.abs(corr.values[np.triu_indices_from(corr, k=1)]))
    rolling_density.append(avg_corr)

density_index = pd.Series(
    rolling_density,
    index=returns.index[window:]
)

fig_density = px.line(density_index,
                      template="plotly_dark",
                      labels={"value":"Average Correlation"},
                      title="Contagion Intensity Over Time")

st.plotly_chart(fig_density, use_container_width=True)

# ==========================================
# 2️⃣ CURRENT CORRELATION HEATMAP
# ==========================================

st.subheader("Current Correlation Structure")

corr_matrix = returns.tail(window).corr()

fig_heat = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    template="plotly_dark"
)

st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================
# 3️⃣ NETWORK CENTRALITY
# ==========================================

st.subheader("Asset Centrality Score")

centrality = corr_matrix.abs().mean()
centrality = centrality.sort_values(ascending=False)

fig_cent = px.bar(
    centrality,
    template="plotly_dark",
    title="Systemic Importance (Average Absolute Correlation)"
)

st.plotly_chart(fig_cent, use_container_width=True)

# ==========================================
# 4️⃣ DIVERSIFICATION RATIO
# ==========================================

st.subheader("Diversification Ratio")

vol = returns.std()
weighted_vol = vol.mean()
portfolio_vol = returns.mean(axis=1).std()

div_ratio = weighted_vol / portfolio_vol

st.metric("Diversification Ratio", f"{div_ratio:.2f}")

# ==========================================
# 5️⃣ EIGENVALUE CONCENTRATION
# ==========================================

st.subheader("Eigenvalue Concentration")

eigenvalues = np.linalg.eigvals(corr_matrix)
eigenvalues = np.sort(eigenvalues)[::-1]

fig_eig = px.bar(
    x=range(1, len(eigenvalues)+1),
    y=eigenvalues,
    template="plotly_dark",
    labels={"x":"Eigenvalue Rank","y":"Magnitude"}
)

st.plotly_chart(fig_eig, use_container_width=True)

# ==========================================
# 6️⃣ TAIL DEPENDENCE APPROXIMATION
# ==========================================

st.subheader("Tail Dependence Approximation")

threshold = returns.quantile(0.05)

tail_events = returns < threshold
tail_corr = tail_events.corr()

fig_tail = px.imshow(
    tail_corr,
    text_auto=True,
    color_continuous_scale="Inferno",
    template="plotly_dark",
    title="Lower Tail Co-Movement"
)

st.plotly_chart(fig_tail, use_container_width=True)

# ==========================================
# 7️⃣ VOLATILITY VS CORRELATION
# ==========================================

st.subheader("Volatility vs Contagion")

rolling_vol = returns.rolling(window).std().mean(axis=1)

combined = pd.concat([rolling_vol, density_index], axis=1).dropna()
combined.columns = ["Volatility","Contagion"]

fig_scatter = px.scatter(
    combined,
    x="Volatility",
    y="Contagion",
    template="plotly_dark",
    trendline="ols"
)

st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# 8️⃣ NETWORK STRESS REGIME CLASSIFICATION
# ==========================================

st.subheader("Systemic Stress Regimes")

high = density_index.quantile(0.75)
low = density_index.quantile(0.25)

stress_regime = pd.cut(density_index,
                       bins=[-np.inf, low, high, np.inf],
                       labels=["Low Stress","Medium Stress","High Stress"])

fig_stress = px.histogram(
    stress_regime,
    template="plotly_dark"
)

st.plotly_chart(fig_stress, use_container_width=True)

# ==========================================
# 9️⃣ CORRELATION BREAKDOWN DETECTOR
# ==========================================

st.subheader("Correlation Spike Detection")

spikes = density_index[density_index > density_index.quantile(0.9)]

fig_spikes = px.scatter(
    x=spikes.index,
    y=spikes.values,
    template="plotly_dark",
    title="Top 10% Contagion Spikes"
)

st.plotly_chart(fig_spikes, use_container_width=True)

# ==========================================
# 🔟 SYSTEMIC RISK SCORE
# ==========================================

st.subheader("Systemic Risk Score")

risk_score = (density_index.iloc[-1] + rolling_vol.iloc[-1]) / 2
st.metric("Current Systemic Risk Level", f"{risk_score:.3f}")

st.markdown("""
### Interpretation

• Rising density = correlation convergence  
• Centrality identifies fragile nodes  
• Eigenvalue dominance = factor concentration  
• Tail heatmap reveals crisis co-movement  
• Volatility–contagion link validates stress transmission  

This layer detects when diversification is collapsing.
""")
