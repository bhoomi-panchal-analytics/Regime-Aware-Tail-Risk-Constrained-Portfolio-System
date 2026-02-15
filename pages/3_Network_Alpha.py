import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Systemic Contagion & Network Diagnostics")

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
# BASIC CLEANING
# =========================

# If Date column exists, use it
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date")

df.index = pd.to_datetime(df.index, errors="coerce")
df = df.dropna()

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Keep only columns with actual variation
valid_cols = []
for col in df.columns:
    if df[col].notna().sum() > 20 and df[col].std() > 0:
        valid_cols.append(col)

df = df[valid_cols]

if df.shape[1] < 2:
    st.error("Not enough valid asset columns after cleaning.")
    st.stop()

# Fill missing safely
df = df.fillna(0)

st.success("Data ready for network diagnostics.")

# =========================
# CORRELATION MATRIX
# =========================

st.subheader("1. Correlation Heatmap")

corr_matrix = df.corr()

fig1 = px.imshow(
    corr_matrix,
    zmin=-1,
    zmax=1,
    color_continuous_scale="RdBu",
)

fig1.update_layout(template="plotly_dark")
st.plotly_chart(fig1, use_container_width=True)

# =========================
# CONTAGION INDEX
# =========================

st.subheader("2. Contagion Index (Average Absolute Correlation)")

upper_vals = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]
contagion = np.mean(np.abs(upper_vals))

st.metric("Contagion Level", round(contagion, 3))

# =========================
# CORRELATION DISTRIBUTION
# =========================

st.subheader("3. Correlation Distribution")

fig2 = px.histogram(
    upper_vals,
    nbins=30,
    template="plotly_dark",
    title="Pairwise Correlation Distribution"
)

st.plotly_chart(fig2, use_container_width=True)

# =========================
# SYSTEMIC CENTRALITY
# =========================

st.subheader("4. Systemic Centrality")

centrality = corr_matrix.abs().mean().sort_values(ascending=False)

fig3 = px.bar(
    centrality,
    template="plotly_dark",
    title="Average Absolute Correlation per Asset"
)

st.plotly_chart(fig3, use_container_width=True)

# =========================
# EIGENVALUE SPECTRUM
# =========================

st.subheader("5. Eigenvalue Spectrum")

try:
    eigvals = np.linalg.eigvals(corr_matrix.values)
    eigvals = np.real(eigvals)

    fig4 = px.bar(
        eigvals,
        template="plotly_dark",
        title="Eigenvalue Magnitudes"
    )

    st.plotly_chart(fig4, use_container_width=True)

    systemic_share = eigvals.max() / eigvals.sum()
    st.metric("Systemic Concentration", round(systemic_share, 3))

except:
    st.warning("Eigenvalue decomposition unstable.")

# =========================
# DIVERSIFICATION RATIO
# =========================

st.subheader("6. Diversification Ratio")

vol = df.std()
cov = df.cov()

weights = np.ones(len(vol)) / len(vol)

portfolio_vol = np.sqrt(weights @ cov.values @ weights)
weighted_vol = weights @ vol.values

div_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 0

st.metric("Diversification Ratio", round(div_ratio, 3))

st.markdown("""
### Interpretation

• High contagion → diversification breakdown  
• Large first eigenvalue → dominant systemic factor  
• Diversification ratio close to 1 → assets moving together  
• Wide correlation distribution → heterogeneous structure  
""")
