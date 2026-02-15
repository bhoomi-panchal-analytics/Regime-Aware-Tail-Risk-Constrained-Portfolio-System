import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("Contagion Network & Systemic Risk Analytics")

# ==========================
# Load Data
# ==========================

data = load_all()
assets = data.get("market_data", pd.DataFrame())

if assets.empty:
    st.error("Market data missing.")
    st.stop()

assets = assets.apply(pd.to_numeric, errors="coerce").dropna()
returns = assets.pct_change().dropna()

# ==========================
# Timeline Selection
# ==========================

start_date = st.date_input("Start Date", returns.index.min())
end_date = st.date_input("End Date", returns.index.max())

returns = returns.loc[
    (returns.index >= pd.to_datetime(start_date)) &
    (returns.index <= pd.to_datetime(end_date))
]

if returns.empty:
    st.warning("No data in selected range.")
    st.stop()

# ==========================
# Rolling Correlation
# ==========================

window = st.slider("Rolling Window (days)", 30, 252, 60)

rolling_corr = returns.rolling(window).corr()

# ==========================
# Average Correlation Over Time
# ==========================

avg_corr_series = []

for date in returns.index[window:]:
    corr_matrix = returns.loc[:date].tail(window).corr()
    upper_tri = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]
    avg_corr_series.append(np.mean(upper_tri))

avg_corr = pd.Series(avg_corr_series, index=returns.index[window:])

st.subheader("Average Pairwise Correlation")

fig_avg = px.line(avg_corr, template="plotly_dark")
st.plotly_chart(fig_avg, use_container_width=True)

# ==========================
# Network Density Over Time
# ==========================

density_series = []

threshold = st.slider("Correlation Threshold", 0.3, 0.9, 0.6)

for date in returns.index[window:]:
    corr_matrix = returns.loc[:date].tail(window).corr()
    G = nx.Graph()

    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i != j and abs(corr_matrix.loc[i, j]) > threshold:
                G.add_edge(i, j)

    density_series.append(nx.density(G))

density = pd.Series(density_series, index=returns.index[window:])

st.subheader("Network Density (Contagion Intensity)")

fig_density = px.line(density, template="plotly_dark")
st.plotly_chart(fig_density, use_container_width=True)

# ==========================
# Snapshot Network Graph
# ==========================

st.subheader("Network Snapshot")

selected_date = st.selectbox(
    "Select Date for Network Snapshot",
    options=list(returns.index[window:])
)

corr_snapshot = returns.loc[:selected_date].tail(window).corr()

G = nx.Graph()

for i in corr_snapshot.columns:
    for j in corr_snapshot.columns:
        if i != j and abs(corr_snapshot.loc[i, j]) > threshold:
            G.add_edge(i, j, weight=corr_snapshot.loc[i, j])

pos = nx.spring_layout(G, seed=42)

edge_x = []
edge_y = []

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

node_x = []
node_y = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

fig_network = go.Figure()

fig_network.add_trace(go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=1),
    mode='lines',
    showlegend=False
))

fig_network.add_trace(go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers+text',
    text=list(G.nodes()),
    textposition="top center",
    marker=dict(size=10),
    showlegend=False
))

fig_network.update_layout(template="plotly_dark")
st.plotly_chart(fig_network, use_container_width=True)

# ==========================
# Centrality Ranking
# ==========================

st.subheader("Eigenvector Centrality")

centrality = nx.eigenvector_centrality_numpy(G)
centrality_df = pd.DataFrame.from_dict(centrality, orient='index', columns=['Centrality'])
centrality_df = centrality_df.sort_values("Centrality", ascending=False)

fig_cent = px.bar(
    centrality_df,
    y="Centrality",
    template="plotly_dark"
)
st.plotly_chart(fig_cent, use_container_width=True)

# ==========================
# Degree Distribution
# ==========================

st.subheader("Degree Distribution")

degrees = [val for (node, val) in G.degree()]
fig_deg = px.histogram(degrees, nbins=10, template="plotly_dark")
st.plotly_chart(fig_deg, use_container_width=True)

# ==========================
# Correlation Heatmap Snapshot
# ==========================

st.subheader("Correlation Heatmap (Snapshot)")

fig_heat = px.imshow(
    corr_snapshot,
    color_continuous_scale="RdBu",
    template="plotly_dark"
)
st.plotly_chart(fig_heat, use_container_width=True)

# ==========================
# Concentration Risk Metric
# ==========================

concentration = centrality_df["Centrality"].max()

st.subheader("Systemic Concentration Risk")
st.metric("Max Centrality", f"{concentration:.4f}")

st.markdown("""
### Systemic Risk Interpretation

• Rising average correlation indicates diversification breakdown  
• Network density measures contagion tightening  
• Centrality ranking identifies dominant systemic nodes  
• High concentration suggests fragility  
• Threshold control allows stress sensitivity adjustment  
""")
