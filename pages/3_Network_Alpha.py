import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("Contagion & Systemic Risk Network")

data = load_all()

# --------------------------
# AUTO-DETECT PRICE DATASET
# --------------------------

assets = None
max_cols = 0

for key in data:
    df = data[key]
    if isinstance(df, pd.DataFrame) and df.shape[1] > max_cols:
        assets = df
        max_cols = df.shape[1]

if assets is None:
    st.error("No suitable price dataset found in /data folder.")
    st.stop()

assets = assets.apply(pd.to_numeric, errors="coerce").dropna()

# --------------------------
# Timeline
# --------------------------

min_date = assets.index.min()
max_date = assets.index.max()

if pd.isna(min_date) or pd.isna(max_date):
    st.error("Invalid date index.")
    st.stop()

start_date = st.date_input("Start Date", min_date.date())
end_date = st.date_input("End Date", max_date.date())

end_date = st.date_input("End Date", assets.index.max())

assets = assets.loc[
    (assets.index >= pd.to_datetime(start_date)) &
    (assets.index <= pd.to_datetime(end_date))
]

if assets.empty:
    st.warning("No data in selected period.")
    st.stop()

returns = assets.pct_change().dropna()

# --------------------------
# Rolling Window
# --------------------------

window = st.slider("Rolling Window (days)", 30, 252, 60)
threshold = st.slider("Correlation Threshold", 0.3, 0.9, 0.6)

# --------------------------
# Average Correlation
# --------------------------

avg_corr_series = []

for i in range(window, len(returns)):
    corr_matrix = returns.iloc[i-window:i].corr()
    upper = corr_matrix.values[np.triu_indices_from(corr_matrix, 1)]
    avg_corr_series.append(np.mean(upper))

avg_corr = pd.Series(avg_corr_series, index=returns.index[window:])

st.subheader("Average Pairwise Correlation")

fig_avg = px.line(avg_corr, template="plotly_dark")
st.plotly_chart(fig_avg, use_container_width=True)

# --------------------------
# Network Density
# --------------------------

density_series = []

for i in range(window, len(returns)):
    corr_matrix = returns.iloc[i-window:i].corr()
    G = nx.Graph()

    for a in corr_matrix.columns:
        for b in corr_matrix.columns:
            if a != b and abs(corr_matrix.loc[a, b]) > threshold:
                G.add_edge(a, b)

    density_series.append(nx.density(G))

density = pd.Series(density_series, index=returns.index[window:])

st.subheader("Network Density (Contagion Intensity)")
fig_density = px.line(density, template="plotly_dark")
st.plotly_chart(fig_density, use_container_width=True)

# --------------------------
# Snapshot Selector
# --------------------------

st.subheader("Network Snapshot")

snapshot_date = st.selectbox(
    "Select Snapshot Date",
    options=list(returns.index[window:])
)

corr_snapshot = returns.loc[:snapshot_date].tail(window).corr()

G = nx.Graph()

for a in corr_snapshot.columns:
    for b in corr_snapshot.columns:
        if a != b and abs(corr_snapshot.loc[a, b]) > threshold:
            G.add_edge(a, b)

pos = nx.spring_layout(G, seed=42)

edge_x, edge_y = [], []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

node_x, node_y = [], []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

fig_net = go.Figure()

fig_net.add_trace(go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=1),
    mode='lines',
    showlegend=False
))

fig_net.add_trace(go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers+text',
    text=list(G.nodes()),
    textposition="top center",
    marker=dict(size=10),
    showlegend=False
))

fig_net.update_layout(template="plotly_dark")
st.plotly_chart(fig_net, use_container_width=True)

# --------------------------
# Centrality
# --------------------------

st.subheader("Eigenvector Centrality")

if len(G.nodes) > 0:
    centrality = nx.eigenvector_centrality_numpy(G)
    centrality_df = pd.DataFrame.from_dict(
        centrality,
        orient='index',
        columns=["Centrality"]
    ).sort_values("Centrality", ascending=False)

    fig_cent = px.bar(centrality_df, y="Centrality", template="plotly_dark")
    st.plotly_chart(fig_cent, use_container_width=True)

    st.metric("Systemic Concentration Risk",
              f"{centrality_df['Centrality'].max():.4f}")
else:
    st.info("No edges above threshold.")

# --------------------------
# Heatmap
# --------------------------

st.subheader("Correlation Heatmap")

fig_heat = px.imshow(
    corr_snapshot,
    template="plotly_dark",
    color_continuous_scale="RdBu"
)

st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("""
### Interpretation

• Rising average correlation signals diversification breakdown  
• Network density measures contagion tightening  
• Centrality identifies systemic dominant assets  
• Threshold slider controls stress sensitivity  
• Timeline selector allows crisis inspection  
""")
