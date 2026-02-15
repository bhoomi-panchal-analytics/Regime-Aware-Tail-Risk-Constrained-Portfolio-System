import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from utils.load_data import load_all

st.title("Contagion Network & Survival vs Alpha")

data = load_all()
contagion = data["contagion"]
metrics = data["metrics"]

st.subheader("Contagion Network Snapshot")

date = st.selectbox("Select Date", contagion.index)

corr_matrix = contagion.loc[date].values.reshape(5,5)
assets = ["SPY","TLT","GLD","DBC","UUP"]

G = nx.Graph()

for i in range(len(assets)):
    G.add_node(assets[i])

for i in range(len(assets)):
    for j in range(i+1, len(assets)):
        weight = corr_matrix[i][j]
        if abs(weight) > 0.5:
            G.add_edge(assets[i], assets[j], weight=weight)

pos = nx.spring_layout(G)

edge_x = []
edge_y = []

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=assets,
    textposition="top center",
    marker=dict(size=20)
)

fig = go.Figure(data=[edge_trace, node_trace])
st.plotly_chart(fig, use_container_width=True)

st.subheader("Survival vs Alpha Matrix")

st.write("Top: Survival Metrics | Bottom: Alpha Metrics")

st.dataframe(metrics)
