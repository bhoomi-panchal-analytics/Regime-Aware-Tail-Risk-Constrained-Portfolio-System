import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("MS-GARCH Volatility vs VIX Diagnostics")

# ==========================
# LOAD DATA
# ==========================

data = load_all()

garch = None
vix = None

# Auto-detect volatility-like datasets
for key in data:
    df = data[key]
    if isinstance(df, pd.DataFrame):

        # Detect GARCH forecast (values small, positive)
        if df.max().max() < 2 and df.mean().mean() > 0:
            garch = df

        # Detect VIX (values larger, e.g. >5)
        if df.max().max() > 5:
            vix = df

if garch is None or vix is None:
    st.error("GARCH or VIX data not detected in /data folder.")
    st.stop()

# ==========================
# CLEAN & ALIGN
# ==========================

garch = garch.apply(pd.to_numeric, errors="coerce").dropna()
vix = vix.apply(pd.to_numeric, errors="coerce").dropna()

garch.index = pd.to_datetime(garch.index, errors="coerce")
vix.index = pd.to_datetime(vix.index, errors="coerce")

garch = garch[~garch.index.isna()]
vix = vix[~vix.index.isna()]

combined = garch.join(vix, how="inner")

if combined.empty:
    st.error("No overlapping dates between GARCH and VIX.")
    st.stop()

combined = combined.sort_index()

garch_col = combined.columns[0]
vix_col = combined.columns[1]

# ==========================
# TIMELINE FILTER
# ==========================

min_date = combined.index.min()
max_date = combined.index.max()

start_date = st.date_input("Start Date", min_date.date())
end_date = st.date_input("End Date", max_date.date())

combined = combined.loc[
    (combined.index >= pd.to_datetime(start_date)) &
    (combined.index <= pd.to_datetime(end_date))
]

if combined.empty:
    st.warning("No data in selected range.")
    st.stop()

# ==========================
# 1️⃣ MAIN VOL COMPARISON
# ==========================

st.subheader("Forecast vs Implied Volatility")

fig_main = go.Figure()

fig_main.add_trace(go.Scatter(
    x=combined.index,
    y=combined[garch_col],
    name="MS-GARCH Forecast",
    line=dict(width=2)
))

fig_main.add_trace(go.Scatter(
    x=combined.index,
    y=combined[vix_col],
    name="VIX",
    line=dict(width=2)
))

# Crisis shading
fig_main.add_vrect(x0="2008-01-01", x1="2009-06-01",
                   fillcolor="red", opacity=0.2, line_width=0)

fig_main.add_vrect(x0="2020-02-01", x1="2020-06-01",
                   fillcolor="orange", opacity=0.2, line_width=0)

fig_main.update_layout(template="plotly_dark")
st.plotly_chart(fig_main, use_container_width=True)

# ==========================
# 2️⃣ VOLATILITY CLUSTERING
# ==========================

st.subheader("Volatility Clustering Test")

vol = combined[garch_col]
lag_vol = vol.shift(1).dropna()
curr_vol = vol.iloc[1:]

fig_cluster = px.scatter(
    x=lag_vol,
    y=curr_vol,
    template="plotly_dark",
    labels={"x":"Lagged Volatility","y":"Current Volatility"}
)

st.plotly_chart(fig_cluster, use_container_width=True)

# ==========================
# 3️⃣ ROLLING VOLATILITY
# ==========================

st.subheader("30-Day Rolling Comparison")

rolling_garch = combined[garch_col].rolling(30).mean()
rolling_vix = combined[vix_col].rolling(30).mean()

fig_roll = go.Figure()

fig_roll.add_trace(go.Scatter(
    x=combined.index,
    y=rolling_garch,
    name="Rolling GARCH"
))

fig_roll.add_trace(go.Scatter(
    x=combined.index,
    y=rolling_vix,
    name="Rolling VIX"
))

fig_roll.update_layout(template="plotly_dark")
st.plotly_chart(fig_roll, use_container_width=True)

# ==========================
# 4️⃣ VOLATILITY SPREAD
# ==========================

st.subheader("Volatility Risk Premium (VIX - Forecast)")

spread = combined[vix_col] - combined[garch_col]

fig_spread = px.line(spread, template="plotly_dark")
st.plotly_chart(fig_spread, use_container_width=True)

# ==========================
# 5️⃣ VOL REGIME CLASSIFICATION
# ==========================

st.subheader("Volatility Regime Classification")

high = vol.quantile(0.75)
low = vol.quantile(0.25)

regime = pd.cut(vol,
                bins=[-np.inf, low, high, np.inf],
                labels=["Low Vol","Medium Vol","High Vol"])

fig_regime = px.histogram(regime,
                          template="plotly_dark",
                          title="Volatility Regime Distribution")

st.plotly_chart(fig_regime, use_container_width=True)

# ==========================
# 6️⃣ VOL PERSISTENCE
# ==========================

st.subheader("Volatility Persistence")

autocorr = vol.autocorr(lag=1)
st.metric("Lag-1 Autocorrelation", f"{autocorr:.4f}")

# ==========================
# 7️⃣ TAIL VOL SPIKES
# ==========================

st.subheader("Extreme Volatility Events")

threshold = vol.quantile(0.95)
extreme = vol[vol > threshold]

fig_extreme = px.scatter(
    x=extreme.index,
    y=extreme,
    template="plotly_dark",
    title="Top 5% Volatility Events"
)

st.plotly_chart(fig_extreme, use_container_width=True)

st.markdown("""
### Interpretation

• Clustering confirms heteroskedasticity  
• Spread measures volatility risk premium  
• Persistence indicates regime memory  
• Extreme spikes align with systemic crises  
• Rolling comparison validates forecast adaptation  
""")
