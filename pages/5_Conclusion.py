import streamlit as st
import pandas as pd
import numpy as np
from utils.load_data import load_all

st.set_page_config(layout="wide")
st.title("System Conclusion & Capital Survival Assessment")

data = load_all()

regime_probs = data.get("regime_probs", pd.DataFrame())
garch = data.get("garch", pd.DataFrame())
contagion = data.get("contagion", pd.DataFrame())
assets = data.get("market_data", pd.DataFrame())

st.markdown("""
## Executive Summary

This system integrates macro regime detection, regime-dependent volatility modeling,
systemic contagion monitoring, and tail-risk constrained allocation.

The objective is capital preservation across structural uncertainty.
""")

st.markdown("---")

# ================================
# Regime Assessment
# ================================

st.subheader("Macro Regime Assessment")

if not regime_probs.empty:
    latest_regime = regime_probs.iloc[-1]
    dominant = latest_regime.idxmax()
    confidence = latest_regime.max()

    st.metric("Current Regime", dominant)
    st.metric("Confidence Level", round(confidence, 3))
else:
    st.warning("Regime data unavailable.")

st.markdown("---")

# ================================
# Volatility Assessment
# ================================

st.subheader("Volatility Risk Assessment")

if not garch.empty:
    current_vol = garch.iloc[-1, 0]
    vol_percentile = (garch.iloc[:, 0] < current_vol).mean()

    st.metric("Current Forecast Volatility", round(current_vol, 4))
    st.metric("Volatility Percentile", round(vol_percentile, 2))
else:
    st.warning("Volatility data unavailable.")

st.markdown("---")

# ================================
# Contagion Assessment
# ================================

st.subheader("Systemic Contagion Assessment")

if not contagion.empty:
    current_contagion = contagion.iloc[-1, 0]
    contagion_percentile = (contagion.iloc[:, 0] < current_contagion).mean()

    st.metric("Contagion Index", round(current_contagion, 3))
    st.metric("Contagion Percentile", round(contagion_percentile, 2))
else:
    st.warning("Contagion data unavailable.")

st.markdown("---")

# ================================
# Tail Risk Analysis
# ================================

st.subheader("Tail Risk Diagnostics")

if not assets.empty:
    returns = assets.pct_change().dropna().mean(axis=1)

    mean_r = returns.mean()
    std_r = returns.std()

    skew = ((returns - mean_r) ** 3).mean() / (std_r ** 3)
    kurt = ((returns - mean_r) ** 4).mean() / (std_r ** 4)

    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Skewness", round(skew, 3))
    col2.metric("Kurtosis", round(kurt, 3))
    col3.metric("VaR (95%)", round(var_95, 4))
    col4.metric("CVaR (95%)", round(cvar_95, 4))
else:
    st.warning("Asset return data unavailable.")

st.markdown("---")

# ================================
# Structural Scorecard
# ================================

st.subheader("Structural Stability Scorecard")

score = 0

if not regime_probs.empty and confidence > 0.6:
    score += 1
if not garch.empty and vol_percentile < 0.7:
    score += 1
if not contagion.empty and contagion_percentile < 0.7:
    score += 1
if not assets.empty and kurt < 5:
    score += 1

st.metric("System Stability Score (0-4)", score)

if score >= 3:
    st.success("System environment structurally stable.")
elif score == 2:
    st.warning("Mixed structural conditions.")
else:
    st.error("Elevated systemic risk conditions.")

st.markdown("""
## Final Interpretation

• Regime confidence defines macro clarity  
• Volatility percentile defines stress intensity  
• Contagion percentile defines systemic spread risk  
• Kurtosis defines fat-tail severity  

This model prioritizes survival over speculative return maximization.
""")
