import streamlit as st
import pandas as pd
from utils.load_data import load_all


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Regime-Aware Capital Allocation System",
    layout="wide"
)

st.title("Regime-Aware Tail-Risk Constrained Portfolio System")

# =====================================================
# SIDEBAR – INVESTOR PROFILE
# =====================================================

st.sidebar.header("Investor Profile")

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 18, 90, 30)
capital = st.sidebar.number_input("Capital ($)", min_value=1000.0, value=100000.0)
risk_threshold = st.sidebar.slider("Risk Tolerance (0-1)", 0.0, 1.0, 0.5)
horizon = st.sidebar.selectbox(
    "Investment Horizon",
    ["Short Term", "Medium Term", "Long Term"]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Governance Settings")
enable_regime = st.sidebar.checkbox("Enable Regime Overlay", True)
enable_vol = st.sidebar.checkbox("Enable Volatility Scaling", True)
enable_contagion = st.sidebar.checkbox("Enable Contagion Filter", True)

# =====================================================
# LOAD DATA
# =====================================================

data = load_all()

regime_probs = data.get("regime_probs", pd.DataFrame())
garch = data.get("garch", pd.DataFrame())
vix = data.get("vix", pd.DataFrame())
contagion = data.get("contagion", pd.DataFrame())
market_data = data.get("market_data", pd.DataFrame())

# =====================================================
# SYSTEM OVERVIEW
# =====================================================

st.markdown("""
### System Purpose

This platform implements a hierarchical capital allocation framework designed to:

1. Detect macroeconomic regimes via probabilistic models.
2. Estimate regime-dependent volatility (MS-GARCH).
3. Monitor systemic contagion dynamics.
4. Allocate capital under strict tail-risk constraints.
5. Enforce governance-first risk control.

The objective is not prediction.  
The objective is **capital preservation across structural regime shifts**.
""")

st.markdown("---")

# =====================================================
# LIVE SYSTEM STATUS
# =====================================================

st.subheader("Current System State")

col1, col2, col3 = st.columns(3)

# Regime
if not regime_probs.empty:
    latest = regime_probs.iloc[-1]
    dominant_regime = latest.idxmax()
    confidence = latest.max()
    col1.metric("Dominant Regime", dominant_regime)
    col1.metric("Regime Confidence", f"{confidence:.2f}")
else:
    col1.warning("Regime data unavailable.")

# Volatility
if not garch.empty:
    current_vol = garch.iloc[-1, 0]
    col2.metric("Volatility Forecast", f"{current_vol:.4f}")
else:
    col2.warning("Volatility layer unavailable.")

# Contagion
if not contagion.empty:
    contagion_score = contagion.iloc[-1, 0]
    col3.metric("Contagion Index", f"{contagion_score:.3f}")
else:
    col3.warning("Contagion layer unavailable.")

st.markdown("---")

# =====================================================
# SYSTEM CONFIDENCE SCORE
# =====================================================

st.subheader("System Confidence Score")

score = 0
layers = 3

if not regime_probs.empty:
    score += 1
if not garch.empty:
    score += 1
if not contagion.empty:
    score += 1

confidence_percent = int((score / layers) * 100)

st.progress(confidence_percent / 100)

st.markdown(f"**Infrastructure Completeness:** {confidence_percent}%")

if confidence_percent == 100:
    st.success("All core layers operational.")
elif confidence_percent >= 60:
    st.warning("Partial system functionality.")
else:
    st.error("Critical layers missing.")

st.markdown("---")

# =====================================================
# INVESTOR–SYSTEM FIT ANALYSIS
# =====================================================

st.subheader("Investor Risk Alignment")

if not garch.empty:
    vol_level = garch.iloc[-1, 0]

    if risk_threshold < 0.3 and vol_level > 0.02:
        st.error("Current volatility exceeds conservative tolerance.")
    elif risk_threshold > 0.7:
        st.info("Aggressive profile aligned with dynamic allocation.")
    else:
        st.success("Risk profile moderately aligned with system state.")
else:
    st.warning("Cannot assess alignment without volatility data.")

st.markdown("---")

# =====================================================
# ARCHITECTURE EXPLANATION
# =====================================================

st.subheader("Architecture Hierarchy")

st.markdown("""
**Layer 1 — Macro Regime Detection**  
Hidden Markov Models identify probabilistic market states.

**Layer 2 — Volatility Estimation**  
MS-GARCH captures regime-dependent variance clustering.

**Layer 3 — Contagion Monitoring**  
Dynamic dependency structures detect systemic amplification.

**Layer 4 — Risk-Constrained Allocation**  
Capital is allocated under CVaR and drawdown constraints.

Higher layers override lower layers.  
Governance always dominates alpha logic.
""")

st.markdown("---")

# =====================================================
# DATA DIAGNOSTICS
# =====================================================

st.subheader("Data Diagnostics")

colA, colB = st.columns(2)

colA.write("Regime Data:", "Available" if not regime_probs.empty else "Missing")
colA.write("Volatility Data:", "Available" if not garch.empty else "Missing")
colA.write("Contagion Data:", "Available" if not contagion.empty else "Missing")

colB.write("VIX Data:", "Available" if not vix.empty else "Missing")
colB.write("Market Data:", "Available" if not market_data.empty else "Missing")

st.markdown("---")

# =====================================================
# NAVIGATION GUIDE
# =====================================================

st.subheader("Navigation")

st.markdown("""
Use sidebar pages to explore:

• **Macro Regime** → Regime heatmaps and crisis detection  
• **Volatility** → MS-GARCH clustering vs VIX  
• **Network Alpha** → Contagion networks and systemic risk  
• **Portfolio Allocation** → Risk-constrained optimization  
• **Conclusion** → Model limitations and governance review  

This system demonstrates hierarchical risk control under structural uncertainty.
""")

st.caption("Institutional-grade risk-aware capital allocation research framework.")
