import streamlit as st
import pandas as pd
from utils.load_data import load_all

st.set_page_config(
    page_title="Regime-Aware Risk Allocation System",
    layout="wide"
)

st.title("Regime-Aware Tail-Risk Constrained Portfolio System")

# ======================================
# LOAD DATA
# ======================================

data = load_all()

regime_probs = data.get("regime_probs", pd.DataFrame())
garch = data.get("garch", pd.DataFrame())
contagion = data.get("contagion", pd.DataFrame())

# ======================================
# SYSTEM OVERVIEW
# ======================================

st.markdown("""
### System Objective

This platform demonstrates a hierarchical capital allocation framework designed to:

- Detect macroeconomic regimes using unsupervised models (HMM)
- Estimate regime-dependent volatility using MS-GARCH
- Monitor systemic contagion risk
- Allocate capital under strict tail-risk constraints
- Preserve capital during regime shifts

The system does **not** attempt short-term prediction.  
It focuses on **risk adaptation and survival across market states**.
""")

st.markdown("---")

# ======================================
# LIVE SYSTEM STATUS
# ======================================

st.subheader("Current System State")

col1, col2, col3 = st.columns(3)

# --- Regime Status ---
if not regime_probs.empty:
    latest_regime = regime_probs.iloc[-1]
    dominant_regime = latest_regime.idxmax()
    regime_prob = latest_regime.max()
    col1.metric("Dominant Regime", dominant_regime)
    col1.metric("Regime Confidence", round(regime_prob, 3))
else:
    col1.warning("Regime data unavailable.")

# --- Volatility Status ---
if not garch.empty:
    latest_vol = garch.iloc[-1, 0]
    col2.metric("Current Volatility Forecast", round(latest_vol, 4))
else:
    col2.warning("Volatility data unavailable.")

# --- Contagion Status ---
if not contagion.empty:
    latest_contagion = contagion.iloc[-1, 0]
    col3.metric("Contagion Index", round(latest_contagion, 3))
else:
    col3.warning("Contagion data unavailable.")

st.markdown("---")

# ======================================
# INTERACTIVE CONTROL PANEL
# ======================================

st.subheader("Interactive System Controls")

risk_mode = st.radio(
    "Risk Management Mode",
    ["Conservative", "Balanced", "Aggressive"]
)

regime_overlay = st.checkbox("Enable Regime Overlay Logic", value=True)
volatility_scaling = st.checkbox("Enable Volatility Scaling", value=True)
contagion_filter = st.checkbox("Enable Contagion Risk Filter", value=True)

st.markdown(f"""
**Selected Mode:** {risk_mode}

- Conservative → Strong drawdown control  
- Balanced → Risk-adjusted optimization  
- Aggressive → Higher capital exposure  

Regime Overlay: {regime_overlay}  
Volatility Scaling: {volatility_scaling}  
Contagion Filter: {contagion_filter}
""")

st.markdown("---")

# ======================================
# SYSTEM ARCHITECTURE EXPLANATION
# ======================================

st.subheader("System Architecture")

st.markdown("""
**Layer 1 – Macro Regime Detection**  
Autoencoder + Hidden Markov Model infer latent market states.

**Layer 2 – Volatility Modeling**  
MS-GARCH estimates regime-dependent conditional variance.

**Layer 3 – Contagion Monitoring**  
Dynamic correlation structures detect systemic stress.

**Layer 4 – Risk-Constrained Allocation**  
Optimization engine allocates capital under CVaR and drawdown limits.

Higher layers override lower ones.  
Risk governance always dominates allocation logic.
""")

st.markdown("---")

# ======================================
# SYSTEM HEALTH CHECK
# ======================================

st.subheader("System Health Check")

health_status = []

if regime_probs.empty:
    health_status.append("Regime layer missing")
if garch.empty:
    health_status.append("Volatility layer missing")
if contagion.empty:
    health_status.append("Contagion layer missing")

if len(health_status) == 0:
    st.success("All layers operational.")
else:
    for issue in health_status:
        st.error(issue)

st.markdown("---")

# ======================================
# USER NAVIGATION GUIDE
# ======================================

st.subheader("Navigation Guide")

st.markdown("""
Use the sidebar to explore:

- **Macro Regime** → View latent regime probability structure and crisis detection  
- **Volatility** → Compare MS-GARCH forecasts vs market volatility  
- **Network Alpha** → Inspect contagion dynamics and systemic stress  
- **Portfolio Allocation** → Construct optimized portfolios under risk constraints  

This dashboard demonstrates capital preservation under structural uncertainty.
""")

st.markdown("---")

st.caption("Designed for institutional-grade risk-aware capital allocation research.")
