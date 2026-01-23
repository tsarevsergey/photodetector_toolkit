
import streamlit as st

st.set_page_config(
    page_title="Noise Spectral Analyzer",
    page_icon="📉",
    layout="wide"
)

st.title("Noise Spectral Analyzer & Photodetector Characterization")

import atexit

# Global cleanup to handle server shutdowns or reloads
def cleanup():
    # Attempt to close SMU
    if 'smu' in st.session_state and st.session_state.smu:
        try:
            st.session_state.smu.disconnect()
            print("Cleanup: Disconnected SMU")
        except:
            pass
            
    # Attempt to close Scope
    if 'scope' in st.session_state and st.session_state.scope:
        try:
            st.session_state.scope.disconnect()
            print("Cleanup: Disconnected Scope")
        except:
            pass

atexit.register(cleanup)

st.markdown("""
### 🔬 Professional Photodetector Characterization Suite

Welcome to the **Noise Spectral Analyzer**. This integrated software platform provides a complete workflow for characterizing high-performance photodetectors, from hardware validation to advanced noise analysis and linear dynamic range (LDR) extraction.

#### 🔭 Scientific Foundation

The analysis performed by this software is based on several key physical principles:

1.  **Johnson-Nyquist Noise**: The fundamental thermal noise limit of any resistive element (e.g., your load resistor).
    $$V_n = \sqrt{4 k_B T R} \quad [\text{V}/\sqrt{\text{Hz}}]$$
2.  **Noise Equivalent Power (NEP)**: The optical power that produces a signal equal to the noise floor at a 1 Hz bandwidth.
    $$NEP = \frac{i_n}{R(\lambda)} \quad [\text{W}/\sqrt{\text{Hz}}]$$
3.  **Specific Detectivity ($D^*$)**: Area-normalized sensitivity metric, allowing comparison between different detector sizes.
    $$D^* = \frac{\sqrt{A}}{NEP} \quad [\text{cm} \cdot \sqrt{\text{Hz}} / \text{W} \text{ (Jones)}]$$
4.  **Linear Dynamic Range (LDR)**: The range over which the detector response remains linear, typically expressed in decibels.
    $$LDR = 20 \cdot \log_{10}\left(\frac{P_{max}}{P_{min}}\right) \quad [\text{dB}]$$

---

#### 🛠️ Core Capabilities

*   **Noise Analysis**: Real-time FFT and Welchs-method spectral density estimation ($V/\sqrt{Hz}$ and $A/\sqrt{Hz}$). Compare measurements directly against theoretical Johnson limits.
*   **Linearity & LDR**: Stitched measurements across multiple orders of magnitude using Neutral Density (ND) filters. Automated power-law fitting to extract the linearity coefficient ($\\alpha$).
*   **Hardware Control**: Low-latency acquisition via PicoScope 4000-series hardware and precision biasing via Source Measure Units (SMU).
*   **Post-Analysis**: Advanced offline processing including digital lock-in amplification for extracting signals buried in noise.

---
""")


# --- GLOBAL CONFIGURATION ---

# 1. Initialize Settings Manager (Singleton Access)
from utils.settings_manager import SettingsManager
from utils.ui_components import render_global_sidebar

if 'settings_mgr' not in st.session_state:
    st.session_state.settings_mgr = SettingsManager()

# Render Global Sidebar (Shared)
render_global_sidebar(st.session_state.settings_mgr)

# Note: st.session_state.global_amp_type etc. are updated by the sidebar function.

st.write("---")
st.write("**Current Configuration:**")
if st.session_state.global_amp_type == "FEMTO TIA":
    st.write(f"**Amplifier**: FEMTO TIA | **Gain**: {st.session_state.global_tia_gain:.1e} V/A")
    st.write(f"**Safety Limit**: {st.session_state.global_safety_max_v} V")
else:
    st.write(f"**Amplifier**: Passive Resistor | **R**: {st.session_state.global_resistor_val:.1f} Ω")
