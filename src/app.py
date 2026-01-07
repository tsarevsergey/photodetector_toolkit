
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
### Welcome to the Noise Spectral Analyzer Control Software.

This toolset allows for precise characterization of photodetector noise performance, linear dynamic range (LDR), and signal stability.

**Available Tools:**

- **1. SMU Direct Control**: Manual control of SMU for IV curves and basic testing.
- **2. Pulse Generator**: Generate square wave pulses for LED driving / Linearity tests.
- **3. Scope Commissioning**: Setup, test, and validate PicoScope acquisition.
- **4. LDR Measurement**: **Main Workflow**. Automated sweeps for LDR, SNR, and Noise Density analysis.
- **5. Post Analysis**: Offline processing, Calibration, Linear Fitting, and NEP/Trace visualization.

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
