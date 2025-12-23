
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

Please select a tool from the sidebar to begin:

- **1. SMU Direct Control**: Manual control of SMU for IV curves and testing.
- **2. Pulse Generator**: Generate square wave pulses for LED driving.
- **3. Scope Commissioning**: Setup and test PicoScope acquisition.
- **4. LDR Measurement**: Main Linear Dynamic Range and Noise Analysis workflow.

---
""")

# --- GLOBAL CONFIGURATION ---

if 'global_amp_type' not in st.session_state: st.session_state.global_amp_type = "Passive Resistor"
if 'global_resistor_val' not in st.session_state: st.session_state.global_resistor_val = 1000.0
if 'global_tia_gain' not in st.session_state: st.session_state.global_tia_gain = 1000.0
if 'global_safety_max_v' not in st.session_state: st.session_state.global_safety_max_v = 9.5

with st.sidebar:
    st.header("⚙️ Global Device Config")
    st.info("These settings apply to ALL measurements.")
    
    # Amplifier
    amp_type = st.radio("Amplifier/TIA Type", ["Passive Resistor", "FEMTO TIA"], 
                        index=0 if st.session_state.global_amp_type == "Passive Resistor" else 1)
    st.session_state.global_amp_type = amp_type
    
    if amp_type == "FEMTO TIA":
        femto_gains = {
            "10^3 (1k)": 1e3, "10^4 (10k)": 1e4, "10^5 (100k)": 1e5, 
            "10^6 (1M)": 1e6, "10^7 (10M)": 1e7, "10^8 (100M)": 1e8, 
            "10^9 (1G)": 1e9, "10^10 (10G)": 1e10, "10^11 (100G)": 1e11
        }
        # Find index of current gain
        curr_gain = st.session_state.global_tia_gain
        curr_idx = 0
        keys = list(femto_gains.keys())
        # Try to match
        for i, k in enumerate(keys):
            if femto_gains[k] == curr_gain:
                curr_idx = i
                break
        
        sel_gain_key = st.selectbox("TIA Gain (V/A)", keys, index=curr_idx)
        st.session_state.global_tia_gain = femto_gains[sel_gain_key]
        
        st.session_state.global_safety_max_v = st.number_input("Safety Guardrail (V)", 
                                                               value=st.session_state.global_safety_max_v, 
                                                               min_value=1.0, max_value=10.0)
        st.warning("⚠️ Verify TIA hardware settings match this selection!")
        
    else:
        r_val = st.number_input("Load Resistor (Ω)", value=st.session_state.global_resistor_val, format="%.2f")
        st.session_state.global_resistor_val = r_val
        st.session_state.global_safety_max_v = 10.0

st.write("---")
st.write("**Current Configuration:**")
if st.session_state.global_amp_type == "FEMTO TIA":
    st.write(f"**Amplifier**: FEMTO TIA | **Gain**: {st.session_state.global_tia_gain:.1e} V/A")
    st.write(f"**Safety Limit**: {st.session_state.global_safety_max_v} V")
else:
    st.write(f"**Amplifier**: Passive Resistor | **R**: {st.session_state.global_resistor_val:.1f} Ω")
