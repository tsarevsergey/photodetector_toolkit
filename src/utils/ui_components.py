import streamlit as st
import logging

def render_global_sidebar(settings):
    """
    Renders the global device configuration sidebar and ensures settings are synced.
    Args:
        settings: An instance of SettingsManager
    """
    st.sidebar.header("⚙️ Global Device Config")
    st.sidebar.info("These settings apply to ALL measurements.")
    
    # --- Initialization Step ---
    # Ensure all variables exist in session state to avoid AttributeErrors during persistence
    if 'global_amp_type' not in st.session_state:
        st.session_state.global_amp_type = settings.get("global_amp_type", "Passive Resistor")
    if 'global_resistor_val' not in st.session_state:
        st.session_state.global_resistor_val = settings.get("global_resistor_val", 1000.0)
    if 'global_tia_gain' not in st.session_state:
        st.session_state.global_tia_gain = settings.get("global_tia_gain", 1e9)
    if 'global_safety_max_v' not in st.session_state:
        st.session_state.global_safety_max_v = settings.get("global_safety_max_v", 9.5)
    if 'pref_suppress_info' not in st.session_state:
        st.session_state.pref_suppress_info = settings.get("suppress_info_logs", True)

    # --- UI Components ---
    amp_type = st.sidebar.radio("Amplifier/TIA Type", ["Passive Resistor", "FEMTO TIA"], 
                        index=0 if st.session_state.global_amp_type == "Passive Resistor" else 1)
    st.session_state.global_amp_type = amp_type
    
    if amp_type == "FEMTO TIA":
        femto_gains = {
            "10^3 (1k)": 1e3, "10^4 (10k)": 1e4, "10^5 (100k)": 1e5, 
            "10^6 (1M)": 1e6, "10^7 (10M)": 1e7, "10^8 (100M)": 1e8, 
            "10^9 (1G)": 1e9, "10^10 (10G)": 1e10, "10^11 (100G)": 1e11
        }
        curr_gain = st.session_state.global_tia_gain
        curr_idx = 0
        keys = list(femto_gains.keys())
        for i, k in enumerate(keys):
            if femto_gains[k] == curr_gain:
                curr_idx = i
                break
        
        sel_gain_key = st.sidebar.selectbox("TIA Gain (V/A)", keys, index=curr_idx)
        st.session_state.global_tia_gain = femto_gains[sel_gain_key]
        
        st.session_state.global_safety_max_v = st.sidebar.number_input("Safety Guardrail (V)", 
                                                               value=st.session_state.global_safety_max_v, 
                                                               min_value=1.0, max_value=10.0)
        st.sidebar.warning("⚠️ Verify TIA hardware settings match this selection!")
        
    else:
        r_val = st.sidebar.number_input("Load Resistor (Ω)", value=st.session_state.global_resistor_val, format="%.2f")
        st.session_state.global_resistor_val = r_val
        st.session_state.global_safety_max_v = 10.0
        
    st.sidebar.divider()
    
    # Log Level Control
        
    suppress_logs = st.sidebar.checkbox("Suppress INFO Logs", value=st.session_state.pref_suppress_info, 
                               help="Hide verbose green INFO messages in the log window.")
    
    # Check change and persist
    if suppress_logs != st.session_state.pref_suppress_info:
        st.session_state.pref_suppress_info = suppress_logs
        settings.set("suppress_info_logs", suppress_logs)
        st.toast(f"Log Level Updated: {'Suppressed' if suppress_logs else 'Verbose'}")

    # Persist Amp settings on change
    settings.save_settings({
        "global_amp_type": st.session_state.global_amp_type,
        "global_resistor_val": st.session_state.global_resistor_val,
        "global_tia_gain": st.session_state.global_tia_gain,
        "global_safety_max_v": st.session_state.global_safety_max_v
    })
