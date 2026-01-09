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
    def init_pref(key, setting_key=None, default=None):
        if setting_key is None: setting_key = key
        if key not in st.session_state:
            st.session_state[key] = settings.get(setting_key, default)

    init_pref("global_amp_type", default="Passive Resistor")
    init_pref("global_resistor_val", default=1000.0)
    init_pref("global_tia_gain", default=1e9)
    init_pref("global_safety_max_v", default=9.5)
    init_pref("suppress_info_logs", default=True)
    init_pref("last_led_wavelength", default=461.0)
    
    # --- UI Components ---
    st.sidebar.radio("Amplifier/TIA Type", ["Passive Resistor", "FEMTO TIA"], 
                     key="global_amp_type", on_change=update_setting_cb, args=(settings, "global_amp_type"))
    
    if st.session_state.global_amp_type == "FEMTO TIA":
        # ...
        st.sidebar.number_input("Safety Guardrail (V)", min_value=1.0, max_value=10.0, step=0.1,
                               key="global_safety_max_v", on_change=update_setting_cb, args=(settings, "global_safety_max_v"))
        st.sidebar.warning("⚠️ Verify TIA hardware settings match this selection!")
        
    else:
        st.sidebar.number_input("Load Resistor (Ω)", format="%.2f", 
                               key="global_resistor_val", on_change=update_setting_cb, args=(settings, "global_resistor_val"))
        st.session_state.global_safety_max_v = 10.0
        
    st.sidebar.divider()
    
    st.sidebar.number_input("LED Wavelength (nm)", min_value=100.0, max_value=2000.0, step=0.1,
                          key="last_led_wavelength", on_change=update_setting_cb, args=(settings, "last_led_wavelength"))
    
    st.sidebar.divider()
    
    st.sidebar.checkbox("Suppress INFO Logs", key="suppress_info_logs", on_change=update_setting_cb, args=(settings, "suppress_info_logs"),
                       help="Hide verbose green INFO messages in the log window.")

def update_setting_cb(settings, key):
    if key in st.session_state:
        settings.set(key, st.session_state[key])
