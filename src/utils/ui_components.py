import streamlit as st
import logging

def render_global_sidebar(settings, hide_config=False):
    """
    Renders the global device configuration sidebar and ensures settings are synced.
    Args:
        settings: An instance of SettingsManager
        hide_config: If True, hides amplifier/TIA/Resistor settings (managed by page).
    """
    st.sidebar.header("⚙️ Global Device Config")
    st.sidebar.info("These settings apply to ALL measurements.")

    # --- Initialization Step ---
    def init_pref(key, setting_key=None, default=None):
        if setting_key is None: setting_key = key
        if key not in st.session_state:
            st.session_state[key] = settings.get(setting_key, default)

    init_pref("global_amp_type", default="Passive Resistor")
    init_pref("global_resistor_val", default=47000.0)
    init_pref("global_tia_gain", default=1e6)
    init_pref("global_safety_max_v", default=9.5)
    init_pref("suppress_info_logs", default=True)
    init_pref("last_led_wavelength", default=461.0)
    
    # --- UI Components ---
    if not hide_config:
        st.sidebar.radio("Amplifier/TIA Type", ["Passive Resistor", "FEMTO TIA"], 
                         key="global_amp_type", on_change=update_setting_cb, args=(settings, "global_amp_type"))
        
        if st.session_state.global_amp_type == "FEMTO TIA":
            # Show nothing or just TIA safeguards? 
            # If hiding config, we assume page handles Selection AND Gain.
            # Sidebar handles safety guardrail? Let's hide it too if page is managing TIA context.
            # actually, safety might remain global? But duplicate keys on page might exist if page repeats guardrail.
            # For now, let's keep guardrail in sidebar unless requested otherwise, 
            # but usually page only duplicates Gain/Resistor.
            
            st.sidebar.number_input("Safety Guardrail (V)", min_value=1.0, max_value=10.0, step=0.1,
                                   key="global_safety_max_v", on_change=update_setting_cb, args=(settings, "global_safety_max_v"))
            st.sidebar.warning("⚠️ Verify TIA hardware settings match this selection!")
            
        else:
            st.sidebar.number_input("Load Resistor (Ω)", format="%.2f", 
                                   key="global_resistor_val", on_change=update_setting_cb, args=(settings, "global_resistor_val"))
            st.session_state.global_safety_max_v = 10.0
    else:
        # even if hidden, we might want to show READ ONLY status?
        # No, simpler to just hide controls. The page will show controls.
        pass
        
    st.sidebar.divider()
    
    st.sidebar.number_input("LED Wavelength (nm)", min_value=100.0, max_value=2000.0, step=0.1,
                          key="last_led_wavelength", on_change=update_setting_cb, args=(settings, "last_led_wavelength"))
    
    st.sidebar.divider()
    
    st.sidebar.checkbox("Suppress INFO Logs", key="suppress_info_logs", on_change=update_setting_cb, args=(settings, "suppress_info_logs"),
                       help="Hide verbose green INFO messages in the log window.")
                       
    if st.sidebar.button("⚠️ Hard Reset App State", help="Clears all temporary memory and reloads settings from config file. Use this if controls behave strangely."):
        st.session_state.clear()
        st.rerun()

def update_setting_cb(settings, key):
    if key in st.session_state:
        settings.set(key, st.session_state[key])
