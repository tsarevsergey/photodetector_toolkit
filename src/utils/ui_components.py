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

    def sync_global(st_key, setting_key):
        settings.set(setting_key, st.session_state[st_key])

    # --- Initialization Step ---
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

    # Detect out-of-sync state
    if settings.disk_is_newer():
        st.sidebar.warning("⚠️ settings.ini has been modified on disk. Click reload below to apply changes.")

    st.sidebar.radio("Amplifier/TIA Type", ["Passive Resistor", "FEMTO TIA"], 
                     key="global_amp_type", index=0 if st.session_state.global_amp_type == "Passive Resistor" else 1,
                     on_change=sync_global, args=("global_amp_type", "global_amp_type"))
    
    amp_type = st.session_state.global_amp_type
    
    if amp_type == "FEMTO TIA":
        femto_gains = {
            "10^3 (1k)": 1e3, "10^4 (10k)": 1e4, "10^5 (100k)": 1e5, 
            "10^6 (1M)": 1e6, "10^7 (10M)": 1e7, "10^8 (100M)": 1e8, 
            "10^9 (1G)": 1e9, "10^10 (10G)": 1e10, "10^11 (100G)": 1e11
        }
        curr_gain = st.session_state.global_tia_gain
        keys = list(femto_gains.keys())
        curr_idx = next((i for i, k in enumerate(keys) if femto_gains[k] == curr_gain), 0)
        
        st.sidebar.selectbox("TIA Gain (V/A)", keys, index=curr_idx, key="global_tia_gain_key", 
                             on_change=lambda: settings.set("global_tia_gain", femto_gains[st.session_state.global_tia_gain_key]))
        
        st.sidebar.number_input("Safety Guardrail (V)", min_value=1.0, max_value=10.0, 
                                key="global_safety_max_v", value=st.session_state.global_safety_max_v,
                                on_change=sync_global, args=("global_safety_max_v", "global_safety_max_v"))
        st.sidebar.warning("⚠️ Verify TIA hardware settings match this selection!")
        
    else:
        st.sidebar.number_input("Load Resistor (Ω)", format="%.2f", key="global_resistor_val", 
                                value=st.session_state.global_resistor_val,
                                on_change=sync_global, args=("global_resistor_val", "global_resistor_val"))
        st.session_state.global_safety_max_v = 10.0
        
    st.sidebar.divider()
    
    st.sidebar.checkbox("Suppress INFO Logs", key="pref_suppress_info", value=st.session_state.pref_suppress_info,
                        on_change=sync_global, args=("pref_suppress_info", "suppress_info_logs"))

    # --- Debug Information ---
    st.sidebar.divider()
    with st.sidebar.expander("🛠️ System Debug", expanded=False):
        st.write(f"**Config Path:** `{settings.get_config_path()}`")
        if st.checkbox("Show Raw Settings", key="debug_show_raw"):
            st.json(settings.settings)
    
    # --- Reload Mechanism ---
    st.sidebar.divider()
    if st.sidebar.button("🔄 Reload Settings from Config", help="Force refresh all UI values from settings.ini or config_beta.ini"):
        # 1. Force reload the physical settings file into memory
        settings.reload()
        
        # 2. Clear all persistent keys from session state so init_pref reloads them
        keys_to_clear = [k for k in st.session_state.keys() if k.startswith(('pref_', 'comm_', 'smu_', 'p_gen_', 'p_', 'sweep_', 'w_', 'ldr_', 'sample_', 'last_'))]
        for k in keys_to_clear:
            del st.session_state[k]
        st.success("Config reloaded! All fields updated.")
        st.rerun()
