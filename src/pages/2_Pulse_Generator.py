
import streamlit as st
import time
import sys
import os
import plotly.graph_objects as go
import numpy as np

# Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hardware.smu_controller import SMUController, InstrumentState
from utils.settings_manager import SettingsManager
from utils.ui_components import render_global_sidebar

st.set_page_config(page_title="SMU List Sweep / Pulse Generator", layout="wide")
settings = SettingsManager()

def sync_setting(st_key, setting_key):
    """Callback to sync session state with persistent settings."""
    settings.set(setting_key, st.session_state[st_key])

# --- Init Session State ---
def init_pref(key, setting_key=None):
    if setting_key is None: setting_key = key
    if key not in st.session_state:
        st.session_state[key] = settings.get(setting_key)

init_pref("p_gen_mode", "pulse_gen_mode")
init_pref("p_gen_high", "pulse_gen_high_level")
init_pref("p_gen_low", "pulse_gen_low_level")
init_pref("p_gen_comp", "pulse_gen_compliance")
init_pref("p_gen_freq", "pulse_gen_frequency")
init_pref("p_gen_duty", "pulse_gen_duty_cycle")
init_pref("p_gen_cycles", "pulse_gen_cycles")
init_pref("p_gen_res", "pulse_gen_resistor")
init_pref("p_gen_scope_range", "pulse_gen_scope_range")
init_pref("p_gen_duration", "pulse_gen_capture_duration")
init_pref("p_gen_samples", "pulse_gen_samples")
init_pref("p_gen_ac", "pulse_gen_ac_coupling")
init_pref("p_gen_delay", "pulse_gen_delay_cycles")

render_global_sidebar(settings)
st.title("🌊 SMU List Sweep & Pulse Generator")

# --- Utils ---
def render_status_banner(smu):
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Output", "ON" if getattr(smu, '_output_enabled', False) else "OFF")
    c2.metric("State", str(smu.state.name))
    st.divider()

if not st.session_state.get('is_connected', False):
    st.warning("Please connect to the SMU via the 'SMU Direct Control' page first.")
    st.stop()

smu = st.session_state.smu

# --- Error Recovery ---
if smu.state == InstrumentState.ERROR:
    st.error("⚠️ INSTRUMENT ERROR: The SMU is in an error state.")
    if st.button("Force Disconnect / Reset"):
        try:
            smu.disconnect()
        except:
            pass
        st.session_state.smu = None
        st.session_state.is_connected = False
        st.switch_page("pages/1_SMU_Direct_Control.py") # Redirect to main to reconnect
    st.stop()

render_status_banner(smu)

# --- TIA SAFETY INTERLOCK ---
if st.session_state.get('global_amp_type') == 'FEMTO TIA':
    st.warning("⚠️ **FEMTO TIA AMPLIFIER SELECTED**")
    st.info("High Gain TIA connected. Improper source settings (High Current/Voltage) can DESTROY the TIA input stage.")
    tia_confirmed = st.checkbox("✅ I confirm that Source Limits are safe for the selected TIA Gain.", key='tia_confirm_pulse')
    
    if not tia_confirmed:
        st.error("🛑 Operations Locked. Please confirm TIA safety above.")
        st.stop()

# --- Visualization Helper ---
def preview_pulse_train(high, low, period, duty, count=3):
    """Generates a simple preview of the waveform."""
    t_high = period * duty
    t_low = period * (1 - duty)
    
    times = []
    values = []
    t_curr = 0
    
    times.append(0)
    values.append(low)
    
    for i in range(count):
        times.append(t_curr)
        values.append(high)
        t_curr += t_high
        times.append(t_curr)
        values.append(high)
        times.append(t_curr)
        values.append(low)
        t_curr += t_low
        times.append(t_curr)
        values.append(low)
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=values, mode='lines', line=dict(shape='hv'), name='Waveform'))
    fig.update_layout(
        title="Waveform Preview (Ideal)",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

tab_pulse, tab_custom = st.tabs(["Pulse Generator (Square Wave)", "Custom List"])

with tab_pulse:
    main_col, guide_col = st.columns([2, 1])
    
    with main_col:
        st.subheader("Square Wave Generator")
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.markdown("**Levels**")
            st.selectbox("Pulse Mode", ["Current", "Voltage"], key="p_gen_mode", on_change=sync_setting, args=("p_gen_mode", "pulse_gen_mode"))
            pulse_mode = st.session_state.p_gen_mode
            mode_str = "CURR" if pulse_mode == "Current" else "VOLT"
            
            MAX_CURR_A = 0.1
            MAX_VOLT_V = 9.0
            
            if mode_str == 'CURR':
                high_val = st.number_input(f"High Level (A) [Max {MAX_CURR_A}]", max_value=MAX_CURR_A, format="%.6e", key="p_gen_high", on_change=sync_setting, args=("p_gen_high", "pulse_gen_high_level"))
                low_val = st.number_input(f"Low Level (A)", max_value=MAX_CURR_A, format="%.6e", key="p_gen_low", on_change=sync_setting, args=("p_gen_low", "pulse_gen_low_level"))
                comp_val = st.number_input(f"Compliance (V) [Max {MAX_VOLT_V}]", max_value=MAX_VOLT_V, key="p_gen_comp", on_change=sync_setting, args=("p_gen_comp", "pulse_gen_compliance"))
            else:
                high_val = st.number_input(f"High Level (V) [Max {MAX_VOLT_V}]", max_value=MAX_VOLT_V, format="%.6e", key="p_gen_high", on_change=sync_setting, args=("p_gen_high", "pulse_gen_high_level"))
                low_val = st.number_input(f"Low Level (V)", max_value=MAX_VOLT_V, format="%.6e", key="p_gen_low", on_change=sync_setting, args=("p_gen_low", "pulse_gen_low_level"))
                comp_val = st.number_input(f"Compliance (A) [Max {MAX_CURR_A}]", max_value=MAX_CURR_A, key="p_gen_comp", on_change=sync_setting, args=("p_gen_comp", "pulse_gen_compliance"))

        with col_p2:
            st.markdown("**Timing**")
            freq = st.number_input("Frequency (Hz)", min_value=0.001, max_value=1000.0, key="p_gen_freq", on_change=sync_setting, args=("p_gen_freq", "pulse_gen_frequency"))
            period = 1.0 / freq
            st.caption(f"Period: {period*1000:.2f} ms")
            
            duty_pct = st.slider("Duty Cycle (%)", 1, 99, key="p_gen_duty", on_change=lambda: settings.set("pulse_gen_duty_cycle", st.session_state.p_gen_duty / 100.0))
            duty = duty_pct / 100.0
            cycles = st.number_input("Cycles (0 = Infinite)", min_value=0, key="p_gen_cycles", on_change=sync_setting, args=("p_gen_cycles", "pulse_gen_cycles"))
            
        st.plotly_chart(preview_pulse_train(high_val, low_val, period, duty), use_container_width=True)
            
    with guide_col:
        st.info("### 📘 User Guide")
        st.markdown("""
        **Pulse Generator & Verification:**
        1. **Set Parameters**: Define High/Low and Timing.
        2. **Configure & Arm**: Uploads points to SMU.
        3. **Trigger Options**:
           - **TRIGGER**: Just runs the pulse.
           - **Pulse & Measure**: Runs pulse + Scope Capture.
        """)
        
        st.divider()
        st.subheader("Scope Status")
        if 'scope' in st.session_state and st.session_state.scope_connected: 
            st.success("✅ Connected")
            scope = st.session_state.scope
            enable_scope = st.checkbox("Enable Measurement", value=True)
            if enable_scope:
                resistor = st.number_input("Resistor (Ω)", key="p_gen_res", on_change=sync_setting, args=("p_gen_res", "pulse_gen_resistor"))
                min_dur = max(3 * period, 0.05)
                st.caption(f"Capture Dur: {min_dur*1000:.1f}ms")
        else:
            st.warning("Scope Not Connected")
            enable_scope = False
            scope = None

    with main_col:
        st.divider()
        st.markdown("### Controls")
        c_act1, c_act2, c_act3 = st.columns(3)
        
        with c_act1:
            if st.button("1. Configure & Arm", use_container_width=True):
                try:
                    smu.disable_output()
                    comp_type = "VOLT" if mode_str == "CURR" else "CURR"
                    smu.set_compliance(comp_val, comp_type)
                    count = cycles if cycles > 0 else 10000 
                    smu.generate_square_wave(high_val, low_val, period, duty, count, mode_str)
                    st.success("Armed! Ready to Trigger.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Config Failed: {e}")
    
        with c_act2:
            if st.button("2. TRIGGER (Pulse Only)", type="primary", use_container_width=True):
                 try:
                    if smu.state not in [InstrumentState.ARMED, InstrumentState.RUNNING]:
                        smu.disable_output()
                        comp_type = "VOLT" if mode_str == "CURR" else "CURR"
                        smu.set_compliance(comp_val, comp_type)
                        count = cycles if cycles > 0 else 10000 
                        smu.generate_square_wave(high_val, low_val, period, duty, count, mode_str)
                    smu.enable_output()
                    smu.trigger_list()
                    st.success("Trigger sent!")
                    st.rerun()
                 except Exception as e:
                    st.error(f"Trigger Failed: {e}")
        
        with c_act3:
            if st.button("🛑 STOP / ABORT", use_container_width=True):
                smu.disable_output()
                st.warning("Output Disabled.")
                st.rerun()
    
    if enable_scope:
        st.divider()
        st.markdown("### Pulse & Measure Logic")
        c_sc1, c_sc2, c_sc3, c_sc4 = st.columns(4)
        with c_sc1:
             scope_range = st.selectbox("Scope Range", ["10V", "5V", "2V", "1V", "500mV", "200mV"], key="p_gen_scope_range", on_change=sync_setting, args=("p_gen_scope_range", "pulse_gen_scope_range"))
        with c_sc2:
             calc_dur = max(3 * period, 0.05)
             override_dur = st.number_input("Capture Duration (s)", format="%.4f", key="p_gen_duration", on_change=sync_setting, args=("p_gen_duration", "pulse_gen_capture_duration"))
        with c_sc3:
             pts = st.number_input("Samples", key="p_gen_samples", on_change=sync_setting, args=("p_gen_samples", "pulse_gen_samples"))
        with c_sc4:
             ac_coupling = st.checkbox("AC Coupling", key="p_gen_ac", on_change=sync_setting, args=("p_gen_ac", "pulse_gen_ac_coupling"))
        
        delay_cycles = st.slider("Measurement Delay (Cycles)", 0, max(1, int(cycles)-1) if cycles>0 else 1000, key="p_gen_delay", on_change=sync_setting, args=("p_gen_delay", "pulse_gen_delay_cycles"))

        if st.button("🚀 Trigger & Measure Pulse", type="primary", use_container_width=True):
             try:
                 tb = scope.calculate_timebase_index(override_dur, pts)
                 coupling_str = 'AC' if ac_coupling else 'DC'
                 scope.configure_channel('A', True, scope_range, coupling_str)
                 time.sleep(0.1) 
                 smu.enable_output()
                 smu.trigger_list()
                 if delay_cycles > 0:
                     time.sleep(period * delay_cycles)
                 st.caption(f"Capturing {override_dur*1000:.1f}ms...")
                 time.sleep(0.01) 
                 times, volts = scope.capture_block(tb, pts)
                 smu.disable_output()
                 if len(volts) > 0:
                     st.success(f"Captured {len(times)} pts!")
                     v_high, v_low = np.percentile(volts, 95), np.percentile(volts, 5)
                     vpp = v_high - v_low
                     i_photo = abs(vpp) / resistor
                     m1, m2, m3 = st.columns(3)
                     m1.metric("Vpp", f"{vpp*1000:.1f} mV")
                     m2.metric("I_photo", f"{i_photo*1e6:.1f} µA")
                     m3.metric("High Level", f"{v_high:.2f} V")
                     fig = go.Figure()
                     fig.add_trace(go.Scatter(x=times, y=volts, name='Signal'))
                     fig.add_hline(y=v_high, line_dash="dash", line_color="green")
                     fig.add_hline(y=v_low, line_dash="dash", line_color="red")
                     st.plotly_chart(fig, use_container_width=True)
                 else:
                     st.error("No data returned from scope.")
             except Exception as e:
                 st.error(f"Measurement Failed: {e}")

with tab_custom:
    st.write("Custom CSV loader coming soon.")
