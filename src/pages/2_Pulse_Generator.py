
import streamlit as st
import time
import sys
import os

# Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hardware.smu_controller import SMUController, InstrumentState
from utils.settings_manager import SettingsManager
from utils.ui_components import render_global_sidebar

st.set_page_config(page_title="SMU List Sweep / Pulse Generator", layout="wide")
settings = SettingsManager()
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

# Init Session State from Settings if missing
def init_pref(key, setting_key=None):
    if setting_key is None: setting_key = key
    if key not in st.session_state:
        st.session_state[key] = settings.get(setting_key)

# Generic callback to save any setting from session_state
def update_setting(key):
    if key in st.session_state:
        settings.set(key, st.session_state[key])

def update_pulse_duty():
    if "pulse_duty_slider" in st.session_state:
        fraction = st.session_state.pulse_duty_slider / 100.0
        st.session_state.pulse_duty = fraction
        settings.set("pulse_duty", fraction)

def update_tia_gain_pulse():
    femto_gains = {"10^3 (1k)": 1e3, "10^4 (10k)": 1e4, "10^5 (100k)": 1e5, "10^6 (1M)": 1e6, "10^7 (10M)": 1e7, "10^8 (100M)": 1e8, "10^9 (1G)": 1e9, "10^10 (10G)": 1e10, "10^11 (100G)": 1e11}
    if "pulse_tia_gain_local" in st.session_state:
        val = femto_gains[st.session_state.pulse_tia_gain_local]
        st.session_state.global_tia_gain = val
        settings.set("global_tia_gain", val)

# Initialize all prefs
init_pref("pulse_mode")
init_pref("pulse_high")
init_pref("pulse_low")
init_pref("pulse_compliance")
init_pref("pulse_freq")
init_pref("pulse_duty") # Now stores fraction directly
init_pref("pulse_cycles")
init_pref("pulse_measure_resistor")
init_pref("pulse_scope_range")
init_pref("pulse_duration")
init_pref("pulse_samples")
init_pref("pulse_ac_coupling")
init_pref("pulse_delay_cycles")
init_pref("sample_name")


import plotly.graph_objects as go
import numpy as np

# --- Visualization Helper ---
def preview_pulse_train(high, low, period, duty, count=3):
    """Generates a simple preview of the waveform."""
    # Create logic for ~3 cycles
    t_high = period * duty
    t_low = period * (1 - duty)
    
    times = []
    values = []
    t_curr = 0
    
    # Pre-pad
    times.append(0)
    values.append(low)
    
    for i in range(count):
        # Rising edge
        times.append(t_curr)
        values.append(high)
        
        t_curr += t_high
        times.append(t_curr)
        values.append(high)
        
        # Falling edge
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
    
    # Layout: Control Panel (Left/Center) + Guide (Right)
    main_col, guide_col = st.columns([2, 1])
    
    with main_col:
        st.subheader("Square Wave Generator")
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.markdown("**Levels**")
            pulse_mode = st.selectbox("Pulse Mode", ["Current", "Voltage"], key="pulse_mode", on_change=update_setting, args=("pulse_mode",))
            mode_str = "CURR" if pulse_mode == "Current" else "VOLT"
            
            # SAFETY LIMITS (LED Protection)
            # Max 100mA, Max 9V
            MAX_CURR_A = 0.1
            MAX_VOLT_V = 9.0
            
            if mode_str == 'CURR':
                st.number_input(f"High Level (A) [Max {MAX_CURR_A}]", max_value=MAX_CURR_A, format="%.6e", key="pulse_high", on_change=update_setting, args=("pulse_high",))
                st.number_input(f"Low Level (A)", max_value=MAX_CURR_A, format="%.6e", key="pulse_low", on_change=update_setting, args=("pulse_low",))
                st.number_input(f"Compliance (V) [Max {MAX_VOLT_V}]", max_value=MAX_VOLT_V, key="pulse_compliance", on_change=update_setting, args=("pulse_compliance",))
            else:
                st.number_input(f"High Level (V) [Max {MAX_VOLT_V}]", max_value=MAX_VOLT_V, format="%.6e", key="pulse_high", on_change=update_setting, args=("pulse_high",))
                st.number_input(f"Low Level (V)", max_value=MAX_VOLT_V, format="%.6e", key="pulse_low", on_change=update_setting, args=("pulse_low",))
                st.number_input(f"Compliance (A) [Max {MAX_CURR_A}]", max_value=MAX_CURR_A, key="pulse_compliance", on_change=update_setting, args=("pulse_compliance",))

        with col_p2:
            st.markdown("**Timing**")
            freq = st.number_input("Frequency (Hz)", min_value=0.001, max_value=1000.0, key="pulse_freq", on_change=update_setting, args=("pulse_freq",))
            period = 1.0 / freq
            st.caption(f"Period: {period*1000:.2f} ms")
            
            st.slider("Duty Cycle (%)", 1, 99, value=int(st.session_state.pulse_duty * 100), key="pulse_duty_slider", on_change=update_pulse_duty)
            duty = st.session_state.pulse_duty
            cycles = st.number_input("Cycles (0 = Infinite)", min_value=0, key="pulse_cycles", on_change=update_setting, args=("pulse_cycles",))
        
        st.divider()
        st.subheader("Sample Information")
        st.text_input("Sample Name / ID", key='sample_name', on_change=update_setting, args=("sample_name",))
        
        # Preview
        st.plotly_chart(preview_pulse_train(st.session_state.pulse_high, st.session_state.pulse_low, period, duty), use_container_width=True)
            
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
        
        # --- Scope Status Widget ---
        st.divider()
        st.subheader("Scope Status")
        if 'scope' in st.session_state and st.session_state.scope_connected: 
            st.success("✅ Connected")
            scope = st.session_state.scope
            enable_scope = True # Set enable_scope here if connected
            
            if enable_scope:
                current_type = st.session_state.global_amp_type
                st.info(f"Amplifier Mode: **{current_type}**")
                if current_type == "FEMTO TIA":
                    femto_gains = {"10^3 (1k)": 1e3, "10^4 (10k)": 1e4, "10^5 (100k)": 1e5, "10^6 (1M)": 1e6, "10^7 (10M)": 1e7, "10^8 (100M)": 1e8, "10^9 (1G)": 1e9, "10^10 (10G)": 1e10, "10^11 (100G)": 1e11}
                    curr_val = st.session_state.global_tia_gain
                    keys = list(femto_gains.keys())
                    def_idx = next((i for i, k in enumerate(keys) if femto_gains[k] == curr_val), 0)
                    st.selectbox("TIA Gain (V/A)", keys, index=def_idx, key='pulse_tia_gain_local', on_change=update_tia_gain_pulse)
                else:
                    st.number_input("Resistor (Ω)", key="global_resistor_val", on_change=update_setting, args=("global_resistor_val",))
                
                # Auto-calc scope setting
                # Ensure we capture at least 3 periods or 50ms
                min_dur = max(3 * period, 0.05)
                st.caption(f"Capture Dur: {min_dur*1000:.1f}ms")
        else:
            st.warning("Scope Not Connected")
            enable_scope = False
            scope = None

            scope = None

    # ... (Main Col Logic) ...
    with main_col:
        st.divider()
        st.markdown("### Controls")
        
        c_act1, c_act2, c_act3 = st.columns(3)
        
        with c_act1:
            if st.button("1. Configure & Arm", help="Stops current output, uploads list, and arms trigger."):
                try:
                    # Auto-Stop first for safety/state reset
                    smu.disable_output()
                    
                    # 1. Protection
                    comp_type = "VOLT" if mode_str == "CURR" else "CURR"
                    smu.set_compliance(st.session_state.pulse_compliance, comp_type)
                    
                    # 2. Generate
                    count = st.session_state.pulse_cycles if st.session_state.pulse_cycles > 0 else 10000 
                    smu.generate_square_wave(st.session_state.pulse_high, st.session_state.pulse_low, period, duty, count, mode_str)
                    st.success("Armed! Ready to Trigger.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Config Failed: {e}")
    
    # We inject the new "Trigger & Measure" button in the Controls section
    
    with c_act2:
        is_armed = smu.state == InstrumentState.ARMED
        
        if st.button("2. TRIGGER (Pulse Only)", type="primary"):
             try:
                # Auto-Arm if not ready
                if smu.state not in [InstrumentState.ARMED, InstrumentState.RUNNING]:
                    smu.disable_output()
                    comp_type = "VOLT" if mode_str == "CURR" else "CURR"
                    smu.set_compliance(st.session_state.pulse_compliance, comp_type)
                    count = st.session_state.pulse_cycles if st.session_state.pulse_cycles > 0 else 10000 
                    smu.generate_square_wave(st.session_state.pulse_high, st.session_state.pulse_low, period, duty, count, mode_str)
                
                smu.enable_output()
                smu.trigger_list()
                st.success("Trigger sent!")
                st.rerun()
             except Exception as e:
                st.error(f"Trigger Failed: {e}")
                
    with c_act3:
        # Replace simple STOP with the Measure button (and move STOP/ABORT elswhere or add 4th col)
        pass 
        
    c_act3.empty() # Clear old slot
    
    # New row for Measurement
    # New row for Measurement
    if enable_scope: # and is_armed:
        st.divider()
        st.markdown("### Pulse & Measure Logic")
        
        # User requested manual overrides for Scope
        c_sc1, c_sc2, c_sc3, c_sc4 = st.columns(4)
        with c_sc1:
             # Default 10V as per LDR workflow
             scope_range_options = ["10V", "5V", "2V", "1V", "500mV", "200mV"]
             st.selectbox("Scope Range", scope_range_options, key="pulse_scope_range", on_change=update_setting, args=("pulse_scope_range",))
        with c_sc2:
             # Use persistent duration if it's reasonable, else use calculated
             st.number_input("Capture Duration (s)", format="%.4f", key="pulse_duration", on_change=update_setting, args=("pulse_duration",))
        with c_sc3:
             st.number_input("Samples", key="pulse_samples", on_change=update_setting, args=("pulse_samples",))
        with c_sc4:
             st.checkbox("AC Coupling", key="pulse_ac_coupling", on_change=update_setting, args=("pulse_ac_coupling",))
        
        # New: Cycle Delay
        st.slider("Measurement Delay (Cycles)", 0, max(1, int(st.session_state.pulse_cycles)-1) if st.session_state.pulse_cycles>0 else 1000, key="pulse_delay_cycles", on_change=update_setting, args=("pulse_delay_cycles",), help="Delay capture by N cycles after trigger.")

        if st.button("🚀 Trigger & Measure Pulse", type="primary", use_container_width=True):
            try:
                # 1. Logic: If SMU is not armed, arm it now with current settings
                if smu.state not in [InstrumentState.ARMED, InstrumentState.RUNNING]:
                    comp_type = "VOLT" if st.session_state.pulse_mode == "Current" else "CURR"
                    smu.set_compliance(st.session_state.pulse_compliance, comp_type)
                    count = st.session_state.pulse_cycles if st.session_state.pulse_cycles > 0 else 10000 
                    smu.generate_square_wave(st.session_state.pulse_high, st.session_state.pulse_low, period, duty, count, mode_str)

                # Trigger
                smu.enable_output()
                smu.trigger_list()
                 
                # Optional Delay
                if st.session_state.pulse_delay_cycles > 0:
                    time.sleep(period * st.session_state.pulse_delay_cycles)
                 
                # 2. Capture
                st.caption(f"Capturing {st.session_state.pulse_duration*1000:.1f}ms...")
                # Small align delay
                time.sleep(0.01) 
                 
                enable_scope = 'scope' in st.session_state and st.session_state.scope_connected

                if enable_scope:
                    # Capture data
                    raw_data = st.session_state.scope.capture_block(
                        duration_s=st.session_state.pulse_duration,
                        num_samples=st.session_state.pulse_samples,
                        range_v=st.session_state.pulse_scope_range,
                        ac_coupling=st.session_state.pulse_ac_coupling
                    )
                    times, volts = raw_data['times'], raw_data['volts']
                else:
                    times, volts = [], [] # No scope, no data
                 
                # 3. Stop SMU (Clean up whatever is left)
                smu.disable_output()
                 
                # 4. Analyze
                if len(volts) > 0:
                    st.success(f"Captured {len(times)} pts!")
                    
                    v_high = np.percentile(volts, 95)
                    v_low = np.percentile(volts, 5)
                    vpp = v_high - v_low
                    
                    r_effective = st.session_state.global_tia_gain if st.session_state.global_amp_type == "FEMTO TIA" else st.session_state.global_resistor_val
                    i_photo = abs(vpp) / r_effective
                    
                    # Metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Vpp", f"{vpp*1000:.1f} mV")
                    m2.metric("I_photo", f"{i_photo*1e6:.1f} µA")
                    m3.metric("High Level", f"{v_high:.2f} V")
                    
                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=times, y=volts, name='Signal'))
                    fig.add_hline(y=v_high, line_dash="dash", line_color="green")
                    fig.add_hline(y=v_low, line_dash="dash", line_color="red")
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("No data returned from scope.")
                      
            except Exception as e:
                st.error(f"Measurement Failed: {e}")
                import traceback
                traceback.print_exc()

    st.markdown("### Controls")
    c_stop = st.container()
    if c_stop.button("🛑 STOP / ABORT ALL", use_container_width=True):
        smu.disable_output()
        st.warning("Output Disabled.")
        st.rerun()


with tab_custom:
    st.write("Custom CSV loader coming soon.")
