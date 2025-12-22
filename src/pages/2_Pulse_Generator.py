
import streamlit as st
import time
import sys
import os

# Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hardware.smu_controller import SMUController, InstrumentState
from pages.__pycache__ import * # Trick to ensure paths? No, not needed.

st.set_page_config(page_title="SMU List Sweep / Pulse Generator", layout="wide")

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
            pulse_mode = st.selectbox("Pulse Mode", ["Current", "Voltage"])
            mode_str = "CURR" if pulse_mode == "Current" else "VOLT"
            
            # SAFETY LIMITS (LED Protection)
            # Max 100mA, Max 9V
            MAX_CURR_A = 0.1
            MAX_VOLT_V = 9.0
            
            if mode_str == 'CURR':
                high_val = st.number_input(f"High Level (A) [Max {MAX_CURR_A}]", value=1e-3, max_value=MAX_CURR_A, format="%.6e")
                low_val = st.number_input(f"Low Level (A)", value=0.0, max_value=MAX_CURR_A, format="%.6e")
                comp_val = st.number_input(f"Compliance (V) [Max {MAX_VOLT_V}]", value=8.0, max_value=MAX_VOLT_V)
            else:
                high_val = st.number_input(f"High Level (V) [Max {MAX_VOLT_V}]", value=1.0, max_value=MAX_VOLT_V, format="%.6e")
                low_val = st.number_input(f"Low Level (V)", value=0.0, max_value=MAX_VOLT_V, format="%.6e")
                comp_val = st.number_input(f"Compliance (A) [Max {MAX_CURR_A}]", value=0.01, max_value=MAX_CURR_A)

        with col_p2:
            st.markdown("**Timing**")
            freq = st.number_input("Frequency (Hz)", value=10.0, min_value=0.001, max_value=1000.0)
            period = 1.0 / freq
            st.caption(f"Period: {period*1000:.2f} ms")
            
            duty = st.slider("Duty Cycle (%)", 1, 99, 50) / 100.0
            cycles = st.number_input("Cycles (0 = Infinite)", value=100, min_value=0)
        
        # Preview
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
        
        # --- Scope Status Widget ---
        st.divider()
        st.subheader("Scope Status")
        if 'scope' in st.session_state and st.session_state.scope_connected: 
            st.success("✅ Connected")
            scope = st.session_state.scope
            
            enable_scope = st.checkbox("Enable Measurement", value=True)
            if enable_scope:
                resistor = st.number_input("Resistor (Ω)", value=47000)
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
                    smu.set_compliance(comp_val, comp_type)
                    
                    # 2. Generate
                    count = cycles if cycles > 0 else 10000 
                    smu.generate_square_wave(high_val, low_val, period, duty, count, mode_str)
                    st.success("Armed! Ready to Trigger.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Config Failed: {e}")
    
    # We inject the new "Trigger & Measure" button in the Controls section
    
    with c_act2:
        is_armed = smu.state == InstrumentState.ARMED
        
        if st.button("2. TRIGGER (Pulse Only)", type="primary", disabled=not is_armed):
             try:
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
        c_sc1, c_sc2, c_sc3 = st.columns(3)
        with c_sc1:
             # Default 10V as per LDR workflow
             scope_range = st.selectbox("Scope Range", ["10V", "5V", "2V", "1V", "500mV", "200mV"], index=0)
        with c_sc2:
             calc_dur = max(3 * period, 0.05)
             override_dur = st.number_input("Capture Duration (s)", value=calc_dur, format="%.4f")
        with c_sc3:
             pts = st.number_input("Samples", value=2000)

        if st.button("🚀 Trigger & Measure Pulse", type="primary", use_container_width=True):
             try:
                 # 1. Config Scope
                 tb = scope.calculate_timebase_index(override_dur, pts)
                 scope.configure_channel('A', True, scope_range)
                 # Wait for relay to switch if range changed
                 time.sleep(0.1) 
                 
                 # 2. Trigger SMU Pulse (Finite)
                 # We trigger first, then capture immediately? 
                 # Or capture then trigger?
                 # If pulse is 100 cycles, it lasts 100*period.
                 # If period is 100ms, total 10s.
                 # If period is 1ms, total 0.1s.
                 # We must ensure we capture WHILE pulse is happening.
                 
                 smu.enable_output()
                 smu.trigger_list() # Starts the train
                 
                 # 3. Capture
                 st.caption(f"Capturing {override_dur*1000:.1f}ms...")
                 # Small align delay
                 time.sleep(0.01) 
                 times, volts = scope.capture_block(tb, pts)
                 
                 # 4. Stop SMU (Clean up whatever is left)
                 smu.disable_output()
                 
                 # 5. Analyze
                 if len(volts) > 0:
                     st.success(f"Captured {len(times)} pts!")
                     
                     v_high = np.percentile(volts, 95)
                     v_low = np.percentile(volts, 5)
                     vpp = v_high - v_low
                     i_photo = abs(vpp) / resistor
                     
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
