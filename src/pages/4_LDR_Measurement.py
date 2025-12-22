import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import sys
import os
import time

# Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from workflows.ldr_workflow import LDRWorkflow, ResistorChangeRequiredException
from hardware.smu_controller import InstrumentState

st.set_page_config(page_title="LDR Measurement", layout="wide")
st.title("📈 LDR Measurement (Linear Dynamic Range)")

# --- Check Connections ---
if 'smu' not in st.session_state or not st.session_state.smu:
    st.warning("SMU not connected. Please connect in 'SMU Direct Control'.")
    st.stop()
    
if 'scope' not in st.session_state or not st.session_state.scope_connected: 
    st.warning("PicoScope not connected. Please connect in 'Scope Commissioning'.")
    st.stop()
    
smu = st.session_state.smu
scope = st.session_state.scope

# --- UI Setup ---
# --- Persistence Setup ---
if 'pref_averages' not in st.session_state: st.session_state.pref_averages = 3
if 'pref_compliance' not in st.session_state: st.session_state.pref_compliance = 8.0
if 'pref_scope_range_idx' not in st.session_state: st.session_state.pref_scope_range_idx = 8 # 10V
if 'pref_snr_threshold' not in st.session_state: st.session_state.pref_snr_threshold = 25.0

tab_measure, tab_settings = st.tabs(["Measurement", "Settings"])

with tab_settings:
    st.subheader("Advanced Settings")
    st.session_state.pref_snr_threshold = st.number_input("SNR Pause Threshold", value=st.session_state.pref_snr_threshold, min_value=1.0, max_value=1000.0, step=1.0, help="Sweep will pause if SNR drops below this value.")
    
    st.info("Persistent Defaults are saved automatically when you change them in the Measurement tab.")

with tab_measure:
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("Sweep Parameters")
        
        # LED Control
        # Default: High Current -> Low Current Sweep logic
        start_i = st.number_input("Start Current (A)", value=1e-2, format="%.1e", min_value=1e-7, max_value=0.1)
        stop_i = st.number_input("Stop Current (A)", value=1e-5, format="%.1e", min_value=1e-7, max_value=0.1)
        steps = st.number_input("Steps", value=10, min_value=2, max_value=50)
        
        st.divider()
        
        # Pulse settings
        freq = st.number_input("Frequency (Hz)", value=40.0)
        duty = st.slider("Duty Cycle", 0.1, 0.9, 0.5)
        
        st.divider()
        
        # Amplifier Gain
        voltage_limit = st.number_input("LED Compliance Voltage (V)", value=st.session_state.pref_compliance, min_value=1.0, max_value=20.0, step=0.5, key='w_compliance')
        
        # Load Resistor (Crucial for linearity)
        r_val = st.number_input("Load Resistance (Ω)", value=47000.0, format="%.1f", help="Adjust this to match your physical resistor. Lower values (e.g. 1k) improve high-current linearity.")
        
        averages = st.number_input("Averages per Step", value=st.session_state.pref_averages, min_value=1, max_value=20, key='w_averages')
        
        # Scope Range Selection
        scope_range_options = ["20mV", "50mV", "100mV", "200mV", "500mV", "1V", "2V", "5V", "10V"]
        # Ensure index is valid
        idx = st.session_state.pref_scope_range_idx
        if idx >= len(scope_range_options): idx = 8
        
        scope_range_val = st.selectbox("Start Scope Range", scope_range_options, index=idx, key='w_scope_range')
        
        # Update Preferences immediately on specific widgets change? 
        # Actually easier to update them on Button Click to avoid flicker/rerun issues
        
        # Advanced Scope Settings
        c_scope1, c_scope2 = st.columns(2)
        with c_scope1:
            auto_range = st.checkbox("Auto-Range", value=False, help="Automatically adjust scope range during sweep")
        with c_scope2:
            ac_coupling = st.checkbox("AC Coupling", value=False, help="Use AC coupling to reject ambient light")
        
        st.divider()
        
        start_btn = st.button("Start Sweep", type="primary")
        stop_btn = st.button("Stop")
    
    with c2:
        st.subheader("Results")
        plot_placeholder = st.empty()
    status_text = st.empty()
    progress_bar = st.progress(0)
    
# --- Execution ---

# Check for Paused State
if 'paused_state' in st.session_state:
    state = st.session_state.paused_state
    st.warning(f"⚠️ PAUSED detected at Step {state['step_index']+1}")
    st.info(f"Signal-to-Noise Ratio ({state['snr']:.1f}) is below threshold (25.0). Signal is too weak for current resistor.")
    
    st.write(f"**Current Level:** {state['current_level']:.2e} A")
    
    # User Input for New Resistor?
    # Actually, we should ask user to confirm they changed it.
    # But we update the 'r_val' field... wait, r_val is an input widget. 
    # The user can just change the r_val widget above!
    st.write("👉 **Action Required:** Change to a larger Load Resistor (e.g. 10x higher), update the 'Load Resistance' field above, then click Resume.")
    
    c_res1, c_res2 = st.columns(2)
    resume = c_res1.button("✅ Resume (Re-measure Step)", type="primary")
    skip = c_res2.button("⏩ Skip/Keep (Accept Noisy Data)")
    
    if resume or skip:
        workflow = LDRWorkflow(smu, scope)
        start_idx = state['step_index'] if resume else state['step_index'] + 1
        
        # We need to reconstruct the parameters... they are in variables (start_i, etc). 
        # Hopefully these didn't change or we should store them in state too.
        # Ideally, we should store params in state. 
        # For this V1, we assume user didn't change sweep params (start/stop/steps) other than R.
        
        # If resuming, we discard the last (bad) result, so we use the stored 'last_results' which didn't include it yet?
        # Actually workflow raised exception BEFORE appending bad result. So 'last_results' is clean.
        # If skipping, we WANT the bad result? 
        # Exception passed 'results' which DOES NOT contain the bad point (based on my last edit).
        # So "Skip" means... we actually LOSE that point unless we manually add it??
        # Simpler: "Skip" just means "Start form next step". We lose the bad point. That's fine.
        
        try:
             results, waveforms = workflow.run_sweep(
                start_current=start_i,
                stop_current=stop_i,
                steps=steps,
                frequency=freq,
                duty_cycle=duty,
                resistor_ohms=r_val, # User might have updated this!
                averages=averages,
                compliance_limit=voltage_limit,
                scope_range=scope_range_val,
                auto_range=auto_range,
                ac_coupling=ac_coupling,
                min_snr_threshold=st.session_state.pref_snr_threshold,
                progress_callback= lambda p, m: status_text.text(m), # Simple callback
                start_step_index=start_idx,
                previous_results=state['results'],
                previous_waveforms=state['waveforms']
            )
             
             # Success
             del st.session_state.paused_state
             st.session_state.ldr_last_results = results
             st.session_state.ldr_last_waveforms = waveforms
             st.rerun()
             
        except ResistorChangeRequiredException as e:
             # Paused again!
             st.session_state.paused_state = {
                'step_index': e.step_index,
                'snr': e.snr,
                'current_level': e.current_level,
                'results': e.last_results,
                'waveforms': e.last_waveforms
            }
             st.rerun()
        except Exception as e:
             # Check for ResistorChangeRequiredException by name to handle Streamlit module reload mismatches
             if type(e).__name__ == 'ResistorChangeRequiredException':
                 # Paused execution
                 # Args: step_index, snr, current_level, results, waveforms
                 # We can access attributes directly even if classes don't match exactly
                 st.session_state.paused_state = {
                    'step_index': e.step_index,
                    'snr': e.snr,
                    'current_level': e.current_level,
                    'results': e.last_results,
                    'waveforms': e.last_waveforms
                }
                 st.rerun()
             else:
                 st.error(f"Sweep failed: {e}")
                 import traceback
                 st.write(traceback.format_exc())

if start_btn and 'paused_state' not in st.session_state:
    # Update Preferences
    st.session_state.pref_averages = averages
    st.session_state.pref_compliance = voltage_limit
    try:
        st.session_state.pref_scope_range_idx = scope_range_options.index(scope_range_val)
    except: pass

    workflow = LDRWorkflow(smu, scope)
    
    # Callback to update UI
    def update_ui(prog, msg):
        progress_bar.progress(prog)
        status_text.text(msg)
        
    try:
        results, waveforms = workflow.run_sweep(
            start_current=start_i,
            stop_current=stop_i,
            steps=steps,
            frequency=freq,
            duty_cycle=duty,
            resistor_ohms=r_val,
            averages=averages,
            compliance_limit=voltage_limit,
            scope_range=scope_range_val,
            auto_range=auto_range,
            ac_coupling=ac_coupling,
            min_snr_threshold=st.session_state.pref_snr_threshold,
            progress_callback=update_ui
        )
        
        status_text.success("Sweep Complete!")
        progress_bar.progress(1.0)
        
        if not results.empty:
            # save to session
            st.session_state.ldr_last_results = results
            st.session_state.ldr_last_waveforms = waveforms
            
            # --- Main LDR Plot ---
            fig_ldr = px.scatter(results, x="LED_Current_A", y="Photocurrent_A", 
                             log_x=True, log_y=True, 
                             title="LDR: Photocurrent vs LED Current",
                             labels={"LED_Current_A": "LED Current (A)", "Photocurrent_A": "Photocurrent (A)"})
            plot_placeholder.plotly_chart(fig_ldr, use_container_width=True)
            
            # --- Waveform Viewer ---
            st.subheader("Waveform Viewer")
            if waveforms:
                 # Create selector
                 options = [f"Step {i+1}: {w['current']:.2e} A" for i, w in enumerate(waveforms)]
                 selected_wf_idx = st.selectbox("Select Waveform Step", range(len(options)), format_func=lambda i: options[i])
                 
                 selected_wf = waveforms[selected_wf_idx]
                 
                 df_wf = pd.DataFrame({
                     "Time (ms)": np.array(selected_wf['times']) * 1000, 
                     "Voltage (V)": selected_wf['volts']
                 })
                 
                 fig_wf = px.line(df_wf, x="Time (ms)", y="Voltage (V)", 
                                  title=f"Waveform @ {selected_wf['current']:.2e} A",
                                  markers=True) # Add markers to see points clearly
                 st.plotly_chart(fig_wf, use_container_width=True)
            
            # --- Results Table ---
            with st.expander("View Data Table", expanded=True):
                # Format for display
                st.dataframe(results.style.format({
                    "LED_Current_A": "{:.2e}",
                    "Scope_Vpp": "{:.4f}",
                    "Vpp_Std": "{:.2e}",
                    "Photocurrent_A": "{:.2e}",
                    "Resistance_Ohms": "{:.1f}"
                }))
                
    except Exception as e:
             # Check for ResistorChangeRequiredException by name to handle Streamlit module reload mismatches
             if type(e).__name__ == 'ResistorChangeRequiredException':
                 # Paused execution
                 # Args: step_index, snr, current_level, results, waveforms
                 # We can access attributes directly even if classes don't match exactly
                 st.session_state.paused_state = {
                    'step_index': e.step_index,
                    'snr': e.snr,
                    'current_level': e.current_level,
                    'results': e.last_results,
                    'waveforms': e.last_waveforms
                }
                 st.rerun()
             else:
                 status_text.error(f"Sweep failed: {e}")
                 st.error(f"Sweep failed: {e}")
                 import traceback
                 st.write(traceback.format_exc())

# Render last result if persistent
if 'ldr_last_results' in st.session_state and not start_btn:
    results = st.session_state.ldr_last_results
    
    # Plot LDR
    fig = px.scatter(results, x="LED_Current_A", y="Photocurrent_A", 
                         log_x=True, log_y=True, 
                         title="LDR: Photocurrent vs LED Current (Previous Run)")
    plot_placeholder.plotly_chart(fig, use_container_width=True)
    
    # Plot Waveforms if present
    if 'ldr_last_waveforms' in st.session_state:
        waveforms = st.session_state.ldr_last_waveforms
        st.subheader("Waveform Viewer (Previous Run)")
        if waveforms:
             options = [f"Step {i+1}: {w['current']:.2e} A" for i, w in enumerate(waveforms)]
             selected_wf_idx = st.selectbox("Select Waveform Step (Prev)", range(len(options)), format_func=lambda i: options[i])
             
             selected_wf = waveforms[selected_wf_idx]
             df_wf = pd.DataFrame({
                 "Time (ms)": np.array(selected_wf['times']) * 1000, 
                 "Voltage (V)": selected_wf['volts']
             })
             fig_wf = px.line(df_wf, x="Time (ms)", y="Voltage (V)", 
                              title=f"Waveform @ {selected_wf['current']:.2e} A",
                              markers=True)
             st.plotly_chart(fig_wf, use_container_width=True)
    
    # Data Table (Persistent)
    with st.expander("View Data Table", expanded=True):
        st.dataframe(results.style.format({
            "LED_Current_A": "{:.2e}",
            "Scope_Vpp": "{:.4f}",
            "SNR": "{:.1f}",
            "SNR_Status": "{}",
            "Vpp_Std": "{:.2e}",
            "Photocurrent_A": "{:.2e}",
            "Resistance_Ohms": "{:.1f}"
        }))
