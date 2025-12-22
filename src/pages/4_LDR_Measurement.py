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
from analysis.calibration import CalibrationManager
import datetime
from utils.settings_manager import SettingsManager

# Initialize Settings
settings = SettingsManager()

# Init Session State from Settings if not set
if 'pref_initialized' not in st.session_state:
    st.session_state.pref_base_folder = settings.get("base_save_folder")
    st.session_state.pref_save_raw = settings.get("save_raw_traces")
    st.session_state.pref_snr_threshold = settings.get("snr_threshold")
    st.session_state.pref_min_cycles = settings.get("min_cycles")
    st.session_state.pref_compliance = settings.get("led_compliance")
    st.session_state.pref_averages = settings.get("averages")
    st.session_state.sweep_resistor = settings.get("resistor")
    st.session_state.sweep_start_i = settings.get("sweep_start")
    st.session_state.sweep_stop_i = settings.get("sweep_stop")
    st.session_state.sweep_steps = settings.get("sweep_steps")
    st.session_state.sweep_delay_cycles = settings.get("capture_delay_cycles")
    st.session_state.cal_led_wavelength = settings.get("last_led_wavelength")
    
    # Initialize defaults for others if needed
    if 'sweep_freq' not in st.session_state: st.session_state.sweep_freq = 40.0
    if 'sweep_duty' not in st.session_state: st.session_state.sweep_duty = 0.5
    if 'pref_scope_range_idx' not in st.session_state: st.session_state.pref_scope_range_idx = 4
    
    st.session_state.pref_initialized = True

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
if 'pref_min_cycles' not in st.session_state: st.session_state.pref_min_cycles = 100

# Memory of previous settings
# Memory of previous settings
if 'sweep_start_i' not in st.session_state: st.session_state.sweep_start_i = 1e-2
if 'sweep_stop_i' not in st.session_state: st.session_state.sweep_stop_i = 1e-5
if 'sweep_steps' not in st.session_state: st.session_state.sweep_steps = 10
if 'sweep_freq' not in st.session_state: st.session_state.sweep_freq = 40.0
if 'sweep_duty' not in st.session_state: st.session_state.sweep_duty = 0.5
if 'sweep_resistor' not in st.session_state: st.session_state.sweep_resistor = 47000.0
if 'suppress_info_logs' not in st.session_state: st.session_state.suppress_info_logs = False

tab_measure, tab_settings = st.tabs(["Measurement", "Settings"])

with tab_settings:
    st.subheader("General Settings")
    st.session_state.suppress_info_logs = st.checkbox("Suppress INFO logs (Show only Warnings/Errors)", value=st.session_state.suppress_info_logs, help="Reduces console output clutter.")
    
    # Apply logging level
    import logging
    log_level = logging.WARNING if st.session_state.suppress_info_logs else logging.INFO
    # Set level for project loggers
    for logger_name in ["workflow.LDR", "instrument.SMU", "instrument.Scope"]:
        logging.getLogger(logger_name).setLevel(log_level)

    st.divider()
    st.divider()
    
    st.subheader("Data Saving")
    if 'pref_base_folder' not in st.session_state: st.session_state.pref_base_folder = "data"
    st.session_state.pref_base_folder = st.text_input("Base Save Folder", value=st.session_state.pref_base_folder, help="Root folder for saving measurements.")
    
    if 'pref_save_raw' not in st.session_state: st.session_state.pref_save_raw = False
    st.session_state.pref_save_raw = st.checkbox("Save Raw Traces (Oscilloscope Data)", value=st.session_state.pref_save_raw, help="If enabled, saves every raw waveform trace to CSV.")
    
    st.divider()
    st.subheader("Sweep Quality")
    st.session_state.pref_snr_threshold = st.number_input("SNR Pause Threshold", value=st.session_state.pref_snr_threshold, min_value=1.0, max_value=1000.0, step=1.0, help="Sweep will pause if SNR (Signal/NoiseDensity) drops below this value.")
    
    st.divider()
    st.number_input("Min Cycles per Step", value=st.session_state.pref_min_cycles, min_value=10, max_value=10000, key='pref_min_cycles', help="Minimum number of cycles/pulses to generate for each step.")
    
    st.divider()
    st.subheader("Light Source Calibration")
    
    if 'cal_led_wavelength' not in st.session_state: st.session_state.cal_led_wavelength = 461.0
    st.number_input("LED Wavelength (nm)", value=st.session_state.cal_led_wavelength, min_value=200.0, max_value=2000.0, key='cal_led_wavelength')
    
    # File Selection (Assume files are in data folder or root)
    # We can list files or just let user type path. List is better.
    import glob
    data_files = glob.glob("data/*.csv") + glob.glob("*.csv")
    
    if 'cal_ref_file' not in st.session_state: st.session_state.cal_ref_file = "data/SiDiodeResponsivity.csv"
    if 'cal_meas_file' not in st.session_state: st.session_state.cal_meas_file = "data/Si_cal1.csv"
    
    ref_file = st.selectbox("Reference Responsivity File", options=data_files, index=0 if not data_files else None, key='cal_ref_sel') 
    meas_file = st.selectbox("Reference Measurement File", options=data_files, index=0 if not data_files else None, key='cal_meas_sel')
    
    if st.button("🔄 Generate Power Calibration"):
        try:
            cal_mgr = CalibrationManager()
            # Use absolute paths if selected from list
            led_cal, r_val = cal_mgr.generate_led_calibration(meas_file, ref_file, st.session_state.cal_led_wavelength)
            
            # Save to session
            st.session_state.led_calibration = led_cal
            st.session_state.ref_responsivity_val = r_val
            st.success(f"Calibration Generated! R_ref({st.session_state.cal_led_wavelength}nm) = {r_val:.4f} A/W")
            
            # Show preview
            st.dataframe(led_cal.head())
            
        except Exception as e:
            st.error(f"Calibration failed: {e}")

    if 'led_calibration' in st.session_state:
        st.success("✅ LED Calibration Active")
    else:
        st.info("ℹ️ No calibration loaded. Results will be in Current only.")

    st.info("Settings are saved automatically for this session.")

with tab_measure:
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("Sweep Parameters")
        
        # LED Control
        # Default: High Current -> Low Current Sweep logic
        start_i = st.number_input("Start Current (A)", value=st.session_state.sweep_start_i, format="%.1e", min_value=1e-7, max_value=0.1, key='sweep_start_i')
        stop_i = st.number_input("Stop Current (A)", value=st.session_state.sweep_stop_i, format="%.1e", min_value=1e-7, max_value=0.1, key='sweep_stop_i')
        steps = st.number_input("Steps", value=st.session_state.sweep_steps, min_value=2, max_value=50, key='sweep_steps')
        
        st.divider()
        
        # Pulse settings
        freq = st.number_input("Frequency (Hz)", value=st.session_state.sweep_freq, key='sweep_freq')
        duty = st.slider("Duty Cycle", 0.1, 0.9, value=st.session_state.sweep_duty, key='sweep_duty')
        
        # Measurement Timing
        if 'sweep_delay_cycles' not in st.session_state: st.session_state.sweep_delay_cycles = 0
        delay_cycles = st.number_input("Capture Delay (Cycles)", value=st.session_state.sweep_delay_cycles, min_value=0, max_value=1000, help="Wait for N cycles of the pulse train before starting scope capture.", key='sweep_delay_cycles')
        
        st.divider()
        
        # Sample & Saving
        st.subheader("Sample Information")
        if 'sample_name' not in st.session_state: st.session_state.sample_name = "Sample_1"
        sample_name = st.text_input("Sample Name / ID", value=st.session_state.sample_name, key='sample_name')
        
        if 'enable_saving' not in st.session_state: st.session_state.enable_saving = True
        enable_saving = st.checkbox("💾 Save Data During Measurement", value=st.session_state.enable_saving, key='enable_saving')
        
        st.divider()
        
        # Amplifier Gain
        voltage_limit = st.number_input("LED Compliance Voltage (V)", value=st.session_state.pref_compliance, min_value=1.0, max_value=20.0, step=0.5, key='w_compliance')
        
        # Load Resistor (Crucial for linearity)
        r_val = st.number_input("Load Resistance (Ω)", value=st.session_state.sweep_resistor, format="%.1f", help="Adjust this to match your physical resistor. Lower values (e.g. 1k) improve high-current linearity.", key='sweep_resistor')
        
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
    st.info(f"Signal-to-Noise Ratio (Time Domain: {state['snr']:.1f}) is below threshold ({st.session_state.pref_snr_threshold}). Signal is too weak for current resistor.")
    
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
             df, wfs = workflow.run_sweep(
                start_current=start_i,
                stop_current=stop_i,
                steps=steps,
                frequency=freq,
                duty_cycle=duty,
                compliance_limit=voltage_limit,
                resistor_ohms=r_val,
                averages=averages,
                scope_range=scope_range_val,
                auto_range=auto_range,
                ac_coupling=ac_coupling,
                start_delay_cycles=delay_cycles,
                min_pulse_cycles=st.session_state.pref_min_cycles,
                min_snr_threshold=st.session_state.pref_snr_threshold,
                progress_callback= lambda p, m: status_text.text(m), # Simple callback
                start_step_index=start_idx,
                previous_results=state['results'],
                previous_waveforms=state['waveforms'],
                autosave_path=st.session_state.get('last_autosave_path', None)
            )
             
             # Success
             st.session_state.ldr_last_results = df
             st.session_state.ldr_last_waveforms = wfs
             if 'paused_state' in st.session_state: del st.session_state.paused_state
             
             # --- Save Logic (Mirrored from Main Block) ---
             # We need to reconstruct the path variables or use session state
             # We didn't save 'final_save_path' to session.
             # Check if last_autosave_path exists, use parent?
             # Or reconstruct from UI fields (assuming sample_name didnt change)
             
             if st.session_state.get('enable_saving', True): # Default to true
                 # Reconstruct path
                 timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # This creates NEW timestamp.
                 # Actually, better to append to EXISTING folder if possible?
                 # 'last_autosave_path' points to the folder created at start.
                 
                 final_save_path = st.session_state.get('last_autosave_path', None)
                 
                 # If we have a path, use it. If not (maybe raw saving off?), reconstruct base...
                 # But wait, last_autosave_path is set ONLY if saving RAW traces?
                 # Let's fix that. In the main block, we should ALWAYS set 'last_save_folder' in session if saving is on.
                 
                 # For now, let's try to use last_autosave_path if it exists, otherwise warn.
                 # Or just save to a new folder to be safe? 
                 # User might prefer one folder.
                 
                 if final_save_path and os.path.exists(final_save_path):
                     # Use existing
                     pass
                 else:
                     # Re-create/New folder? 
                     # If we just resumed, we might want to overwrite the old incomplete csv?
                     # Let's just try to save to the last_autosave_path if 'pref_save_raw' was on.
                     # If 'pref_save_raw' was off, 'last_autosave_path' is None.
                     # We need a robust 'last_data_folder' variable.
                     
                     # Quick fix: Re-construct based on UI logic
                     base = st.session_state.pref_base_folder
                     # We don't have the original timestamp. 
                     # Let's just save to a NEW timestamped folder to avoid overwriting/confusion
                     # OR check 'sample_name'
                     
                     safe_sample = "".join([c for c in sample_name if c.isalpha() or c.isdigit() or c in (' ', '_', '-')]).strip()
                     if not safe_sample: safe_sample = "Unnamed"
                     folder_name = f"{safe_sample}_{timestamp}_Resumed"
                     final_save_path = os.path.abspath(os.path.join(base, folder_name))
                     os.makedirs(final_save_path, exist_ok=True)
                     
                 try:
                    csv_path = os.path.join(final_save_path, f"{sample_name}_results_resumed.csv")
                    df.to_csv(csv_path, index=False)
                    st.success(f"✅ Data saved: {csv_path}")
                 except Exception as e:
                    st.error(f"Failed to save summary CSV: {e}")

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
    
    # Setup Autosave Path if enabled
    autosave_path_val = None
    final_save_path = None
    
    if enable_saving:
        # Save Preferences before run
        settings.save_settings({
            "base_save_folder": st.session_state.pref_base_folder,
            "save_raw_traces": st.session_state.pref_save_raw,
            "snr_threshold": st.session_state.pref_snr_threshold,
            "min_cycles": st.session_state.pref_min_cycles,
            "led_compliance": voltage_limit,
            "averages": averages,
            "resistor": r_val,
            "sweep_start": start_i,
            "sweep_stop": stop_i,
            "sweep_steps": steps,
            "capture_delay_cycles": delay_cycles
        })
    
        # Construct Path: Base / SampleName_Timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_sample = "".join([c for c in sample_name if c.isalpha() or c.isdigit() or c in (' ', '_', '-')]).strip()
        if not safe_sample: safe_sample = "Unnamed"
        
        folder_name = f"{safe_sample}_{timestamp}"
        base = st.session_state.pref_base_folder
        final_save_path = os.path.abspath(os.path.join(base, folder_name))
        
        # Handle collision (unlikely with sec timestamp but good practice)
        if os.path.exists(final_save_path):
             folder_name = f"{safe_sample}_{timestamp}_1"
             final_save_path = os.path.abspath(os.path.join(base, folder_name))
        
        # Create folder now? Or let workflow do it?
        # Workflow only creates if we pass it.
        # But we want to save Summary CSV there too, so let's create it.
        try:
            os.makedirs(final_save_path, exist_ok=True)
            st.info(f"📂 Data will be saved to: `{final_save_path}`")
            
            # Store this path for Resume logic!
            st.session_state.last_autosave_path = final_save_path 
            
            # Pass to workflow ONLY if Raw Traces are requested
            if st.session_state.pref_save_raw:
                autosave_path_val = final_save_path
                st.caption("✅ Saving Raw Traces enabled.")
            else:
                st.caption("ℹ️ Raw traces NOT saved (Settings). Only Summary CSV will be saved.")
        except Exception as e:
            st.error(f"Failed to create save folder: {e}")
            final_save_path = None # Disable saving if we can't create folder

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
            min_pulse_cycles=st.session_state.pref_min_cycles,
            start_delay_cycles=delay_cycles,
            progress_callback=update_ui,
            autosave_path=autosave_path_val
        )
        
        status_text.success("Sweep Complete!")
        progress_bar.progress(1.0)
        
        if not results.empty:
            # save to session
            st.session_state.ldr_last_results = results
            st.session_state.ldr_last_waveforms = waveforms
            
            # Save Summary CSV
            if final_save_path:
                try:
                    csv_path = os.path.join(final_save_path, f"{sample_name}_results.csv")
                    results.to_csv(csv_path, index=False)
                    st.success(f"✅ Data saved: {csv_path}")
                except Exception as e:
                    st.error(f"Failed to save summary CSV: {e}")
            
            # --- Calibration & Processing ---
            if 'led_calibration' in st.session_state:
                # Interpolate Power for measured currents
                cal_df = st.session_state.led_calibration
                # simple interp 
                f_pow = np.interp(results['LED_Current_A'], cal_df['LED_Current_A'], cal_df['Optical_Power_W'])
                results['Optical_Power_W'] = f_pow
                
                # Calculate Sensitivity (SNR=3)
                # P_min = P_meas * (3 / SNR)
                # If SNR is 0, we can't calculate.
                results['Sensitivity_W_SNR3'] = results.apply(lambda row: row['Optical_Power_W'] * (3.0 / row['SNR_FFT']) if row['SNR_FFT'] > 0 else np.nan, axis=1)
                
                # Fit Slope (Responsivity)
                cal_mgr = CalibrationManager()
                slope, intercept, r2 = cal_mgr.fit_responsivity_slope(results)
                
                st.metric("Measured Responsivity (Slope)", f"{slope:.4f} A/W", f"R²={r2:.4f}")
                
            # --- Main LDR Plot ---
            # Dual Axis if Power available?
            x_axis = "Optical_Power_W" if 'Optical_Power_W' in results.columns else "LED_Current_A"
            x_label = "Optical Power (W)" if 'Optical_Power_W' in results.columns else "LED Current (A)"
            
            fig_ldr = px.scatter(results, x=x_axis, y="Photocurrent_A", 
                             log_x=True, log_y=True, 
                             title=f"LDR: Photocurrent vs {x_label}",
                             labels={x_axis: x_label, "Photocurrent_A": "Photocurrent (A)"})
                             
            # Add trendline if calibrated
            if 'led_calibration' in st.session_state:
                 # Add the fit line?
                 # Create fit points
                 x_fit = np.linspace(results[x_axis].min(), results[x_axis].max(), 100)
                 y_fit = slope * x_fit + intercept
                 
                 # Plotly trendline is easier with OLS trendline but we did manual fit. 
                 # Let's just create a new scatter trace
                 import plotly.graph_objects as go
                 fig_ldr.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name=f'Fit (R={slope:.2f} A/W)', line=dict(dash='dash')))

            plot_placeholder.plotly_chart(fig_ldr, use_container_width=True)
            
            # Sensitivity Plot if calibrated
            if 'led_calibration' in st.session_state:
                 fig_nep = px.line(results, x=x_axis, y="Sensitivity_W_SNR3", log_x=True, log_y=True,
                                   title="Sensitivity (Min Detectable Power @ SNR=3)",
                                   labels={"Sensitivity_W_SNR3": "Min Power (W) [SNR=3]"})
                 st.plotly_chart(fig_nep, use_container_width=True)
                 
                 min_sens = results['Sensitivity_W_SNR3'].min()
                 st.info(f"**Best Sensitivity (SNR=3):** {min_sens:.2e} W")
            
            # --- Waveform Viewer ---
            st.subheader("Waveform Viewer")
            if waveforms:
                 # Create selector
                 options = [f"Step {i+1}: {w['current']:.2e} A" for i, w in enumerate(waveforms)]
                 selected_wf_idx = st.selectbox("Select Waveform Step", range(len(options)), format_func=lambda i: options[i])
                 
                 selected_wf = waveforms[selected_wf_idx]
                 
                 # Row 1: Time Domain
                 df_wf = pd.DataFrame({
                     "Time (ms)": np.array(selected_wf['times']) * 1000, 
                     "Voltage (V)": selected_wf['volts']
                 })
                 
                 fig_wf = px.line(df_wf, x="Time (ms)", y="Voltage (V)", 
                                  title=f"Waveform @ {selected_wf['current']:.2e} A (SNR_T={selected_wf.get('snr_time', 0):.1f})",
                                  markers=False)
                 st.plotly_chart(fig_wf, use_container_width=True)

                 # Row 2: Frequency Domain (FFT)
                 if 'fft_freqs' in selected_wf and len(selected_wf['fft_freqs']) > 0:
                     df_fft = pd.DataFrame({
                         "Frequency (Hz)": selected_wf['fft_freqs'],
                         "Magnitude (V)": selected_wf['fft_mag']
                     })
                     # Only show up to 10 harmonics
                     mask_fft = df_fft["Frequency (Hz)"] < (freq * 10)
                     fig_fft = px.line(df_fft[mask_fft], x="Frequency (Hz)", y="Magnitude (V)", 
                                       title=f"FFT Spectrum (SNR_F={selected_wf.get('snr_fft', 0):.1f})",
                                       log_y=True)
                     st.plotly_chart(fig_fft, use_container_width=True)
            
            # --- Results Table ---
            with st.expander("View Data Table", expanded=True):
                # Format for display
                st.dataframe(results.style.format({
                    "LED_Current_A": "{:.2e}",
                    "Scope_Vpp": "{:.4f}",
                    "Vpp_Std": "{:.2e}",
                    "SNR_FFT": "{:.1f}",
                    "SNR_Time": "{:.1f}",
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
             
             # Row 1: Time
             df_wf = pd.DataFrame({
                 "Time (ms)": np.array(selected_wf['times']) * 1000, 
                 "Voltage (V)": selected_wf['volts']
             })
             fig_wf = px.line(df_wf, x="Time (ms)", y="Voltage (V)", 
                              title=f"Waveform @ {selected_wf['current']:.2e} A",
                              markers=False)
             st.plotly_chart(fig_wf, use_container_width=True)

             # Row 2: FFT
             if 'fft_freqs' in selected_wf and len(selected_wf['fft_freqs']) > 0:
                 df_fft = pd.DataFrame({
                     "Frequency (Hz)": selected_wf['fft_freqs'],
                     "Magnitude (V)": selected_wf['fft_mag']
                 })
                 fig_fft = px.line(df_fft[df_fft["Frequency (Hz)"] < (freq * 10)], 
                                   x="Frequency (Hz)", y="Magnitude (V)", 
                                   title="FFT Spectrum",
                                   log_y=True)
                 st.plotly_chart(fig_fft, use_container_width=True)
    
    # Data Table (Persistent)
    with st.expander("View Data Table", expanded=True):
        st.dataframe(results.style.format({
            "LED_Current_A": "{:.2e}",
            "Optical_Power_W": "{:.2e}" if 'Optical_Power_W' in results.columns else "{}",
            "Scope_Vpp": "{:.4f}",
            "SNR_FFT": "{:.1f}",
            "SNR_Time": "{:.1f}",
            "SNR_Status": "{}",
            "Vpp_Std": "{:.2e}",
            "Photocurrent_A": "{:.2e}",
            "Resistance_Ohms": "{:.1f}",
            "Sensitivity_W_SNR3": "{:.2e}" if 'Sensitivity_W_SNR3' in results.columns else "{}"
        }, na_rep="-"))
