import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import sys
import os
import time
import datetime
import logging

# Append path for internal modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from workflows.ldr_workflow import LDRWorkflow, ResistorChangeRequiredException
from hardware.smu_controller import InstrumentState
from analysis.calibration import CalibrationManager
from utils.settings_manager import SettingsManager
from utils.ui_components import render_global_sidebar

# Initialize Settings
settings = SettingsManager()

# Init Session State from Settings if missing
def init_pref(key, setting_key=None):
    if setting_key is None: setting_key = key
    if key not in st.session_state:
        st.session_state[key] = settings.get(setting_key)

init_pref("base_save_folder")
init_pref("save_raw_traces")
init_pref("snr_threshold")
init_pref("min_cycles")
init_pref("led_compliance")
init_pref("averages")
init_pref("resistor")
init_pref("sweep_start")
init_pref("sweep_stop")
init_pref("sweep_steps")
init_pref("capture_delay_cycles")
init_pref("sweep_freq")
init_pref("duty_cycle")
init_pref("scope_range_idx")
init_pref("acquisition_mode")
init_pref("capture_duration")
init_pref("sample_rate")
init_pref("auto_range")
init_pref("ac_coupling")
init_pref("sample_name")
init_pref("suppress_info_logs")

# Generic callback to save any setting from session_state
def update_setting(key):
    if key in st.session_state:
        settings.set(key, st.session_state[key])

def update_scope_range_idx():
    options = ["20mV", "50mV", "100mV", "200mV", "500mV", "1V", "2V", "5V", "10V"]
    if "scope_range_idx_widget" in st.session_state:
        val = st.session_state.scope_range_idx_widget
        idx = options.index(val)
        st.session_state.scope_range_idx = idx
        settings.set("scope_range_idx", idx)

def update_tia_gain_local():
    femto_gains = {"10^3 (1k)": 1e3, "10^4 (10k)": 1e4, "10^5 (100k)": 1e5, "10^6 (1M)": 1e6, "10^7 (10M)": 1e7, "10^8 (100M)": 1e8, "10^9 (1G)": 1e9, "10^10 (10G)": 1e10, "10^11 (100G)": 1e11}
    if "ldr_tia_gain_local" in st.session_state:
        val = femto_gains[st.session_state.ldr_tia_gain_local]
        st.session_state.global_tia_gain = val
        settings.set("global_tia_gain", val)

st.set_page_config(page_title="LDR Measurement", layout="wide")
render_global_sidebar(settings)
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

# --- UI Layout ---
tab_measure, tab_settings = st.tabs(["Measurement", "Settings"])

with tab_settings:
    st.subheader("General Settings")
    st.checkbox("Suppress INFO logs (Show only Warnings/Errors)", key="suppress_info_logs", on_change=update_setting, args=("suppress_info_logs",))
    
    # Apply logging level
    log_level = logging.WARNING if st.session_state.suppress_info_logs else logging.INFO
    for logger_name in ["workflow.LDR", "instrument.SMU", "instrument.Scope"]:
        logging.getLogger(logger_name).setLevel(log_level)

    st.divider()
    st.subheader("Data Saving")
    st.text_input("Base Save Folder", key="base_save_folder", on_change=update_setting, args=("base_save_folder",))
    st.checkbox("Save Raw Traces (Oscilloscope Data)", key="save_raw_traces", on_change=update_setting, args=("save_raw_traces",))
    
    st.divider()
    st.subheader("Sweep Quality")
    st.number_input("SNR Pause Threshold", min_value=1.0, max_value=1000.0, step=1.0, key="snr_threshold", on_change=update_setting, args=("snr_threshold",))
    st.number_input("Min Cycles per Step", min_value=10, max_value=10000, key='min_cycles', on_change=update_setting, args=("min_cycles",))
    
    st.divider()
    st.subheader("Acquisition Settings")
    st.radio("Mode", ["Block", "Streaming"], horizontal=True, key='acquisition_mode', on_change=update_setting, args=("acquisition_mode",))
    c_acq1, c_acq2 = st.columns(2)
    with c_acq1:
        st.number_input("Capture Duration (s)", min_value=0.1, max_value=60.0, step=0.1, key='capture_duration', on_change=update_setting, args=("capture_duration",))
    with c_acq2:
         st.number_input("Sample Rate (Hz)", min_value=1000.0, max_value=1000000.0, step=1000.0, key='sample_rate', on_change=update_setting, args=("sample_rate",))

with tab_measure:
    # 1. Check for Paused State (Prominent at Top)
    if 'paused_state' in st.session_state:
        state = st.session_state.paused_state
        st.warning(f"⚠️ PAUSED detected at Step {state['step_index']+1}")
        st.info(f"Signal-to-Noise Ratio ({state['snr']:.1f}) is below threshold. Signal is too weak for current resistor.")
        st.write(f"**Current Level:** {state['current_level']:.2e} A")
        
        if st.session_state.get('global_amp_type') == "FEMTO TIA":
             st.write("👉 **Action Required:** Increase the **TIA Gain** in the 'Amplifier Settings' block above, then click Resume.")
        else:
             st.write("👉 **Action Required:** Switch to a larger **Gain (Resistor)** (e.g. 10x higher), update the 'Gain (Resistor)' field above, then click Resume.")
        
        c_res1, c_res2, c_res3 = st.columns(3)
        c_res1.button("✅ Resume (Re-measure Step)", type="primary", on_click=st.session_state.__setitem__, args=('resume_action', 'resume'))
        c_res2.button("⏩ Skip/Keep (Accept Noisy Data)", on_click=st.session_state.__setitem__, args=('resume_action', 'skip'))
        if c_res3.button("🛑 Cancel Entire Sweep", type="secondary"):
            if 'paused_state' in st.session_state: del st.session_state.paused_state
            st.info("Sweep cancelled by user.")
            st.rerun()

        if 'latest_waveform' in state and state['latest_waveform']:
            import plotly.graph_objects as go
            wf = state['latest_waveform']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=wf['times'], y=wf['volts'], name='Failing Trace'))
            fig.update_layout(height=300, title=f"Failing Trace (Step {state['step_index']+1})", margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        st.divider()

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Sweep Parameters")
        st.number_input("Start Current (A)", format="%.1e", min_value=1e-7, max_value=0.1, key='sweep_start', on_change=update_setting, args=("sweep_start",))
        st.number_input("Stop Current (A)", format="%.1e", min_value=1e-7, max_value=0.1, key='sweep_stop', on_change=update_setting, args=("sweep_stop",))
        st.number_input("Steps", min_value=2, max_value=50, key='sweep_steps', on_change=update_setting, args=("sweep_steps",))
        st.divider()
        st.number_input("Frequency (Hz)", key='sweep_freq', on_change=update_setting, args=("sweep_freq",))
        st.slider("Duty Cycle (Fraction)", 0.1, 0.9, key='duty_cycle', on_change=update_setting, args=("duty_cycle",), help="0.5 = 50% duty cycle. SMU requires a fraction between 0.1 and 0.9.")
        st.number_input("Capture Delay (Cycles)", min_value=0, max_value=1000, key='capture_delay_cycles', on_change=update_setting, args=("capture_delay_cycles",))
        st.divider()
        st.subheader("Sample Information")
        st.text_input("Sample Name / ID", key='sample_name', on_change=update_setting, args=("sample_name",))
        enable_saving = st.checkbox("💾 Save Data During Measurement", value=st.session_state.get('enable_saving', True), key='enable_saving')
        st.divider()
        st.write("#### Amplifier Settings")
        current_type = st.session_state.global_amp_type
        st.info(f"Amplifier Mode: **{current_type}** (Change in Sidebar)")
        if current_type == "FEMTO TIA":
            femto_gains = {"10^3 (1k)": 1e3, "10^4 (10k)": 1e4, "10^5 (100k)": 1e5, "10^6 (1M)": 1e6, "10^7 (10M)": 1e7, "10^8 (100M)": 1e8, "10^9 (1G)": 1e9, "10^10 (10G)": 1e10, "10^11 (100G)": 1e11}
            curr_val = st.session_state.global_tia_gain
            keys = list(femto_gains.keys())
            def_idx = next((i for i, k in enumerate(keys) if femto_gains[k] == curr_val), 0)
            st.selectbox("TIA Gain (V/A)", keys, index=def_idx, key='ldr_tia_gain_local', on_change=update_tia_gain_local)
        else:
            st.number_input("Gain (Resistor) (Ω)", format="%.2f", key='global_resistor_val', on_change=update_setting, args=("global_resistor_val",))
        
        st.number_input("LED Compliance Voltage (V)", min_value=1.0, max_value=20.0, step=0.5, key='led_compliance', on_change=update_setting, args=("led_compliance",))
        st.number_input("Averages per Step", min_value=1, max_value=20, key='averages', on_change=update_setting, args=("averages",))
        
        scope_range_options = ["20mV", "50mV", "100mV", "200mV", "500mV", "1V", "2V", "5V", "10V"]
        idx = min(st.session_state.scope_range_idx, len(scope_range_options)-1)
        st.selectbox("Start Scope Range", scope_range_options, index=idx, key='scope_range_idx_widget', on_change=update_scope_range_idx)
        
        c_scope1, c_scope2 = st.columns(2)
        with c_scope1:
            st.checkbox("Auto-Range", key='auto_range', on_change=update_setting, args=("auto_range",))
        with c_scope2:
            st.checkbox("AC Coupling", key='ac_coupling', on_change=update_setting, args=("ac_coupling",))
        st.divider()
        start_btn = st.button("Start Sweep", type="primary")
        if 'stop_btn_clicked' not in st.session_state: st.session_state.stop_btn_clicked = False
        def on_stop_click():
            st.session_state.stop_btn_clicked = True
            st.toast("🛑 Stop Requested...")
        st.button("Stop", on_click=on_stop_click, type="secondary")

    with c2:
        st.subheader("Results")
        status_text = st.empty()
        progress_bar = st.progress(0)
        metrics_placeholder = st.empty()
        plot_placeholder = st.empty()
        
        if 'ldr_live_metrics' in st.session_state and 'paused_state' not in st.session_state:
            with metrics_placeholder.container():
                mc1, mc2, mc3 = st.columns(3)
                m = st.session_state.ldr_live_metrics
                mc1.metric("Live Vpp", f"{m['vpp']:.2e} V")
                mc2.metric("Live SNR", f"{m['snr']:.1f}")
                mc3.metric("Scope Range", m['range'])
                if 'times' in m and 'volts' in m:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=m['times'], y=m['volts'], name='Last Trace'))
                    fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), title="Last Measured Trace")
                    st.plotly_chart(fig, use_container_width=True)

    def update_ui(prog, msg, metrics=None):
        progress_bar.progress(prog)
        status_text.text(msg)
        if st.session_state.get('stop_btn_clicked', False):
            if 'active_workflow' in st.session_state:
                st.session_state.active_workflow.stop()
        if metrics:
            st.session_state.ldr_live_metrics = metrics
            with metrics_placeholder.container():
                mc1, mc2, mc3 = st.columns(3)
                vpp, snr, rng = metrics.get('vpp', 0), metrics.get('snr', 0), metrics.get('range', 'N/A')
                mc1.metric("Live Vpp", f"{vpp:.2e} V")
                mc2.metric("Live SNR", f"{snr:.1f}")
                mc3.metric("Scope Range", rng)
                if 'times' in metrics and 'volts' in metrics:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=metrics['times'], y=metrics['volts'], name='Live Trace'))
                    fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), title="Latest Trace")
                    st.plotly_chart(fig, use_container_width=True)

    # Intersection logic for Resume/Start
    if st.session_state.get('resume_action'):
        action, state = st.session_state.resume_action, st.session_state.paused_state
        workflow = LDRWorkflow(smu, scope)
        st.session_state.active_workflow, st.session_state.stop_btn_clicked = workflow, False
        start_idx = state['step_index'] if action == 'resume' else state['step_index'] + 1
        try:
            try:
                smu.set_software_current_limit(max(start_i, stop_i))
                df, wfs = workflow.run_sweep(
                    start_level=st.session_state.sweep_start,
                    stop_level=st.session_state.sweep_stop,
                    steps=st.session_state.sweep_steps,
                    freq=st.session_state.sweep_freq,
                    duty=st.session_state.duty_cycle,
                    compliance=st.session_state.led_compliance,
                    resistor=r_val,
                    averages=st.session_state.averages,
                    start_range=state.get('last_range_string', scope_range_options[st.session_state.scope_range_idx]), 
                    auto_range=st.session_state.auto_range, 
                    ac_coupling=st.session_state.ac_coupling,
                    start_delay_cycles=st.session_state.capture_delay_cycles, 
                    min_pulse_cycles=st.session_state.min_cycles, 
                    min_snr_threshold=st.session_state.snr_threshold, 
                    progress_callback=update_ui,
                    start_step_index=start_idx, 
                    previous_results=state['results'], 
                    previous_waveforms=state['waveforms'],
                    autosave_path=st.session_state.get('last_autosave_path'), 
                    acquisition_mode=st.session_state.acquisition_mode,
                    sample_rate=st.session_state.sample_rate, 
                    capture_duration=st.session_state.capture_duration
                )
            finally:
                if 'active_workflow' in st.session_state: del st.session_state.active_workflow
                smu.set_software_current_limit(None)
                st.session_state.resume_action = None
            st.session_state.ldr_last_results, st.session_state.ldr_last_waveforms = df, wfs
            if 'paused_state' in st.session_state: del st.session_state.paused_state
            if enable_saving and st.session_state.get('last_autosave_path'):
                df.to_csv(os.path.join(st.session_state.last_autosave_path, f"{sample_name}_results_resumed.csv"), index=False)
            st.rerun()
        except ResistorChangeRequiredException as e:
            st.session_state.paused_state = {'step_index': e.step_index, 'snr': e.snr, 'current_level': e.current_level, 'results': e.last_results, 'waveforms': e.last_waveforms, 'last_range_string': getattr(e, 'last_range_str', None), 'latest_waveform': getattr(e, 'step_waveform', None)}
            st.rerun()
        except Exception as e:
            st.error(f"Sweep failed: {e}")
            st.session_state.resume_action = None

    if start_btn and 'paused_state' not in st.session_state:
        if st.session_state.sweep_start < st.session_state.sweep_stop: st.error("❌ SAFETY ERROR: Sweep must be Descending.")
        else:
            smu.set_software_current_limit(max(st.session_state.sweep_start, st.session_state.sweep_stop))
            st.session_state.stop_btn_clicked = False
            if 'ldr_live_metrics' in st.session_state: del st.session_state.ldr_live_metrics
            autosave_path_val = None
            if enable_saving:
                final_save_path = os.path.abspath(os.path.join(st.session_state.base_save_folder, f"{st.session_state.sample_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"))
                os.makedirs(final_save_path, exist_ok=True)
                st.session_state.last_autosave_path = final_save_path
                if st.session_state.save_raw_traces: autosave_path_val = final_save_path
            workflow = LDRWorkflow(smu, scope)
            st.session_state.active_workflow = workflow
            try:
                df, wfs = workflow.run_sweep(
                    start_level=st.session_state.sweep_start,
                    stop_level=st.session_state.sweep_stop,
                    steps=st.session_state.sweep_steps,
                    freq=st.session_state.sweep_freq,
                    duty=st.session_state.duty_cycle,
                    compliance=st.session_state.led_compliance,
                    resistor=r_val,
                    averages=st.session_state.averages,
                    start_range=scope_range_options[st.session_state.scope_range_idx],
                    auto_range=st.session_state.auto_range,
                    ac_coupling=st.session_state.ac_coupling,
                    sample_name=st.session_state.sample_name,
                    capture_duration=st.session_state.capture_duration,
                    sample_rate=st.session_state.sample_rate,
                    delay_cycles=st.session_state.capture_delay_cycles,
                    save_enabled=enable_saving,
                    snr_threshold=st.session_state.snr_threshold,
                    min_cycles=st.session_state.min_cycles,
                    progress_callback=update_ui, start_step_index=0, autosave_path=autosave_path_val,
                    acquisition_mode=st.session_state.acquisition_mode
                 )
                st.session_state.ldr_last_results, st.session_state.ldr_last_waveforms = df, wfs
                if enable_saving:
                    df.to_csv(os.path.join(final_save_path, f"{st.session_state.sample_name}_results.csv"), index=False)
                st.rerun()
            except ResistorChangeRequiredException as e:
                st.session_state.paused_state = {'step_index': e.step_index, 'snr': e.snr, 'current_level': e.current_level, 'results': e.last_results, 'waveforms': e.last_waveforms, 'last_range_string': getattr(e, 'last_range_str', None), 'latest_waveform': getattr(e, 'step_waveform', None)}
                st.rerun()
            except Exception as e: st.error(f"Sweep failed: {e}")
            finally:
                if 'active_workflow' in st.session_state: del st.session_state.active_workflow
                smu.set_software_current_limit(None)

    # --- Results Visualization ---
    if 'ldr_last_results' in st.session_state and not start_btn:
        results = st.session_state.ldr_last_results
        
        if 'led_calibration' in st.session_state:
            cal_df = st.session_state.led_calibration
            results['Optical_Power_W'] = np.interp(results['LED_Current_A'], cal_df['LED_Current_A'], cal_df['Optical_Power_W'])
            results['Sensitivity_W_SNR3'] = results.apply(lambda row: row['Optical_Power_W'] * (3.0 / row['SNR_FFT']) if row['SNR_FFT'] > 0 else np.nan, axis=1)
            cal_mgr = CalibrationManager()
            slope, intercept, r2 = cal_mgr.fit_responsivity_slope(results)
            st.metric("Measured Responsivity (Slope)", f"{slope:.4f} A/W", f"R²={r2:.4f}")

        if results.empty:
            st.warning("No data points were captured in the last sweep. Check hardware connections or SNR settings.")
        else:
            x_axis = "Optical_Power_W" if 'Optical_Power_W' in results.columns else "LED_Current_A"
            x_label = "Optical Power (W)" if 'Optical_Power_W' in results.columns else "LED Current (A)"
            
            if "Photocurrent_A" in results.columns:
                fig_ldr = px.scatter(results, x=x_axis, y="Photocurrent_A", log_x=True, log_y=True, 
                                     title=f"LDR: Photocurrent vs {x_label}", labels={x_axis: x_label, "Photocurrent_A": "Photocurrent (A)"})
                if 'led_calibration' in st.session_state:
                     valid_fit = results[x_axis].notna() & (results[x_axis] > 0)
                     if valid_fit.any():
                        x_fit = np.linspace(results[x_axis][valid_fit].min(), results[x_axis][valid_fit].max(), 100)
                        import plotly.graph_objects as go
                        fig_ldr.add_trace(go.Scatter(x=x_fit, y=slope * x_fit + intercept, mode='lines', name=f'Fit ({slope:.2f} A/W)', line=dict(dash='dash')))
                st.plotly_chart(fig_ldr, use_container_width=True)
            
            if 'led_calibration' in st.session_state and 'Sensitivity_W_SNR3' in results.columns:
                 fig_nep = px.line(results, x=x_axis, y="Sensitivity_W_SNR3", log_x=True, log_y=True, title="Sensitivity (Min Detectable Power @ SNR=3)")
                 st.plotly_chart(fig_nep, use_container_width=True)

        if 'ldr_last_waveforms' in st.session_state:
             st.subheader("Waveform Viewer")
             wfs = st.session_state.ldr_last_waveforms
             opts = [f"Step {i+1}: {w['current']:.2e} A" for i, w in enumerate(wfs)]
             sel_idx = st.selectbox("Select Waveform Step", range(len(opts)), format_func=lambda i: opts[i])
             swf = wfs[sel_idx]
             st.plotly_chart(px.line(x=np.array(swf['times'])*1000, y=swf['volts'], title=f"Waveform @ {swf['current']:.2e} A", labels={"x": "Time (ms)", "y": "Voltage (V)"}), use_container_width=True)
             if 'fft_freqs' in swf and len(swf['fft_freqs']) > 0:
                 st.plotly_chart(px.line(x=swf['fft_freqs'], y=swf['fft_mag'], title="FFT Spectrum", log_y=True).update_xaxes(range=[0, freq*10]), use_container_width=True)
        
        with st.expander("View Data Table", expanded=True):
            st.dataframe(results.style.format({"LED_Current_A": "{:.2e}", "Optical_Power_W": "{:.2e}", "Scope_Vpp": "{:.4f}", "SNR_FFT": "{:.1f}", "SNR_Time": "{:.1f}", "Photocurrent_A": "{:.2e}", "Resistance_Ohms": "{:.1f}", "Sensitivity_W_SNR3": "{:.2e}"}, na_rep="-"))
