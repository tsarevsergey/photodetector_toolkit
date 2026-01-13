
import streamlit as st
import time
import pandas as pd
import sys
import os

# Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hardware.smu_controller import SMUController, InstrumentState
from utils.settings_manager import SettingsManager
from utils.ui_components import render_global_sidebar

st.set_page_config(page_title="SMU Direct Control", layout="wide")
settings = SettingsManager()

def sync_setting(st_key, setting_key):
    """Callback to sync session state with persistent settings."""
    settings.set(setting_key, st.session_state[st_key])

render_global_sidebar(settings)
st.title("⚡ SMU Direct Control Interface")

# --- Session State Management ---
if 'smu' not in st.session_state:
    st.session_state.smu = None
if 'is_connected' not in st.session_state:
    st.session_state.is_connected = False
if 'smu_log' not in st.session_state:
    st.session_state.smu_log = []

# Persistent Settings
def init_pref(key, setting_key=None):
    if setting_key is None: setting_key = key
    if key not in st.session_state:
        st.session_state[key] = settings.get(setting_key)

init_pref("pref_smu_address", "smu_visa_address")
init_pref("pref_smu_mock", "smu_mock_mode")
init_pref("smu_swp_start", "smu_sweep_start")
init_pref("smu_swp_stop", "smu_sweep_stop")
init_pref("smu_swp_steps", "smu_sweep_steps")
init_pref("smu_swp_mode", "smu_sweep_mode")
init_pref("smu_swp_dir", "smu_sweep_dir")
init_pref("smu_swp_nplc", "smu_sweep_nplc")
init_pref("smu_swp_comp", "smu_sweep_comp")
init_pref("smu_man_mode", "smu_manual_mode")
init_pref("smu_man_val", "smu_manual_val")
init_pref("smu_man_comp", "smu_manual_comp")

def log(msg):
    ts = time.strftime("%H:%M:%S")
    st.session_state.smu_log.append(f"[{ts}] {msg}")

import plotly.express as px
import numpy as np

# ... (Status Banner etc) ...

# --- Helper Functions ---
def render_status_banner(smu_controller):
    status_cols = st.columns(4)
    state_color = "green" if getattr(smu_controller, '_output_enabled', False) else "red"
    
    with status_cols[0]:
        st.markdown(f"**Output State**")
        st.markdown(f":{state_color}[{'● ON' if getattr(smu_controller, '_output_enabled', False) else '○ OFF'}]")
    with status_cols[1]:
        # Simple heuristic to guess mode or just show current source state
        val = smu_controller._current_source_amps
        unit = "A"
        # If we had a way to query mode, we'd display V or I. For now, let's show what we tracked.
        st.metric("Source Current", f"{val:.2e} {unit}") 
    with status_cols[2]:
        st.metric("V Compliance", f"{smu_controller._voltage_limit_volts} V")

def generate_sweep_points(start, stop, steps, mode, direction):
    if mode == "Linear":
        points = np.linspace(start, stop, steps)
    else: # Log
        # Avoid log(0)
        s = start if start != 0 else 1e-9
        e = stop if stop != 0 else 1e-9
        points = np.logspace(np.log10(abs(s)), np.log10(abs(e)), steps)
        if start < 0: points = -points 
    
    if direction == "Double":
        points = np.concatenate([points, points[::-1]])
        
    return points

# --- Sidebar: Connection & Configuration ---
with st.sidebar:
    st.divider() # Separate from global config
    st.header("SMU Connection")
    
    address = st.text_input("VISA Address", key="pref_smu_address", value=st.session_state.pref_smu_address, on_change=sync_setting, args=("pref_smu_address", "smu_visa_address"))
    mock_mode = st.checkbox("Mock Mode", key="pref_smu_mock", value=st.session_state.pref_smu_mock, on_change=sync_setting, args=("pref_smu_mock", "smu_mock_mode"))
    
    if st.button("Connect"):
        # 1. Reset State (Clear previous connections/errors)
        if st.session_state.smu:
            try: st.session_state.smu.disconnect()
            except: pass
        st.session_state.smu = None
        st.session_state.is_connected = False
        
        # 2. Attempt New Connection
        try:
            smu = SMUController(address=address, mock=mock_mode)
            smu.connect()
            st.session_state.smu = smu
            st.session_state.is_connected = True
            log(f"Connected to SMU {'(Mock)' if mock_mode else ''}")
        except Exception as e:
            st.error(f"Connection failed: {e}")
            if "VI_ERROR" in str(e):
                st.warning("Hardware Error detected. Please output power-cycle the SMU and check USB cable.")
            log(f"Error: {e}")
            
    if st.button("Disconnect"):
        if st.session_state.smu:
            st.session_state.smu.disconnect()
            st.session_state.smu = None
            st.session_state.is_connected = False
            log("Disconnected")

    st.markdown("---")
    st.subheader("Console Log")
    st.text_area("Log", value="\n".join(st.session_state.smu_log[-10:]), height=200, disabled=True)

# --- Main Interface ---
if not st.session_state.is_connected:
    st.info("Please connect to the instrument via the sidebar.")
else:
    smu = st.session_state.smu
    
    # --- Error Recovery ---
    if smu.state == InstrumentState.ERROR:
        st.error("⚠️ INSTRUMENT ERROR: The SMU is in an error state (communication lost or hardware fault).")
        st.write("Please check connections/power and try to reset.")
        
        if st.button("Force Disconnect / Reset"):
            try:
                smu.disconnect()
            except:
                pass
            st.session_state.smu = None
            st.session_state.is_connected = False
            st.rerun()
        st.stop() # Halt UI rendering to prevent further crashes
    # Status Banner
    st.divider()
    render_status_banner(smu)
    st.divider()

    # --- TIA SAFETY INTERLOCK ---
    if st.session_state.get('global_amp_type') == 'FEMTO TIA':
        st.warning("⚠️ **FEMTO TIA AMPLIFIER SELECTED**")
        st.info("High Gain TIA connected. Improper source settings (High Current/Voltage) can DESTROY the TIA input stage.")
        tia_confirmed = st.checkbox("✅ I confirm that Source Limits are safe for the selected TIA Gain.", key='tia_confirm_smu')
        
        if not tia_confirmed:
            st.error("🛑 Operations Locked. Please confirm TIA safety above.")
            st.stop()
    
    # Tabs for different functions
    tab_manual, tab_sweep = st.tabs(["🎛️ Manual Control", "📈 IV Sweep"])
    
    # --- Manual Control Tab ---
    with tab_manual:
        st.subheader("Manual Source & Measure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Source Settings")
            source_mode = st.radio("Source Mode", ["Voltage", "Current"], key="smu_man_mode", index=0 if st.session_state.smu_man_mode == "Voltage" else 1, on_change=sync_setting, args=("smu_man_mode", "smu_manual_mode"))
            
            if source_mode == "Voltage":
                source_val = st.number_input("Set Voltage (V)", step=0.1, format="%.4f", key="smu_man_val", value=st.session_state.smu_man_val, on_change=sync_setting, args=("smu_man_val", "smu_manual_val"))
                compliance = st.number_input("Current Compliance (A)", step=0.01, format="%.6f", key="smu_man_comp", value=st.session_state.smu_man_comp, on_change=sync_setting, args=("smu_man_comp", "smu_manual_comp"))
            else:
                source_val = st.number_input("Set Current (A)", step=1e-6, format="%.8f", key="smu_man_val", value=st.session_state.smu_man_val, on_change=sync_setting, args=("smu_man_val", "smu_manual_val"))
                compliance = st.number_input("Voltage Compliance (V)", step=0.1, format="%.2f", key="smu_man_comp", value=st.session_state.smu_man_comp, on_change=sync_setting, args=("smu_man_comp", "smu_manual_comp"))
                
            if st.button("Apply Settings"):
                try:
                    if source_mode == "Voltage":
                        smu.set_compliance(compliance, "CURR")
                        smu.set_source_mode("VOLT")
                        smu.set_voltage(source_val)
                    else:
                         smu.set_compliance(compliance, "VOLT")
                         smu.set_source_mode("CURR")
                         smu.set_current(source_val)
                    
                    log(f"Applied: {source_mode} = {source_val}, Compl = {compliance}")
                    st.success("Settings Applied")
                except Exception as e:
                    st.error(f"Error: {e}")

            # Output Control
            st.markdown("### Output Control")
            is_on = getattr(smu, '_output_enabled', False)
            
            col_out1, col_out2 = st.columns(2)
            with col_out1:
                # Highlight if ON
                if st.button("Output ON", type="primary" if is_on else "secondary", disabled=is_on):
                    smu.enable_output()
                    log("Output ON")
                    st.rerun()
            with col_out2:
                # Highlight if OFF (or standard)
                if st.button("Output OFF", type="primary" if not is_on else "secondary", disabled=not is_on):
                    smu.disable_output()
                    log("Output OFF")
                    st.rerun()

        with col2:
            st.markdown("### Live Monitor")
            
            # 1. Data Storage
            if 'monitor_data' not in st.session_state:
                st.session_state.monitor_data = {'time': [], 'current': [], 'voltage': []}
                
            # 2. Controls
            mc1, mc2 = st.columns(2)
            measure_btn = mc1.button("Single Shot")
            monitoring = mc2.checkbox("Continuous Monitor")
            
            if st.button("Clear Graph"):
                st.session_state.monitor_data = {'time': [], 'current': [], 'voltage': []}
                st.rerun()

            # 3. Execution Logic
            if measure_btn or monitoring:
                try:
                    res = smu.measure()
                    
                    # Store data if monitoring
                    if monitoring:
                        st.session_state.monitor_data['time'].append(time.time())
                        st.session_state.monitor_data['current'].append(res['current'])
                        st.session_state.monitor_data['voltage'].append(res['voltage'])
                        
                        # Limit buffer to last 1000 points
                        if len(st.session_state.monitor_data['time']) > 1000:
                            for k in st.session_state.monitor_data:
                                st.session_state.monitor_data[k].pop(0)
                    
                    # Display Metrics
                    st.metric("Voltage", f"{res['voltage']:.5f} V")
                    st.metric("Current", f"{res['current']:.5e} A")
                    
                    if not monitoring:
                        log(f"Meas: {res}")
                        
                except Exception as e:
                    st.error(f"Measure failed: {e}")
                
                # Loop
                if monitoring:
                    time.sleep(0.1)
                    st.rerun()
            
            # 4. Plotting
            if len(st.session_state.monitor_data['time']) > 1:
                # Calculate relative time
                t0 = st.session_state.monitor_data['time'][0]
                t_rel = [t - t0 for t in st.session_state.monitor_data['time']]
                y_data = st.session_state.monitor_data['current']
                
                # Use Graph Objects for speed/customization? px is fine.
                fig_mon = px.line(x=t_rel, y=y_data, title="Current Drift Log")
                fig_mon.update_layout(
                    xaxis_title="Time (s)",
                    yaxis_title="Current (A)",
                    height=300,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig_mon, use_container_width=True)

    # --- IV Sweep Tab ---
    with tab_sweep:
        st.subheader("Advanced IV Sweep")
        
        # Configuration
        with st.expander("Sweep Configuration", expanded=True):
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                # Param
                start_v = st.number_input("Start V", key="smu_swp_start", value=st.session_state.smu_swp_start, on_change=sync_setting, args=("smu_swp_start", "smu_sweep_start"))
                stop_v = st.number_input("Stop V", key="smu_swp_stop", value=st.session_state.smu_swp_stop, on_change=sync_setting, args=("smu_swp_stop", "smu_sweep_stop"))
                num_steps = st.number_input("Number of Points", min_value=2, step=1, key="smu_swp_steps", value=st.session_state.smu_swp_steps, on_change=sync_setting, args=("smu_swp_steps", "smu_sweep_steps"))
            with sc2:
                # Modes
                sweep_mode = st.selectbox("Spacing", ["Linear", "Log"], key="smu_swp_mode", index=0 if st.session_state.smu_swp_mode == "Linear" else 1, on_change=sync_setting, args=("smu_swp_mode", "smu_sweep_mode"))
                sweep_dir = st.selectbox("Direction", ["Single", "Double"], key="smu_swp_dir", index=0 if st.session_state.smu_swp_dir == "Single" else 1, on_change=sync_setting, args=("smu_swp_dir", "smu_sweep_dir"))
                smu_nplc_opts = [0.01, 0.1, 1.0, 10.0]
                smu_nplc_idx = smu_nplc_opts.index(st.session_state.smu_swp_nplc) if st.session_state.smu_swp_nplc in smu_nplc_opts else 1
                nplc = st.selectbox("Speed (NPLC)", smu_nplc_opts, key="smu_swp_nplc", index=smu_nplc_idx, on_change=sync_setting, args=("smu_swp_nplc", "smu_sweep_nplc"))
            with sc3:
                # Limits
                sweep_comp = st.number_input("Compliance (A)", format="%.1e", key="smu_swp_comp", value=st.session_state.smu_swp_comp, on_change=sync_setting, args=("smu_swp_comp", "smu_sweep_comp"))
                
        run_sweep = st.button("Run Sweep", type="primary")
        
        # Plot Settings
        st.markdown("### Plot Settings")
        pc1, pc2 = st.columns(2)
        plot_log_y = pc1.checkbox("Log Y (Current)", value=False)
        plot_log_x = pc2.checkbox("Log X (Voltage)", value=False)

        if run_sweep:
            # SAFETY CHECK: Instrument must be OFF before starting a sweep
            if getattr(smu, '_output_enabled', False):
                st.error("⚠️ Safety Interlock: Sweep cannot start while Output is ON. Please turn OFF output manually.")
                st.stop()

            st.info("Initializing Sweep...")
            results = []
            
            try:
                # 1. Setup
                smu.set_nplc(nplc)
                # Set V protection
                max_v = max(abs(start_v), abs(stop_v)) + 2.0
                smu.set_compliance(max_v, "VOLT") 
                # Set Current Compliance
                smu.set_compliance(sweep_comp, "CURR")
                
                smu.set_source_mode("VOLT")
                smu.enable_output()
                
                # 2. Generate Points
                voltages = generate_sweep_points(start_v, stop_v, int(num_steps), sweep_mode, sweep_dir)
                
                # 3. Execution
                progress_bar = st.progress(0)
                
                for i, v in enumerate(voltages):
                    smu.set_voltage(v)
                    time.sleep(0.01) # Short bus latency
                    meas = smu.measure()
                    meas['step_voltage'] = v 
                    results.append(meas)
                    progress_bar.progress((i + 1) / len(voltages))
                
                # 4. Cleanup
                smu.disable_output()
                st.success("Sweep Complete")
                
                # 5. Process Data for Plotting
                df = pd.DataFrame(results)
                
                # Handling Log Plots
                df['abs_current'] = df['current'].abs()
                df['abs_voltage'] = df['voltage'].abs()
                
                y_col = 'abs_current' if plot_log_y else 'current'
                x_col = 'abs_voltage' if plot_log_x else 'voltage'
                
                fig = px.line(df, x=x_col, y=y_col, markers=True, 
                              title="IV Characteristic",
                              log_y=plot_log_y, log_x=plot_log_x)
                
                # Scientific notation
                fig.update_layout(xaxis=dict(tickformat=".2e"), yaxis=dict(tickformat=".2e"))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Data Export
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Data", csv, "iv_sweep_advanced.csv", "text/csv")
                
            except Exception as e:
                st.error(f"Sweep failed: {e}")
                smu.disable_output()

