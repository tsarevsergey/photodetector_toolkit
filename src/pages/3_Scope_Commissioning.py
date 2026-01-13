
import streamlit as st
import time
import sys
import os
import plotly.express as px
import pandas as pd
import numpy as np
import datetime

# Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hardware.scope_controller import ScopeController, InstrumentState
from utils.settings_manager import SettingsManager
from utils.ui_components import render_global_sidebar

st.set_page_config(page_title="Scope Commissioning", layout="wide")
settings = SettingsManager()

def sync_setting(st_key, setting_key):
    """Callback to sync session state with persistent settings."""
    settings.set(setting_key, st.session_state[st_key])

# --- Init Session State ---
def init_pref(key, setting_key=None):
    if setting_key is None: setting_key = key
    if key not in st.session_state:
        st.session_state[key] = settings.get(setting_key)

init_pref("comm_range_a", "scope_comm_range_a")
init_pref("comm_coup_a", "scope_comm_coupling_a")
init_pref("comm_acq_mode", "scope_comm_acq_mode")
init_pref("comm_duration_ms", "scope_comm_duration_ms")
init_pref("comm_num_samples", "scope_comm_num_samples")
init_pref("comm_sample_rate", "scope_comm_sample_rate")
init_pref("comm_quick_range", "scope_comm_quick_range")
init_pref("comm_quick_coup", "scope_comm_quick_coupling")
init_pref("comm_tia_gain", "scope_comm_tia_gain")
if 'comm_gain' not in st.session_state:
    st.session_state.comm_gain = settings.get("scope_comm_tia_gain")

init_pref("pref_scope_mock", "scope_mock_mode")
render_global_sidebar(settings)
st.title("📉 PicoScope 2208B Commissioning")

# --- Session State ---
if 'scope' not in st.session_state:
    st.session_state.scope = None
if 'scope_connected' not in st.session_state:
    st.session_state.scope_connected = False

# --- Context ---
# We keep the SMU session alive if possible, but this page focuses on Scope.

# --- Sidebar: Connection ---
with st.sidebar:
    st.divider() # Separate from global config
    st.header("Scope Connection")
    mock_mode = st.checkbox("Mock Mode", key="pref_scope_mock", value=st.session_state.pref_scope_mock, on_change=sync_setting, args=("pref_scope_mock", "scope_mock_mode"))
    
    if st.button("Connect Scope"):
        # Reset
        if st.session_state.scope:
            try: st.session_state.scope.disconnect()
            except: pass
        st.session_state.scope = None
        st.session_state.scope_connected = False
        
        try:
            scope = ScopeController(mock=mock_mode)
            scope.connect()
            st.session_state.scope = scope
            st.session_state.scope_connected = True
            st.success("Connected to PicoScope!")
        except Exception as e:
            st.error(f"Connection Failed: {e}")
            
    if st.button("Disconnect Scope"):
        if st.session_state.scope:
            st.session_state.scope.disconnect()
        st.session_state.scope = None
        st.session_state.scope_connected = False
        st.info("Disconnected")

    st.divider()
    if st.button("⚠️ Brute Force Driver Reset", help="Click this if 'PICO_NOT_FOUND' persists. Then replug USB."):
        ScopeController.force_close_all()
        st.warning("Sent Close command to all 32 potential handles. Please Replug USB now.")

# --- Main Interface ---

if not st.session_state.scope_connected:
    st.info("Please connect to the PicoScope in the sidebar.")
    st.stop()

scope = st.session_state.scope

# Error Shield
if scope.state == InstrumentState.ERROR:
    st.error("Scope is in ERROR state.")
    if st.button("Reset Scope"):
        st.session_state.scope = None
        st.session_state.scope_connected = False
        st.rerun()
    st.stop()

# --- tabs ---
config_tab, capture_tab, noise_tab, detect_tab = st.tabs(["Configuration", "Capture & View", "Noise Calibration", "Detectivity"])

with config_tab:
    st.subheader("Channel Setup")
    c1, c2 = st.columns(2)
    
    ranges = ['10MV', '20MV', '50MV', '100MV', '200MV', '500MV', '1V', '2V', '5V', '10V', '20V']
    
    with c1:
        st.markdown("### Channel A")
        en_a = st.checkbox("Enable Ch A", value=True)
        st.selectbox("Range A", ranges, key="comm_range_a", index=ranges.index(st.session_state.comm_range_a) if st.session_state.comm_range_a in ranges else 0, on_change=sync_setting, args=("comm_range_a", "scope_comm_range_a"))
        st.selectbox("Coupling A", ["DC", "AC"], key="comm_coup_a", index=0 if st.session_state.comm_coup_a == "DC" else 1, on_change=sync_setting, args=("comm_coup_a", "scope_comm_coupling_a"))
        range_a = st.session_state.comm_range_a
        coup_a = st.session_state.comm_coup_a
        
        if st.button("Apply Ch A"):
            scope.configure_channel('A', en_a, range_a, coup_a)
            st.success("Ch A Configured")
            
    with c2:
        st.markdown("### Channel B")
        en_b = st.checkbox("Enable Ch B", value=False)
        range_b = st.selectbox("Range B", ranges, index=7, key="scope_comm_range_b_local")
        coup_b = st.selectbox("Coupling B", ["DC", "AC"], index=0, key="scope_comm_coup_b_local")
        
        if st.button("Apply Ch B"):
            scope.configure_channel('B', en_b, range_b, coup_b)
            st.success("Ch B Configured")

with capture_tab:
    st.subheader("Capture")
    
    # 1. Acquisition Settings
    c_mode, c_dur = st.columns(2)
    
    with c_mode:
        st.radio("Acquisition Mode", ["Block", "Streaming"], key="comm_acq_mode", index=0 if st.session_state.comm_acq_mode == "Block" else 1, on_change=sync_setting, args=("comm_acq_mode", "scope_comm_acq_mode"), help="Block: Short, high speed. Streaming: Long, continuous.")
        acq_mode = st.session_state.comm_acq_mode
        
    with c_dur:
        if acq_mode == "Block":
            duration_ms = st.number_input("Duration (ms)", min_value=0.1, max_value=5000.0, step=10.0, key="comm_duration_ms", value=st.session_state.comm_duration_ms, on_change=sync_setting, args=("comm_duration_ms", "scope_comm_duration_ms"))
            duration_s = duration_ms / 1000.0
        else:
            duration_s = st.number_input("Duration (s)", min_value=0.1, max_value=100.0, step=0.5, key="comm_duration_s_streaming", value=st.session_state.get('comm_duration_s_streaming', 2.0))

    # 2. Resolution Settings
    c_res1, c_res2 = st.columns(2)
    
    with c_res1:
        if acq_mode == "Block":
            num_samples = st.number_input("Number of Samples", min_value=100, max_value=1000000, key="comm_num_samples", value=st.session_state.comm_num_samples, on_change=sync_setting, args=("comm_num_samples", "scope_comm_num_samples"))
        else:
            sample_rate = st.number_input("Sample Rate (Hz)", min_value=1000.0, max_value=1000000.0, step=10000.0, key="comm_sample_rate", value=st.session_state.comm_sample_rate, on_change=sync_setting, args=("comm_sample_rate", "scope_comm_sample_rate"))
            sample_rate = st.session_state.comm_sample_rate
            
    with c_res2:
        if scope and st.session_state.scope_connected:
            if acq_mode == "Block":
                tb_idx = scope.calculate_timebase_index(duration_s, num_samples)
                st.metric("Effective Rate", f"{(num_samples/duration_s)/1000:.1f} kS/s")
                st.caption(f"Timebase Index: {tb_idx}")
            else:
                st.metric("Total Samples", f"{int(duration_s * sample_rate):,}")

    # 3. Quick Config Override (Alignment with LDR)
    st.divider()
    st.caption("Quick Overrides (Applied before capture)")
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        st.selectbox("Coupling", ["DC", "AC"], key="comm_quick_coup", index=0 if st.session_state.comm_quick_coup == "DC" else 1, on_change=sync_setting, args=("comm_quick_coup", "scope_comm_quick_coupling"))
        quick_coup = st.session_state.comm_quick_coup
    with qc2:
        qr_ranges = ['10MV', '20MV', '50MV', '100MV', '200MV', '500MV', '1V', '2V', '5V', '10V']
        st.selectbox("Range", qr_ranges, key="comm_quick_range", index=qr_ranges.index(st.session_state.comm_quick_range) if st.session_state.comm_quick_range in qr_ranges else 0, on_change=sync_setting, args=("comm_quick_range", "scope_comm_quick_range"))
        quick_range = st.session_state.comm_quick_range
    with qc3:
        tia_gain = st.number_input("TIA Gain (Ω)", format="%.2e", key="comm_tia_gain", value=st.session_state.comm_tia_gain, on_change=sync_setting, args=("comm_tia_gain", "scope_comm_tia_gain"), help="Load Resistor or TIA Gain for noise conversion")
        st.session_state.comm_gain = tia_gain # Sync the snapshot value too

    if st.button("Start Capture", type="primary"):
        with st.spinner("Capturing..."):
            try:
                # Apply Quick Config First to ensure state matches UI expectation
                scope.configure_channel('A', True, quick_range, quick_coup)
                
                if acq_mode == "Block":
                    # Calculate correct index
                    tb_index = scope.calculate_timebase_index(duration_s, num_samples)
                    times, volts = scope.capture_block(int(tb_index), int(num_samples))
                else:
                    # Streaming
                    times, volts = scope.capture_streaming(duration_s, sample_rate)
                
                if len(times) > 0:
                    st.success(f"Captured {len(times)} samples")
                    # Save to Session State
                    st.session_state.comm_times = times
                    st.session_state.comm_volts = volts
                    st.session_state.last_acq_mode_captured = acq_mode
                    st.session_state.comm_gain = tia_gain
                else:
                    st.warning("No data returned.")
            except Exception as e:
                st.error(f"Capture Error: {e}")

    # --- Display Section (Persists independently of button) ---
    if 'comm_times' in st.session_state:
        times = st.session_state.comm_times
        volts = st.session_state.comm_volts
        acq_mode_disp = st.session_state.get('last_acq_mode_captured', 'Block')
        gain_disp = st.session_state.comm_gain
        
        st.divider()
        
        # Downsample for big plots
        if len(times) > 50000:
            st.info("Displaying downsampled trace (full data used for analysis)...")
            plot_times = times[::10]
            plot_volts = volts[::10]
        else:
            plot_times = times
            plot_volts = volts
            
        df = pd.DataFrame({'Time (s)': plot_times, 'Signal (V)': plot_volts})
        
        # Trace Plot
        fig = px.line(df, x='Time (s)', y='Signal (V)', title=f"{acq_mode_disp} Trace (Gain: {gain_disp:.1e} Ω)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats
        st.write("### Signal Statistics")
        st.write(df['Signal (V)'].describe().T)
        
        # FFT Analysis
        st.write(f"### FFT Analysis")
        from scipy.signal import periodogram
        
        fs_actual = 1.0 / (times[1] - times[0])
        f, Pxx = periodogram(volts, fs_actual) 
        
        asd_volts = np.sqrt(Pxx)
        
        if gain_disp <= 0: gain_disp = 1e3
        asd_current = asd_volts / gain_disp
        
        fft_df = pd.DataFrame({
            'Freq (Hz)': f, 
            'Voltage Noise (V/rtHz)': asd_volts,
            'Current Noise (A/rtHz)': asd_current
        })
        
        fft_df = fft_df[fft_df['Freq (Hz)'] > 0]
        
        view_metric = st.radio("View Metric", ["Voltage Noise (V/rtHz)", "Current Noise (A/rtHz)"], horizontal=True)
        
        fft_fig = px.line(fft_df, x='Freq (Hz)', y=view_metric, log_y=True, log_x=True, 
                          title=f"Noise Density ({view_metric}) | Gain: {gain_disp:.1e} Ω")
        
        st.plotly_chart(fft_fig, use_container_width=True)

with noise_tab:
    st.subheader("Noise Floor Calibration (Johnson Noise)")
    st.info("Compare measured noise against the theoretical thermal limits of a source resistor.")
    
    n1, n2, n3, n4, n5 = st.columns(5)
    with n1:
        source_r_ohms = st.number_input("Source Resistor (Ω)", value=st.session_state.get('cal_source_r', 1000.0), format="%.2e", min_value=1.0, key="cal_source_r")
    with n2:
        cal_gain = st.number_input("TIA Gain (V/A)", value=st.session_state.get('cal_gain_val', 1000.0), format="%.2e", min_value=1.0, key="cal_gain_val")
    with n3:
        cal_range = st.selectbox("Scope Range", ['10MV', '20MV', '50MV', '100MV', '200MV', '500MV', '1V', '2V'], index=0, key="cal_range_sel")
    with n4:
        cal_coupling = st.selectbox("Coupling", ["AC", "DC"], index=0, key="cal_coup_sel")
    with n5:
        cal_duration = st.number_input("Cal Duration (s)", value=st.session_state.get('cal_dur', 1.0), step=0.5, key="cal_dur")

    if st.button("Measure Noise Floor", type="primary"):
        with st.spinner("Measuring Noise..."):
             try:
                # Config Scope: 100kS/s, Selected Coupling
                scope.configure_channel('A', True, cal_range, cal_coupling)
                # Block capture
                sample_rate = 100000.0 # 100k is good bandwidth
                num_samples = int(sample_rate * cal_duration)
                
                tb_idx = scope.calculate_timebase_index(cal_duration, num_samples)
                
                times, volts = scope.capture_block(tb_idx, num_samples)
                
                if len(times) > 0:
                     st.session_state.cal_times = times
                     st.session_state.cal_volts = volts
                     st.session_state.cal_params = {
                         'r_source': source_r_ohms,
                         'gain': cal_gain
                     }
                     st.rerun()
                else:
                    st.error("Capture returned empty.")
             except Exception as e:
                 st.error(f"Error: {e}")

    # Persistent Analysis Block
    if 'cal_times' in st.session_state:
        times = st.session_state.cal_times
        volts = st.session_state.cal_volts
        params = st.session_state.cal_params
        
        r_val = params.get('r_source', source_r_ohms)
        g_val = params.get('gain', cal_gain)
        
        # Calculate Theoretical Noise
        k_B = 1.380649e-23
        T = 300.0
        
        i_n_th = np.sqrt(4 * k_B * T / r_val) # A/rtHz
        v_n_th = i_n_th * g_val # V/rtHz (at output)
        
        st.divider()
        st.write(f"**Theoretical Estimates** (R={r_val:.2e}Ω, T=300K):")
        st.write(f"- Johnson Current Noise: **{i_n_th*1e12:.3f} pA/√Hz**")
        
        # Smart Unit Formatting
        if v_n_th < 1e-6:
            v_disp = f"{v_n_th*1e9:.2f} nV/√Hz"
        elif v_n_th < 1e-3:
            v_disp = f"{v_n_th*1e6:.2f} µV/√Hz"
        else:
            v_disp = f"{v_n_th*1e3:.2f} mV/√Hz"
            
        st.write(f"- Exp. Output Noise: **{v_disp}**")
        
        if g_val > 1e9:
            st.warning("⚠️ High Gain (>1G) usually implies Low Bandwidth (<1kHz). Ensure you measure noise inside the TIA's bandwidth!")
        
        # Calculate Measured PSD
        from scipy.signal import welch
        fs = 1.0 / (times[1] - times[0])
        f, Pxx = welch(volts, fs, nperseg=min(len(volts), 4096))
        asd_meas = np.sqrt(Pxx)
        
        # Plot Frame
        df_cal = pd.DataFrame({'Freq (Hz)': f, 'Measured Noise (V/rtHz)': asd_meas})
        df_cal['Theoretical (Johnson)'] = v_n_th
        
        # Filter Display Range
        df_cal = df_cal[(df_cal['Freq (Hz)'] > 5) & (df_cal['Freq (Hz)'] < fs/2.1)]
        
        fig_cal = px.line(df_cal, x='Freq (Hz)', y=['Measured Noise (V/rtHz)', 'Theoretical (Johnson)'], 
                          log_x=True, log_y=True, title="Noise Calibration: Measured vs Theoretical")
        st.plotly_chart(fig_cal, use_container_width=True)
        
        # Metrics (Interactive Band)
        c_b1, c_b2 = st.columns(2)
        with c_b1: 
            f_start = st.number_input("Analysis Freq Start (Hz)", value=10.0, step=10.0)
        with c_b2:
            f_stop = st.number_input("Analysis Freq Stop (Hz)", value=100.0 if g_val > 1e9 else 10000.0, step=100.0)
        
        mask = (df_cal['Freq (Hz)'] >= f_start) & (df_cal['Freq (Hz)'] <= f_stop)
        if mask.any():
            median_noise = df_cal[mask]['Measured Noise (V/rtHz)'].median()
            
            if median_noise < 1e-3:
                meas_disp = f"{median_noise*1e6:.1f} µV/√Hz"
            else:
                meas_disp = f"{median_noise*1e3:.1f} mV/√Hz"
                
            st.metric(f"Measured Floor ({int(f_start)}-{int(f_stop)} Hz)", meas_disp)
            
            excess = median_noise / v_n_th
            st.metric("Excess Noise Factor", f"{excess:.2f}x", delta_color="inverse")
            
            if excess < 0.5:
                st.error("📉 Result too LOW (< 0.5x). Bandwidth limit reached? Check TIA BW.")
            elif excess < 1.0:
                st.success("✅ Below Thermal Limit? Possible roll-off. Or Gain lower than expected.")
            elif excess < 2.0:
                st.success("✅ Good Agreement (< 2x Thermal)")
            else:
                st.warning("⚠️ High Excess Noise (> 2x). Check shielding/grounding.")

with detect_tab:
    st.header("Specific Detectivity ($D^*$) Calculator & Measurement")
    st.markdown("""
    Specific Detectivity ($D^*$) is a measure of the signal-to-noise performance of a photodetector, normalized for the device area.
    $$D^* = \\frac{\\sqrt{A}}{NEP}$$
    where $A$ is the active area and $NEP$ is the Noise Equivalent Power ($W/\\sqrt{Hz}$).
    """)
    
    # --- SETUP & PARAMS ---
    st.subheader("1. Device & TIA Parameters")
    col1, col2 = st.columns(2)
    with col1:
        d_area = st.number_input("Active Area (cm²)", step=0.1, format="%.4f", key="det_area", value=st.session_state.get('det_area', 1.0))
        d_resp = st.number_input("Responsivity (A/W)", step=0.1, key="det_resp", value=st.session_state.get('det_resp', 0.5))
    with col2:
        d_gain = st.number_input("TIA Gain (V/A)", format="%.2e", key="det_gain", value=st.session_state.get('det_gain', 1e6))
        d_input_mode = st.radio("Noise Source", ["Live Capture", "Manual Entry"], horizontal=True, key="det_input_mode")

    st.divider()
    
    # --- NOISE MEASUREMENT ---
    st.subheader("2. Noise Measurement")
    measured_v_noise = 0.0
    
    if d_input_mode == "Live Capture":
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            det_range = st.selectbox("Scope Range", ['10MV', '20MV', '50MV', '100MV', '200MV', '500MV', '1V', '2V'], index=0, key="det_range")
        with mcol2:
            det_coupling = st.selectbox("Coupling", ["AC", "DC"], index=0, key="det_coup")
        with mcol3:
            det_duration = st.number_input("Duration (s)", step=0.5, key="det_dur", value=st.session_state.get('det_dur', 1.0))
            
        if st.button("Measure Detector Noise", type="primary", key="det_meas_btn"):
            with st.spinner("Capturing Noise..."):
                try:
                    scope.configure_channel('A', True, det_range, det_coupling)
                    fs = 100000.0
                    num_samples = int(fs * det_duration)
                    tb_idx = scope.calculate_timebase_index(det_duration, num_samples)
                    t_vec, v_vec = scope.capture_block(tb_idx, num_samples)
                    
                    if len(t_vec) > 0:
                        st.session_state.det_noise_data = {'t': t_vec, 'v': v_vec}
                        st.rerun()
                    else:
                        st.error("Capture failed.")
                except Exception as e:
                    st.error(f"Error: {e}")
                    
        if 'det_noise_data' in st.session_state:
            data = st.session_state.det_noise_data
            from scipy.signal import welch
            fs_actual = 1.0 / (data['t'][1] - data['t'][0])
            f_psd, p_psd = welch(data['v'], fs_actual, nperseg=min(len(data['v']), 4096))
            asd_v = np.sqrt(p_psd)
            
            # Band Selection
            st.markdown("---")
            st.write("### Analysis Band")
            b1, b2 = st.columns(2)
            with b1:
                f_s = st.number_input("Start Freq (Hz)", value=10.0, step=10.0, key="det_f1")
            with b2:
                f_e = st.number_input("End Freq (Hz)", value=1000.0, step=100.0, key="det_f2")
                
            df_psd = pd.DataFrame({'Freq (Hz)': f_psd, 'Voltage Noise (V/rtHz)': asd_v})
            df_psd = df_psd[(df_psd['Freq (Hz)'] > 1) & (df_psd['Freq (Hz)'] < fs_actual/2.1)]
            
            fig_psd = px.line(df_psd, x='Freq (Hz)', y='Voltage Noise (V/rtHz)', log_x=True, log_y=True, title="Detector Noise PSD")
            fig_psd.add_vrect(x0=f_s, x1=f_e, fillcolor="rgba(0,100,255,0.1)", line_width=1, annotation_text="Selected Band")
            st.plotly_chart(fig_psd, use_container_width=True)
            
            # Extract Value
            mask = (df_psd['Freq (Hz)'] >= f_s) & (df_psd['Freq (Hz)'] <= f_e)
            if mask.any():
                measured_v_noise = df_psd[mask]['Voltage Noise (V/rtHz)'].median()
                st.info(f"Representative Noise (Median in band): **{measured_v_noise*1e6:.2f} µV/√Hz**")
    else:
        # Manual Entry
        measured_v_noise = st.number_input("Voltage Noise Density (V/√Hz)", value=1e-6, format="%.2e", key="det_manual_v")
            
    st.divider()
    
    # --- D* RESULTS ---
    st.subheader("3. Final Detectivity")
    
    # Calculations
    d_i_noise = measured_v_noise / d_gain if d_gain > 0 else 0
    nep = d_i_noise / d_resp if d_resp > 0 else 0
    d_star = np.sqrt(d_area) / nep if nep > 0 else 0
    
    res1, res2, res3 = st.columns(3)
    res1.metric("Measured Current Noise", f"{d_i_noise*1e12:.2f} pA/√Hz")
    res2.metric("NEP", f"{nep:.2e} W/√Hz")
    res3.metric("Detectivity (D*)", f"{d_star:.2e} Jones")
    
    if d_star > 1e12:
        st.success("✨ High Detectivity (> 10¹² Jones). Excellent performance.")
    elif d_star > 1e10:
        st.info("ℹ️ Moderate Detectivity. Typical for many commercial detectors.")
    elif d_star > 0:
        st.warning("⚠️ Low Detectivity. Check noise floor and responsivity.")

    st.divider()
    st.subheader("4. Export Data")
    exp_name = st.text_input("Experiment Name", value="Detectivity_Run_1")
    
    if st.button("💾 Export Measurement Data", use_container_width=True):
        if 'det_noise_data' not in st.session_state:
            st.error("No measurement data to export. Please 'Measure Detector Noise' first.")
        else:
            try:
                # Prepare Metadata
                meta = {
                    'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Active_Area_cm2': d_area,
                    'Responsivity_AW': d_resp,
                    'TIA_Gain_VA': d_gain,
                    'Scope_Range': det_range if d_input_mode == "Live Capture" else "N/A",
                    'Coupling': det_coupling if d_input_mode == "Live Capture" else "N/A",
                    'Duration_s': det_duration if d_input_mode == "Live Capture" else "N/A",
                    'Band_Start_Hz': f_s if d_input_mode == "Live Capture" else "N/A",
                    'Band_End_Hz': f_e if d_input_mode == "Live Capture" else "N/A",
                    'D_Star_Jones': f"{d_star:.2e}",
                    'NEP_W_rtHz': f"{nep:.2e}",
                    'Current_Noise_A_rtHz': f"{d_i_noise:.2e}",
                    'Analysis_Mode': d_input_mode
                }
                
                # Paths
                base_dir = "data/commissioning"
                os.makedirs(base_dir, exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 1. Raw Trace
                raw_path = os.path.join(base_dir, f"{exp_name}_{ts}_raw.csv")
                df_raw = pd.DataFrame({'time': st.session_state.det_noise_data['t'], 'voltage': st.session_state.det_noise_data['v']})
                
                # 2. FFT Spectrum
                fft_path = os.path.join(base_dir, f"{exp_name}_{ts}_fft.csv")
                df_fft = pd.DataFrame({'freq_hz': f_psd, 'voltage_noise_V_rtHz': asd_v})
                
                def save_csv_with_meta(path, df, metadata):
                    with open(path, 'w') as f:
                        for k, v in metadata.items():
                            f.write(f"# {k}: {v}\n")
                        df.to_csv(f, index=False)
                
                save_csv_with_meta(raw_path, df_raw, meta)
                save_csv_with_meta(fft_path, df_fft, meta)
                
                st.success(f"Successfully exported to:\n- {raw_path}\n- {fft_path}")
                
            except Exception as e:
                st.error(f"Export failed: {e}")
