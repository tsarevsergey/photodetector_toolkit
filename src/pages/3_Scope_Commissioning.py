
import streamlit as st
import time
import sys
import os
import plotly.express as px
import pandas as pd
import numpy as np

# Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hardware.scope_controller import ScopeController, InstrumentState

st.set_page_config(page_title="Scope Commissioning", layout="wide")
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
    st.header("Scope Connection")
    mock_mode = st.checkbox("Mock Mode", value=False)
    
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
config_tab, capture_tab = st.tabs(["Configuration", "Capture & View"])

with config_tab:
    st.subheader("Channel Setup")
    c1, c2 = st.columns(2)
    
    ranges = ['10MV', '20MV', '50MV', '100MV', '200MV', '500MV', '1V', '2V', '5V', '10V', '20V']
    
    with c1:
        st.markdown("### Channel A")
        en_a = st.checkbox("Enable Ch A", value=True)
        range_a = st.selectbox("Range A", ranges, index=7) # 2V default
        coup_a = st.selectbox("Coupling A", ["DC", "AC"], index=0)
        
        if st.button("Apply Ch A"):
            scope.configure_channel('A', en_a, range_a, coup_a)
            st.success("Ch A Configured")
            
    with c2:
        st.markdown("### Channel B")
        en_b = st.checkbox("Enable Ch B", value=False)
        range_b = st.selectbox("Range B", ranges, index=7)
        coup_b = st.selectbox("Coupling B", ["DC", "AC"], index=0)
        
        if st.button("Apply Ch B"):
            scope.configure_channel('B', en_b, range_b, coup_b)
            st.success("Ch B Configured")

with capture_tab:
    st.subheader("Capture")
    
    # 1. Acquisition Settings
    c_mode, c_dur = st.columns(2)
    
    with c_mode:
        acq_mode = st.radio("Acquisition Mode", ["Block", "Streaming"], index=0, help="Block: Short, high speed. Streaming: Long, continuous.")
        
    with c_dur:
        if acq_mode == "Block":
            duration_ms = st.number_input("Duration (ms)", value=20.0, min_value=0.1, max_value=5000.0, step=10.0)
            duration_s = duration_ms / 1000.0
        else:
            duration_s = st.number_input("Duration (s)", value=2.0, min_value=0.1, max_value=100.0, step=0.5)

    # 2. Resolution Settings
    c_res1, c_res2 = st.columns(2)
    
    with c_res1:
        if acq_mode == "Block":
            num_samples = st.number_input("Number of Samples", value=2000, min_value=100, max_value=20000)
        else:
            sample_rate = st.number_input("Sample Rate (Hz)", value=100000.0, min_value=1000.0, max_value=1000000.0, step=10000.0)
            
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
        quick_coup = st.selectbox("Coupling", ["DC", "AC"], index=0, key="quick_coup")
    with qc2:
        quick_range = st.selectbox("Range", ['10MV', '20MV', '50MV', '100MV', '200MV', '500MV', '1V', '2V', '5V', '10V'], index=7, key="quick_range")
    with qc3:
        tia_gain = st.number_input("TIA Gain (Ω)", value=1000.0, format="%.2e", help="Load Resistor or TIA Gain for noise conversion")

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
                    st.session_state.comm_acq_mode = acq_mode
                    st.session_state.comm_gain = tia_gain
                else:
                    st.warning("No data returned.")
            except Exception as e:
                st.error(f"Capture Error: {e}")

    # --- Display Section (Persists independently of button) ---
    if 'comm_times' in st.session_state:
        times = st.session_state.comm_times
        volts = st.session_state.comm_volts
        acq_mode_disp = st.session_state.comm_acq_mode
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
