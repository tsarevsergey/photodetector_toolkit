
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
    st.subheader("Block Capture")
    
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        # User Friendly: Duration
        duration_ms = st.number_input("Duration (ms)", value=20.0, min_value=0.1, max_value=5000.0, step=10.0, help="Total time window to capture")
        
    with sc2:
        num_samples = st.number_input("Number of Samples", value=2000, min_value=100, max_value=20000)
        
    with sc3:
        # Calculate TB index for display
        if scope and st.session_state.scope_connected:
             tb_idx = scope.calculate_timebase_index(duration_ms/1000.0, num_samples)
             st.metric("Calc. Interval", f"{(duration_ms/num_samples)*1000:.2f} µs")
             st.caption(f"Timebase Index: {tb_idx}")
        
    if st.button("Capture Block", type="primary"):
        with st.spinner("Capturing..."):
            try:
                # Calculate correct index
                tb_index = scope.calculate_timebase_index(duration_ms/1000.0, num_samples)
                
                times, volts = scope.capture_block(int(tb_index), int(num_samples))
                
                if len(times) > 0:
                    st.success(f"Captured {len(times)} samples")
                    
                    df = pd.DataFrame({'Time (s)': times, 'Signal (V)': volts})
                    
                    fig = px.line(df, x='Time (s)', y='Signal (V)', title="Scope Trace")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Stats
                    st.write("### Signal Statistics")
                    st.write(df['Signal (V)'].describe().T)
                else:
                    st.warning("No data returned.")
            except Exception as e:
                st.error(f"Capture Error: {e}")
