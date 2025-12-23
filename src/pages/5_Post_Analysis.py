import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import glob
from scipy import signal

# Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis.calibration import CalibrationManager

st.set_page_config(page_title="Post-Processing & Analysis", layout="wide")
st.title("📊 Post-Processing & Analysis")

# Tabs for different analysis modes
tab_cal, tab_ldr, tab_trace = st.tabs(["1. LED Calibration", "2. LDR Analysis", "3. Trace Analysis"])

# --- TAB 1: CALIBRATION GENERATOR ---
with tab_cal:
    st.header("Generate LED Power Calibration")
    st.markdown("""
    Create a mapping between **LED Current** and **Optical Power** using a calibrated Reference Diode.
    
    **Formula:** $P_{opt} = I_{ref} / R_{ref}(\lambda)$
    """)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("1. Reference Diode Data")
        data_files = glob.glob("data/*.csv") + glob.glob("*.csv")
        ref_file = st.selectbox("Reference Responsivity File (Wavelength vs A/W)", options=data_files, index=0 if data_files else None, key='an_cal_ref_sel')
        
    with c2:
         st.subheader("2. LED Measurement")
         meas_file = st.selectbox("Reference Measurement File (LED Current vs Photocurrent)", options=data_files, index=0 if data_files else None, key='an_cal_meas_sel')
         
    st.subheader("3. Settings")
    c_set1, c_set2 = st.columns(2)
    with c_set1:
        cal_wl = st.number_input("LED Emission Wavelength (nm)", value=461.0, min_value=200.0, max_value=2000.0)
    with c_set2:
        ref_area = st.number_input("Reference Diode Active Area (cm²)", value=1.0, format="%.4f")
    
    if st.button("Generate Calibration Curve", type="primary"):
        if not ref_file or not meas_file:
            st.error("Please select valid files.")
        else:
            try:
                cal_mgr = CalibrationManager()
                df_cal, r_val = cal_mgr.generate_led_calibration(meas_file, ref_file, cal_wl)
                
                # Fit Power Law
                A, B, r2 = cal_mgr.fit_led_power_law(df_cal)
                
                # Store in session state for other tabs
                st.session_state.active_calibration = df_cal
                st.session_state.active_calibration_meta = {'wl': cal_wl, 'r_ref': r_val, 'fit_A': A, 'fit_B': B, 'fit_r2': r2, 'ref_area': ref_area}
                
                st.success(f"Calibration Generated! R_ref({cal_wl}nm) = {r_val:.4f} A/W")
                st.info(f"**Power Model (Extrapolation Enabled):** $P_{{opt}} = {A:.4e} \\cdot I_{{LED}}^{{{B:.4f}}}$ ($R^2={r2:.4f}$)")
                
                # Plot with Fit
                x_fit = np.logspace(np.log10(df_cal['LED_Current_A'].min()), np.log10(df_cal['LED_Current_A'].max()), 100)
                y_fit = A * (x_fit**B)
                
                fig = px.scatter(df_cal, x="LED_Current_A", y="Optical_Power_W", 
                                 log_x=True, log_y=True,
                                 title="LED Calibration Curve: Power vs Current")
                fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name=f'Power Law Fit', line=dict(dash='dash')))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show Data
                st.dataframe(df_cal.style.format({
                    "LED_Current_A": "{:.2e}",
                    "Optical_Power_W": "{:.2e}",
                    "Photocurrent_A": "{:.2e}"
                }))
                
            except Exception as e:
                st.error(f"Error: {e}")

# --- TAB 2: LDR ANALYSIS ---
with tab_ldr:
    st.header("LDR Data Analysis")
    
    if 'active_calibration' not in st.session_state:
        st.warning("⚠️ No Calibration loaded. Please generate one in the 'LED Calibration' tab first.")
    else:
        st.success(f"Using Calibration (WL={st.session_state.active_calibration_meta['wl']}nm)")
    
    st.subheader("Load DUT Data")
    # File selector for DUT data
    c_dut1, c_dut2 = st.columns(2)
    with c_dut1:
         dut_file = st.selectbox("Select LDR Measurement File", options=data_files, key='an_dut_sel')
    with c_dut2:
         dev_area = st.number_input("Device Active Area (cm²)", value=1.0, format="%.4f")
    
    if st.button("Analyze LDR Data"):
        if not dut_file:
            st.error("No file selected.")
        else:
            try:
                df_raw = pd.read_csv(dut_file)
                
                # --- DATA CLEANING ---
                # 1. Sort by File Index (Time) to ensure 'last' is 'latest'
                
                # 2. Group by Control Variable (LED_Current_A)
                if 'LED_Current_A' not in df_raw.columns:
                     st.error("Column 'LED_Current_A' missing.")
                     st.stop()
                     
                unique_currents = df_raw['LED_Current_A'].unique()
                cleaned_rows = []
                
                groups = df_raw.groupby('LED_Current_A')
                
                for amp, group in groups:
                    selected_row = None
                    # Logic: Prioritize OK
                    if 'SNR_Status' in group.columns:
                        mask_ok = group['SNR_Status'].astype(str).str.contains('OK|High|Good', case=False, na=False)
                        ok_group = group[mask_ok]
                        if not ok_group.empty:
                            selected_row = ok_group.iloc[-1] # Latest OK
                        else:
                            selected_row = group.iloc[-1] # Latest
                    else:
                        selected_row = group.iloc[-1] # Latest
                        
                    cleaned_rows.append(selected_row)
                    
                df_dut = pd.DataFrame(cleaned_rows).reset_index(drop=True)
                
                # Sort Highest Power first (Descending Current)
                df_dut = df_dut.sort_values(by='LED_Current_A', ascending=False)
                
                st.write(f"Processed {len(df_raw)} raw points -> {len(df_dut)} unique steps.")
                
                # Validate columns
                if 'Photocurrent_A' not in df_dut.columns:
                    st.error("File must contain 'LED_Current_A' and 'Photocurrent_A' columns.")
                    st.stop()
                
                # Calculate Current Density (J)
                df_dut['Current_Density_A_cm2'] = df_dut['Photocurrent_A'] / dev_area
                    
                # Apply Calibration
                if 'active_calibration_meta' in st.session_state:
                    meta = st.session_state.active_calibration_meta
                    
                    # Use Fitted Model
                    if 'fit_A' in meta and meta.get('fit_r2', 0) > 0.9:
                        A, B = meta['fit_A'], meta['fit_B']
                        p_opt_ref = A * (df_dut['LED_Current_A'] ** B)
                        st.caption(f"Using Fitted Power Model: $P_{{Ref}} = {A:.2e} \\cdot I^{{{B:.3f}}}$")
                    else:
                        cal_df = st.session_state.active_calibration
                        cal_df = cal_df.sort_values('LED_Current_A')
                        p_opt_ref = np.interp(df_dut['LED_Current_A'], cal_df['LED_Current_A'], cal_df['Optical_Power_W'])
                        st.warning("Using Linear Interpolation (No Extrapolation)")
                    
                    # SCALE by Area Ratio (P_dut = P_ref * (A_dut / A_ref))
                    ref_area_val = meta.get('ref_area', 1.0)
                    area_ratio = dev_area / ref_area_val
                    
                    p_opt = p_opt_ref * area_ratio
                    # st.info(f"Applying Area Scaling: A_dut ({dev_area}) / A_ref ({ref_area_val}) = {area_ratio:.2f}")
                    
                    df_dut['Optical_Power_W'] = p_opt
                    
                    # Calculate Sensitivity (SNR=3)
                    if 'SNR_FFT' in df_dut.columns:
                         # P_min = P_meas * (3 / SNR)
                        df_dut['Sensitivity_W_SNR3'] = df_dut.apply(lambda r: r['Optical_Power_W'] * (3.0 / r['SNR_FFT']) if r['SNR_FFT'] > 0 else np.nan, axis=1)
                        
                    # Fit Slope
                    # Fit Slope for J vs P (Responsivity Density?)
                    # Model: J = R_dens * P (intercept 0)
                    # slope = sum(x*y)/sum(x^2)
                    x_fit_data = df_dut['Optical_Power_W'].values
                    y_fit_data = df_dut['Current_Density_A_cm2'].abs().values
                    
                    slope_dens = np.sum(x_fit_data * y_fit_data) / np.sum(x_fit_data**2)
                    intercept = 0.0
                    r2 = 0.0 # simple placeholder or calc if needed
                    
                    # --- CALC: NEP ---
                    # NEP (W/rtHz) = Noise Current Density (A/rtHz) / Responsivity (A/W)
                    # We need Global Responsivity (A/W), not Density yet.
                    # R_global (A/W) = Slope (A/W/cm2) * Area (cm2)?
                    # OR we just fit I vs P directly.
                    # Let's fit I vs P directly for R_global to be safe.
                    
                    x_fit_I = df_dut['Optical_Power_W'].values
                    y_fit_I = df_dut['Photocurrent_A'].abs().values
                    slope_global = np.sum(x_fit_I * y_fit_I) / np.sum(x_fit_I**2)
                    
                    if slope_global > 0:
                         # Ensure we have Resistance info
                         if 'Resistance_Ohms' in df_dut.columns and 'Noise_Density_V_rtHz' in df_dut.columns:
                             df_dut['Current_Noise_Dens_A_rtHz'] = df_dut['Noise_Density_V_rtHz'] / df_dut['Resistance_Ohms']
                             df_dut['NEP_W_rtHz'] = df_dut['Current_Noise_Dens_A_rtHz'] / slope_global
                         else:
                             df_dut['NEP_W_rtHz'] = np.nan
                    else:
                         df_dut['NEP_W_rtHz'] = np.nan

                    # --- REPORT ---
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Responsivity", f"{slope_dens:.4f} A/W/cm²") 
                    k2.metric("Abs. Responsivity", f"{slope_global:.4f} A/W")
                    k3.metric("Linearity (Fit)", "Power Law") 
                    
                    min_nep = df_dut['NEP_W_rtHz'].min() if 'NEP_W_rtHz' in df_dut.columns else 0
                    k4.metric("Best NEP", f"{min_nep:.2e} W/√Hz")
                    
                    # --- PLOTS ---
                    c_p1, c_p2 = st.columns(2)
                    
                    with c_p1:
                        # LDR Plot with Fit (Using J)
                        fig_ldr = px.scatter(df_dut, x="Optical_Power_W", y="Current_Density_A_cm2", 
                                             log_x=True, log_y=True,
                                             title=f"LDR: Current Density vs Optical Power (Area={dev_area} cm²)")
                        # Use logspace for smooth line on log-log plot
                        x_min = df_dut['Optical_Power_W'].min()
                        x_max = df_dut['Optical_Power_W'].max()
                        if x_min <= 0: x_min = 1e-12 
                        x_range = np.geomspace(x_min, x_max, 100)
                        y_fit = slope_dens * x_range # + intercept (0)
                        fig_ldr.add_trace(go.Scatter(x=x_range, y=y_fit, mode='lines', name='Linear Fit', line=dict(dash='dash', color='red')))
                        st.plotly_chart(fig_ldr, use_container_width=True)
                        
                    with c_p2:
                        # Sensitivity / NEP Plot
                        # Switch between NEP and Min Power? Default to NEP as requested.
                        if 'NEP_W_rtHz' in df_dut.columns:
                             # Filter valid
                             df_nep = df_dut[df_dut['NEP_W_rtHz'] > 0]
                             fig_nep = px.line(df_nep, x="Optical_Power_W", y="NEP_W_rtHz",
                                               log_x=True, log_y=True,
                                               title="NEP vs Optical Power",
                                               labels={'NEP_W_rtHz': 'NEP (W/√Hz)', 'Optical_Power_W': 'Optical Power (W)'})
                             st.plotly_chart(fig_nep, use_container_width=True)
                        
                    st.dataframe(df_dut.style.format({
                        "LED_Current_A": "{:.2e}",
                        "Optical_Power_W": "{:.2e}",
                        "Photocurrent_A": "{:.2e}",
                        "Current_Density_A_cm2": "{:.2e}",
                        "NEP_W_rtHz": "{:.2e}",
                        "Sensitivity_W_SNR3": "{:.2e}",
                        "SNR_Time": "{:.1f}",
                        "Scope_Vpp": "{:.2e}"
                    }, na_rep="-"))
                    
                else:
                    st.warning("No calibration active. Plotting raw Current Density.")
                    fig = px.scatter(df_dut, x="LED_Current_A", y="Current_Density_A_cm2", log_x=True, log_y=True)
                    st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Analysis Failed: {e}")

# --- TAB 3: TRACE ANALYSIS ---
with tab_trace:
    st.header("Raw Oscilloscope Trace Analysis")
    st.markdown("Load saved raw traces (`.csv`) to analyze Noise Spectral Density.")
    
    # Trace Selection (Recursive find)
    trace_files = glob.glob("data/**/*.csv", recursive=True) 
    
    if not trace_files:
        st.warning("No CSV files found in 'data/' subfolders. Run a sweep with 'Save Raw Traces' enabled.")
    
    trace_file = st.selectbox("Select Trace File", options=trace_files, key='trace_sel_box')
    
    c_t1, c_t2 = st.columns(2)
    with c_t1:
        load_res = st.number_input("Gain (Resistor) Used (Ω)", value=47000.0, help="Required to convert Voltage Noise to Current Noise.")
    with c_t2:
        # Window
        fs_override = st.number_input("Sampling Rate Override (Hz)", value=0.0, help="Leave as 0.0 to Auto-Detect from time data.")
        
    if st.button("Analyze Trace"):
        if not trace_file:
            st.error("No file selected.")
        else:
            try:
                df_trace = pd.read_csv(trace_file)
                if 'time' not in df_trace.columns or 'voltage' not in df_trace.columns:
                     st.error("Invalid trace format. Needs 'time' and 'voltage' columns.")
                     st.stop()
                     
                t = df_trace['time'].values
                v = df_trace['voltage'].values
                
                # Plot Time Domain
                fig_time = px.line(df_trace, x='time', y='voltage', title=f"Time Domain: {os.path.basename(trace_file)}")
                st.plotly_chart(fig_time, use_container_width=True)
                
                # PSD Analysis
                # 1. Determine Fs
                if fs_override > 0:
                    fs = fs_override
                else:
                    if len(t) > 1:
                        dt = np.mean(np.diff(t))
                        fs = 1.0 / dt
                    else:
                        st.error("Cannot determine Sampling Rate.")
                        st.stop()
                        
                # Display Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Detected Sampling Rate", f"{fs/1000:.2f} kHz")
                m2.metric("Trace Duration", f"{t[-1]:.4f} s")
                m3.metric("Data Points", f"{len(v)}")
                
                # 2. Compute PSD (Welch)
                # FIX: Previous nperseg=4096 with fs=138kHz gave ~33Hz resolution.
                # 80Hz would be blurred.
                # Use Full Length for High Resolution of periodic signals.
                
                # Resolution = fs / nperseg
                # With nperseg = 16000, res = 138000/16000 = 8.6Hz. This is better.
                freqs, psd_v2 = signal.welch(v, fs, nperseg=len(v), window='hann')
                
                # 3. Convert to Spectral Density (V / rtHz)
                asd_v = np.sqrt(psd_v2) 
                
                # 4. Convert to Current Noise Density (A / rtHz)
                # I_n = V_n / R
                asd_i = asd_v / load_res
                
                # Plot PSD
                df_psd = pd.DataFrame({
                    "Frequency (Hz)": freqs,
                    "Current Noise Density (A/√Hz)": asd_i,
                    "Voltage Noise Density (V/√Hz)": asd_v
                })
                
                # Remove DC component (f=0)
                df_psd = df_psd[df_psd["Frequency (Hz)"] > 0]
                
                fig_psd = px.line(df_psd, x="Frequency (Hz)", y="Current Noise Density (A/√Hz)", 
                                  log_x=True, log_y=True, 
                                  title=f"Current Noise Spectral Density (Load = {load_res} Ω)")
                st.plotly_chart(fig_psd, use_container_width=True)
                
            except Exception as e:
                st.error(f"Analysis Failed: {e}")
