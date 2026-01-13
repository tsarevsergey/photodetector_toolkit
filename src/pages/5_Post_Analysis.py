import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import glob
from scipy import signal
from scipy.interpolate import interp1d

# Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis.calibration import CalibrationManager
from utils.settings_manager import SettingsManager
from utils.ui_components import render_global_sidebar

st.set_page_config(page_title="Post-Processing & Analysis", layout="wide")
settings = SettingsManager()

def sync_setting(st_key, setting_key):
    """Callback to sync session state with persistent settings."""
    settings.set(setting_key, st.session_state[st_key])

# --- Init Session State ---
def init_pref(key, setting_key=None):
    if setting_key is None: setting_key = key
    if key not in st.session_state:
        st.session_state[key] = settings.get(setting_key)

init_pref("last_ref_file")
init_pref("last_meas_file")
init_pref("last_trace_file")
init_pref("p_cal_wl", "last_led_wavelength")
init_pref("p_ref_area", "last_ref_area")
init_pref("p_dut_area", "last_dut_area")

render_global_sidebar(settings)
st.title("📊 Post-Processing & Analysis")

# Tabs for different analysis modes
tab_cal, tab_ldr, tab_trace = st.tabs(["1. LED Calibration", "2. LDR Analysis", "3. Trace Analysis"])

# --- Global File Discovery ---
# File Filtering Option (Global, impacts Cal/LDR tabs)
hide_traces = st.checkbox("🔍 Hide 'trace_step' files (raw data)", value=True, help="Omit files with 'trace_step' in the name.")

# Recursive Discovery
base_dir = settings.get("base_save_folder", "data")
all_files = glob.glob(os.path.join(base_dir, "**/*.csv"), recursive=True) + glob.glob(os.path.join(base_dir, "*.csv"))
all_files = sorted(list(set([os.path.normpath(f) for f in all_files])))

if hide_traces:
    data_files = [f for f in all_files if "trace_step" not in f.lower()]
else:
    data_files = all_files

def get_index(path, options):
    if not path: return 0
    norm_path = os.path.normpath(path)
    try:
        return options.index(norm_path)
    except ValueError:
        return 0

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
        ref_file = st.selectbox("Reference Responsivity File (Wavelength vs A/W)", 
                                options=data_files, 
                                index=get_index(st.session_state.last_ref_file, data_files), 
                                key='last_ref_file',
                                on_change=sync_setting, args=('last_ref_file', 'last_ref_file'))
        
    with c2:
         st.subheader("2. Source Measurements")
         num_cal = st.number_input("Number of Calibration Curves", min_value=1, max_value=10, value=1)
         
         cal_entries = []
         for i in range(num_cal):
             cc1, cc2 = st.columns([3, 1])
             with cc1:
                 # Keyed selection to allow different files per row
                 row_file = st.selectbox(f"File {i+1}", options=data_files, key=f"cal_file_{i}")
             with cc2:
                 row_od = st.number_input(f"OD {i+1}", min_value=0.0, max_value=10.0, step=0.1, value=0.0, key=f"cal_od_{i}")
             cal_entries.append({'file': row_file, 'od': row_od})
         
    st.subheader("3. Settings")
    c_set1, c_set2 = st.columns(2)
    with c_set1:
        cal_wl = st.number_input("LED Emission Wavelength (nm)", min_value=200.0, max_value=2000.0, key="p_cal_wl", value=st.session_state.p_cal_wl, on_change=sync_setting, args=("p_cal_wl", "last_led_wavelength"))
    with c_set2:
        ref_area = st.number_input("Reference Diode Active Area (cm²)", format="%.4f", key="p_ref_area", value=st.session_state.p_ref_area, on_change=sync_setting, args=("p_ref_area", "last_ref_area"))
    
    if st.button("Generate Calibration Curve", type="primary"):
        if not ref_file or not any(e['file'] for e in cal_entries):
            st.error("Please select valid files.")
        else:
            try:
                cal_mgr = CalibrationManager()
                df_cal, r_val, segment_fits = cal_mgr.generate_multi_led_calibration(cal_entries, ref_file, cal_wl)
                
                # Fit Global Power Law (for generic comparison on SOURCE power)
                temp_global = df_cal[['LED_Current_A', 'Source_Power_W']].rename(columns={'Source_Power_W': 'Optical_Power_W'})
                A_glob, B_glob, r2_glob = cal_mgr.fit_led_power_law(temp_global)
                
                # Store in session state for other tabs
                st.session_state.active_calibration = df_cal
                st.session_state.active_calibration_meta = {
                    'wl': cal_wl, 'r_ref': r_val, 
                    'fit_A_glob': A_glob, 'fit_B_glob': B_glob, 'fit_r2_glob': r2_glob, 
                    'ref_area': ref_area,
                    'segment_fits': segment_fits
                }
                
                st.success(f"Calibration Generated! R_ref({cal_wl}nm) = {r_val:.4f} A/W")
                st.info(f"**Global Source Power Model (OD 0):** $P_{{source}} = {A_glob:.4e} \\cdot I_{{LED}}^{{{B_glob:.4f}}}$ ($R^2={r2_glob:.4f}$)")
                
                # Plot RAW curves
                fig = px.scatter(df_cal, x="LED_Current_A", y="Measured_Power_W", 
                                 color="Source_Segment" if "Source_Segment" in df_cal.columns else None,
                                 log_x=True, log_y=True,
                                 title="Individual LED Calibration Curves (Measured at Diode)")
                
                # Add individual fits
                for seg_id, fit in segment_fits.items():
                    seg_data = df_cal[df_cal['Source_Segment'] == seg_id]
                    x_f = np.geomspace(seg_data['LED_Current_A'].min(), seg_data['LED_Current_A'].max(), 50)
                    y_f = fit['A_meas'] * (x_f ** fit['B'])
                    fig.add_trace(go.Scatter(x=x_f, y=y_f, mode='lines', name=f'Seg {seg_id} (OD {fit["od"]}) Fit'))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show Data
                df_cal['Photocurrent_Density_A_cm2'] = df_cal['Photocurrent_A'].abs() / ref_area
                st.dataframe(df_cal.style.format({
                    "LED_Current_A": "{:.2e}",
                    "Photocurrent_A": "{:.2e}",
                    "Photocurrent_Density_A_cm2": "{:.2e}",
                    "Measured_Power_W": "{:.2e}",
                    "Source_Power_W": "{:.2e}"
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
    c_dut_set, c_dut_area, c_dut_mode = st.columns([1, 1, 1])
    with c_dut_set:
        num_dut = st.number_input("Number of DUT Measurements", min_value=1, max_value=12, value=1)
    with c_dut_area:
        dev_area = st.number_input("Device Active Area (cm²)", format="%.4f", key="p_dut_area", value=st.session_state.p_dut_area, on_change=sync_setting, args=("p_dut_area", "last_dut_area"))
    with c_dut_mode:
        map_mode = st.radio("Calibration Mapping Mode", options=["Interpolation", "Power Law Fit"], index=0, help="Interpolation ensures perfect consistency for measured points. Power Law Fit is better for noisy data or extrapolation.")
    
    st.divider()
    dut_entries = []
    has_cal = 'active_calibration_meta' in st.session_state
    
    for i in range(num_dut):
        cd1, cd2, cd3 = st.columns([3, 1, 1])
        with cd1:
            row_file = st.selectbox(f"DUT File {i+1}", options=data_files, key=f"dut_file_{i}")
        with cd2:
            row_od = st.number_input(f"DUT OD {i+1}", min_value=0.0, max_value=10.0, step=0.1, value=0.0, key=f"dut_od_{i}")
        with cd3:
            if has_cal:
                seg_opts = list(st.session_state.active_calibration_meta.get('segment_fits', {}).keys())
                row_seg = st.selectbox(f"Cal Match {i+1}", options=seg_opts, key=f"dut_seg_{i}", help="Choose which calibration run this measurement matches.")
            else:
                row_seg = 1
        dut_entries.append({'file': row_file, 'od': row_od, 'cal_seg': row_seg})
    
    if st.button("Analyze LDR Data"):
        if not any(e['file'] for e in dut_entries):
            st.error("No files selected.")
        else:
            try:
                all_dut_dfs = []
                for entry_idx, entry in enumerate(dut_entries):
                    dut_file = entry['file']
                    dut_od = entry['od']
                    if not dut_file or not os.path.exists(dut_file): continue
                    
                    df_raw = pd.read_csv(dut_file)
                    
                    # --- DATA CLEANING ---
                    if 'LED_Current_A' not in df_raw.columns: continue
                         
                    groups = df_raw.groupby('LED_Current_A')
                    cleaned_rows = []
                    for amp, group in groups:
                        selected_row = None
                        if 'SNR_Status' in group.columns:
                            mask_ok = group['SNR_Status'].astype(str).str.contains('OK|High|Good', case=False, na=False)
                            ok_group = group[mask_ok]
                            selected_row = ok_group.iloc[-1] if not ok_group.empty else group.iloc[-1]
                        else:
                            selected_row = group.iloc[-1]
                        cleaned_rows.append(selected_row)
                        
                    df_segment = pd.DataFrame(cleaned_rows).reset_index(drop=True)
                    df_segment['DUT_Segment'] = f"Meas {entry_idx + 1} (OD {dut_od})"
                    df_segment['DUT_OD'] = dut_od
                    
                    # Calculate Current Density (J)
                    df_segment['Current_Density_A_cm2'] = df_segment['Photocurrent_A'] / dev_area
                    
                    # Apply Calibration
                    if 'active_calibration_meta' in st.session_state:
                        meta = st.session_state.active_calibration_meta
                        ref_area_val = meta.get('ref_area', 1.0)
                        cal_seg = entry.get('cal_seg', 1)
                        
                        if map_mode == "Interpolation" and 'segment_fits' in meta and cal_seg in meta['segment_fits']:
                            seg_info = meta['segment_fits'][cal_seg]
                            cal_od = seg_info['od']
                            c_cal = np.array(seg_info['currents'])
                            p_cal = np.array(seg_info['powers'])
                            A_extrap = seg_info['A_meas']
                            B_extrap = seg_info['B']
                            
                            # Hybrid Logic: Interpolate in range, Fit out of range
                            c_min, c_max = c_cal.min(), c_cal.max()
                            dut_currents = df_segment['LED_Current_A'].values
                            p_meas_at_cal = np.zeros_like(dut_currents)
                            
                            # Pre-create log-log interpolator for inner range
                            # Filter out non-positives to avoid log errors
                            v_mask = (c_cal > 0) & (p_cal > 0)
                            if v_mask.any():
                                f_log = interp1d(np.log10(c_cal[v_mask]), np.log10(p_cal[v_mask]), 
                                                kind='linear', fill_value='extrapolate')
                                
                                for idx, I in enumerate(dut_currents):
                                    if I <= 0:
                                        p_meas_at_cal[idx] = 0.0
                                    elif I >= c_min and I <= c_max:
                                        # Use log-log interpolation
                                        p_meas_at_cal[idx] = 10**float(f_log(np.log10(I)))
                                    else:
                                        # Use Power Law Extrapolation
                                        p_meas_at_cal[idx] = A_extrap * (I ** B_extrap)
                            else:
                                # Fallback to fit only
                                p_meas_at_cal = A_extrap * (dut_currents.clip(min=0) ** B_extrap)
                                
                            p_dut = p_meas_at_cal * (10**(cal_od - dut_od)) * (dev_area / ref_area_val)
                            
                            # Si Reference Density (at this LED current, through the filter)
                            # Current = Power * Responsivity
                            df_segment['Si_Photocurrent_Density_A_cm2'] = (p_meas_at_cal * meta.get('r_ref', 1.0)) / ref_area_val
                        
                        else:
                            # Use Global or Segment-specific Fit (Power Law Only)
                            A_fit, B_fit = meta.get('fit_A_glob', 0), meta.get('fit_B_glob', 1)
                            cal_od = 0.0
                            
                            if 'segment_fits' in meta and cal_seg in meta['segment_fits']:
                                A_fit = meta['segment_fits'][cal_seg]['A_meas']
                                B_fit = meta['segment_fits'][cal_seg]['B']
                                cal_od = meta['segment_fits'][cal_seg]['od']
                            
                            p_meas_at_cal = A_fit * (df_segment['LED_Current_A'].values.clip(min=0) ** B_fit)
                            p_dut = p_meas_at_cal * (10**(cal_od - dut_od)) * (dev_area / ref_area_val)
                            
                            # Si Reference Density
                            df_segment['Si_Photocurrent_Density_A_cm2'] = (p_meas_at_cal * meta.get('r_ref', 1.0)) / ref_area_val
                        
                        df_segment['Optical_Power_W'] = p_dut
                    
                    all_dut_dfs.append(df_segment)
                
                if not all_dut_dfs:
                    st.error("No valid data processed.")
                    st.stop()
                    
                df_dut = pd.concat(all_dut_dfs).sort_values(by='Optical_Power_W', ascending=False).reset_index(drop=True)
                st.info(f"Stitched {len(all_dut_dfs)} curves into a master dataset with {len(df_dut)} unique points.")
                if 'active_calibration_meta' in st.session_state:
                    # --- LINEARITY FIT (Log-Log) ---
                    valid_log = (df_dut['Optical_Power_W'] > 0) & (df_dut['Photocurrent_A'].abs() > 0)
                    df_log = df_dut[valid_log].copy()
                    
                    if len(df_log) >= 2:
                        lx = np.log10(df_log['Optical_Power_W'].values)
                        ly = np.log10(df_log['Photocurrent_A'].abs().values)
                        alpha, c_log = np.polyfit(lx, ly, 1) # Log-log slope
                        
                        # --- DYNAMIC RANGE (dB) ---
                        p_max, p_min = df_log['Optical_Power_W'].max(), df_log['Optical_Power_W'].min()
                        dr_db = 10 * np.log10(p_max / p_min)
                        
                        # --- RESULTS METRICS ---
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("Linearity Slope (α)", f"{alpha:.4f}", help="Ideal slope is 1.0 (I proportional to P).")
                        k2.metric("Dyn. Range", f"{dr_db:.1f} dB", help="10 * log10(Pmax / Pmin)")
                        
                        # Global Responsivity (Linear fit)
                        x_lin = df_log['Optical_Power_W'].values
                        y_lin = df_log['Photocurrent_A'].abs().values
                        slope_r = np.sum(x_lin * y_lin) / np.sum(x_lin**2)
                        k3.metric("Avg Responsivity", f"{slope_r:.4f} A/W")
                        
                        # --- NEP & DETECTIVITY ANALYSIS ---
                        # NEP = Noise_Current / Responsivity
                        # D* = sqrt(Area) / NEP
                        if 'Noise_Density_V_rtHz' in df_dut.columns and 'Resistance_Ohms' in df_dut.columns:
                             df_dut['Current_Noise_A_rtHz'] = df_dut['Noise_Density_V_rtHz'] / df_dut['Resistance_Ohms']
                             df_dut['NEP_W_rtHz'] = df_dut['Current_Noise_A_rtHz'] / slope_r
                             df_dut['Detectivity_Jones'] = np.sqrt(dev_area) / df_dut['NEP_W_rtHz']
                             
                             min_nep_hz = df_dut['NEP_W_rtHz'].min()
                             max_dstar = df_dut['Detectivity_Jones'].max()
                             
                             k4.metric("Best NEP (rtHz)", f"{min_nep_hz:.2e} W/√Hz")
                             # We need more columns if we want to show both, for now let's add D* to k4 or add k5
                        else:
                             # Fallback to Sensitivity from sweep (SNR=3)
                             min_nep = np.nan
                             if 'Sensitivity_W_SNR3' in df_log.columns:
                                  min_nep = df_log['Sensitivity_W_SNR3'].min()
                             k4.metric("Best NEP (SNR=3)", f"{min_nep:.2e} W")
                             max_dstar = np.nan

                        if 'Detectivity_Jones' in df_dut.columns:
                            st.metric("Peak Detectivity (D*)", f"{max_dstar:.2e} Jones", help="D* = sqrt(Area) / NEP")

                        # --- PLOTS ---
                        c_p1, c_p2 = st.columns(2)
                        with c_p1:
                            fig_ldr = px.scatter(df_dut, x="Optical_Power_W", y="Photocurrent_A", 
                                                 color="DUT_Segment",
                                                 log_x=True, log_y=True,
                                                 title="Stitched LDR: Photocurrent vs Power")
                            
                            # Add Common Fit Line
                            x_fit = np.geomspace(df_dut['Optical_Power_W'].min(), df_dut['Optical_Power_W'].max(), 100)
                            y_fit = (10**c_log) * (x_fit**alpha)
                            fig_ldr.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', 
                                                       name=f'Global Fit (α={alpha:.3f})',
                                                       line=dict(dash='dash', color='magenta')))
                            
                            st.plotly_chart(fig_ldr, use_container_width=True)
                            
                        with c_p2:
                             # Plot Linearity
                             df_dut['Normalized_Responsivity'] = (df_dut['Photocurrent_A'].abs() / df_dut['Optical_Power_W']) / slope_r
                             fig_lin = px.scatter(df_dut, x="Optical_Power_W", y="Normalized_Responsivity", 
                                                 log_x=True, color="DUT_Segment",
                                                 title="Linearity: R / <R>")
                             fig_lin.add_hline(y=1.0, line_dash="dash", line_color="tomato")
                             st.plotly_chart(fig_lin, use_container_width=True)
                         
                        c_p3, c_p4 = st.columns(2)
                        with c_p3:
                            if 'NEP_W_rtHz' in df_dut.columns:
                                fig_nep = px.scatter(df_dut, x="Optical_Power_W", y="NEP_W_rtHz",
                                                     log_x=True, log_y=True, color="DUT_Segment",
                                                     title="NEP vs Optical Power")
                                st.plotly_chart(fig_nep, use_container_width=True)
                        with c_p4:
                            if 'Detectivity_Jones' in df_dut.columns:
                                fig_dstar = px.scatter(df_dut, x="Optical_Power_W", y="Detectivity_Jones",
                                                      log_x=True, log_y=True, color="DUT_Segment",
                                                      title="Detectivity (D*) vs Optical Power")
                                st.plotly_chart(fig_dstar, use_container_width=True)
                              
                        st.info(f"**LDR Model Fit:** $I_{{ph}} = {10**c_log:.2e} \\cdot P^{{{alpha:.3f}}}$")
                        
                        with st.expander("Combined Data Table"):
                            st.dataframe(df_dut.style.format({
                                "LED_Current_A": "{:.2e}",
                                "Optical_Power_W": "{:.2e}",
                                "Photocurrent_A": "{:.2e}",
                                "Current_Density_A_cm2": "{:.2e}",
                                "Si_Photocurrent_Density_A_cm2": "{:.2e}",
                                "DUT_OD": "{:.1f}",
                                "NEP_W_rtHz": "{:.2e}",
                                "Detectivity_Jones": "{:.2e}",
                                "Noise_Density_V_rtHz": "{:.2e}",
                                "Resistance_Ohms": "{:.0f}"
                            }, na_rep="-"))
                    else:
                        st.warning("Not enough data points for linearity fit.")
                else:
                    st.warning("No calibration active. Plotting raw Photocurrent vs LED Current.")
                    fig = px.scatter(df_dut, x="LED_Current_A", y="Photocurrent_A", color="DUT_Segment", log_x=True, log_y=True)
                    st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Analysis Failed: {e}")

# --- TAB 3: TRACE ANALYSIS ---
with tab_trace:
    st.header("Raw Oscilloscope Trace Analysis")
    st.markdown("Load saved raw traces (`.csv`) to analyze Noise Spectral Density.")
    
    # Trace Selection (Using Global all_files)
    last_trace = settings.get("last_trace_file")
    
    trace_file = st.selectbox("Select Trace File", 
                              options=all_files, 
                              index=get_index(st.session_state.last_trace_file, all_files),
                              key='last_trace_file',
                              on_change=sync_setting, args=('last_trace_file', 'last_trace_file'))
    
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
