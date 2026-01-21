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
import analysis.signal_processing as signal_processing
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
    
    c_set3, c_set4 = st.columns(2)
    with c_set3:
        cal_duty = st.number_input("Calibration Duty Cycle (%)", min_value=0.1, max_value=100.0, value=50.0, help="The duty cycle used during the LED calibration sweep.")
    
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
                    'cal_duty': cal_duty / 100.0, # Store as decimal
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
        t_int_ldr = st.number_input("Integration Time (s)", min_value=0.01, max_value=100.0, value=1.0, help="Duration of the measurement trace. Used to calculate Noise-Limited LDR.")
    
    c_ldr1, c_ldr2 = st.columns(2)
    with c_ldr1:
        ldr_duty = st.number_input("Measurement Duty Cycle (%)", min_value=0.1, max_value=100.0, value=50.0, help="The duty cycle used during the DUT LDR sweep.")
    with c_ldr2:
        x_axis_mode = st.selectbox("X-Axis Units", 
                                  options=["Average Power (W)", "Peak Power (W)", "Average Irradiance (W/cm²)", "Peak Irradiance (W/cm²)"],
                                  index=0)
    
    st.divider()
    dut_entries = []
    has_cal = 'active_calibration_meta' in st.session_state
    
    # Move Mapping Mode to a dedicated row to free up space for t_int
    map_mode = st.radio("Calibration Mapping Mode", options=["Interpolation", "Power Law Fit"], index=0, horizontal=True)
    
    for i in range(num_dut):
        cd1, cd2, cd3, cd4 = st.columns([3, 1, 1, 1])
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
        with cd4:
            row_scale = st.number_input(f"P-Scale {i+1}", min_value=0.01, max_value=10.0, step=0.01, value=1.0, key=f"dut_scale_{i}", help="Optical power correction multiplier to align segments.")
            
        dut_entries.append({'file': row_file, 'od': row_od, 'cal_seg': row_seg, 'scale': row_scale})
    
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
                            
                            # DUT Power Correction Factors (Fourier)
                            # P_rms_fund = P_peak * sqrt(2) * sin(pi*D) / pi
                            # -> P_peak = P_rms_fund * pi / (sqrt(2) * sin(pi*D))
                            
                            d_cal = meta.get('cal_duty', 0.5)
                            d_dut = ldr_duty / 100.0
                            
                            # Scale factor from "RMS Fund (at D_cal)" back to Peak
                            f_peak_cal = np.pi / (np.sqrt(2) * np.sin(np.pi * d_cal))
                            
                            # Scale factor from Peak to target unit
                            f_target = 1.0
                            if "Peak" in x_axis_mode:
                                f_target = 1.0
                            else: # Average
                                f_target = d_dut
                                
                            if "Irradiance" in x_axis_mode:
                                f_target /= ref_area_val # Scale to W/cm2
                                
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
                                
                            p_peak_led = p_meas_at_cal * f_peak_cal
                            p_dut = p_peak_led * f_target * (10**(cal_od - dut_od)) * (dev_area / ref_area_val if "Power" in x_axis_mode else 1.0)
                            
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
                            
                            p_peak_led = p_meas_at_cal * (np.pi / (np.sqrt(2) * np.sin(np.pi * meta.get('cal_duty', 0.5))))
                            
                            f_target = 1.0
                            if "Peak" in x_axis_mode: f_target = 1.0
                            else: f_target = (ldr_duty / 100.0)
                            
                            if "Irradiance" in x_axis_mode:
                                f_target /= ref_area_val
                                
                            p_dut = p_peak_led * f_target * (10**(cal_od - dut_od)) * (dev_area / ref_area_val if "Power" in x_axis_mode else 1.0)
                            
                            # Si Reference Density
                            df_segment['Si_Photocurrent_Density_A_cm2'] = (p_meas_at_cal * meta.get('r_ref', 1.0)) / ref_area_val
                        
                        df_segment['Optical_Power_W'] = p_dut * entry.get('scale', 1.0)
                        df_segment['X_Axis_Label'] = x_axis_mode
                    
                    all_dut_dfs.append(df_segment)
                
                df_dut = pd.concat(all_dut_dfs).sort_values(by='Optical_Power_W', ascending=False).reset_index(drop=True)
                st.session_state.df_ldr = df_dut # Persist for slider interactions
                st.success(f"Stitched {len(all_dut_dfs)} curves into a master dataset with {len(df_dut)} unique points.")
            except Exception as e:
                st.error(f"Error during analysis: {e}")

    # --- RESULTS & PLOTTING (Outside button, relies on session state) ---
    if 'df_ldr' in st.session_state:
        df_dut = st.session_state.df_ldr
        if 'active_calibration_meta' in st.session_state:
            # --- LINEARITY FIT (Log-Log) ---
            valid_log = (df_dut['Optical_Power_W'] > 0) & (df_dut['Photocurrent_A'].abs() > 0)
            df_log_full = df_dut[valid_log].copy()
            
            if len(df_log_full) >= 2:
                # Sort by Power descending for selection
                df_log_full = df_log_full.sort_values("Optical_Power_W", ascending=False)
                
                # Ensure fitness_n persistent value
                max_pts = len(df_log_full)
                if 'ldr_fit_n' not in st.session_state or st.session_state.ldr_fit_n > max_pts:
                    st.session_state.ldr_fit_n = max_pts
                    
                fit_n = st.slider("Include Top N points in Linearity Fit", 
                                 min_value=2, 
                                 max_value=max_pts, 
                                 value=st.session_state.ldr_fit_n,
                                 key="ldr_fit_n_slider",
                                 help="Select how many of the highest power points to use for the alpha (slope) fit.")
                
                # Sync state
                st.session_state.ldr_fit_n = fit_n
                df_log = df_log_full.head(fit_n)
                
                lx = np.log10(df_log['Optical_Power_W'].values)
                ly = np.log10(df_log['Photocurrent_A'].abs().values)
                alpha, c_log = np.polyfit(lx, ly, 1) # Log-log slope
                
                # --- DYNAMIC RANGE (dB) ---
                p_max, p_min = df_log['Optical_Power_W'].max(), df_log['Optical_Power_W'].min()
                dr_db = 10 * np.log10(p_max / p_min)
                
                # --- RESULTS METRICS ---
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Linearity Slope (α)", f"{alpha:.4f}", help="Ideal slope is 1.0.")
                k2.metric("Dyn. Range", f"{dr_db:.1f} dB")
                
                # Global Responsivity (Linear fit)
                x_lin = df_log['Optical_Power_W'].values
                y_lin = df_log['Photocurrent_A'].abs().values
                slope_r = np.sum(x_lin * y_lin) / np.sum(x_lin**2)
                k3.metric("Avg Responsivity", f"{slope_r:.4f} A/W")
                
                # --- NEP & DETECTIVITY ANALYSIS ---
                if 'Noise_Density_V_rtHz' in df_dut.columns and 'Resistance_Ohms' in df_dut.columns:
                     df_dut['Current_Noise_A_rtHz'] = df_dut['Noise_Density_V_rtHz'] / df_dut['Resistance_Ohms']
                     df_dut['NEP_W_rtHz'] = df_dut['Current_Noise_A_rtHz'] / slope_r
                     df_dut['Detectivity_Jones'] = np.sqrt(dev_area) / df_dut['NEP_W_rtHz']
                     
                     min_nep_hz = df_dut['NEP_W_rtHz'].min()
                     max_dstar = df_dut['Detectivity_Jones'].max()
                     
                     # Calculate Noise-Limited LDR (SNR=1)
                     enbw = 2.0 / t_int_ldr
                     p_snr1 = min_nep_hz * np.sqrt(enbw)
                     dr_noise_db = 10 * np.log10(p_max / p_snr1)
                     
                     k4.metric("Best NEP (rtHz)", f"{min_nep_hz:.2e} W/√Hz")
                     st.metric("Noise-Limited LDR (SNR=1)", f"{dr_noise_db:.1f} dB", help=f"10*log10(Pmax / P_snr1). P_snr1 = {p_snr1:.2e} W at BW = {enbw:.2f} Hz ({t_int_ldr}s integration).")
                else:
                     min_nep = df_log['Sensitivity_W_SNR3'].min() if 'Sensitivity_W_SNR3' in df_log.columns else np.nan
                     k4.metric("Best NEP (SNR=3)", f"{min_nep:.2e} W")
                     max_dstar = np.nan

                if 'Detectivity_Jones' in df_dut.columns:
                    st.metric("Peak Detectivity (D*)", f"{max_dstar:.2e} Jones")

                x_label = df_dut['X_Axis_Label'].iloc[0] if 'X_Axis_Label' in df_dut.columns else "Optical Power (W)"
                
                # --- PLOTS ---
                c_p1, c_p2 = st.columns(2)
                with c_p1:
                    fig_ldr = px.scatter(df_dut, x="Optical_Power_W", y="Photocurrent_A", 
                                         color="DUT_Segment", log_x=True, log_y=True,
                                         labels={"Optical_Power_W": x_label, "Photocurrent_A": "Photocurrent (A)"},
                                         title=f"Stitched LDR: Photocurrent vs {x_label.split(' (')[0]}")
                    x_fit = np.geomspace(df_dut['Optical_Power_W'].min(), df_dut['Optical_Power_W'].max(), 100)
                    y_fit = (10**c_log) * (x_fit**alpha)
                    fig_ldr.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', 
                                               name=f'Global Fit (α={alpha:.3f})',
                                               line=dict(dash='dash', color='magenta')))
                    st.plotly_chart(fig_ldr, use_container_width=True)
                    
                with c_p2:
                    df_dut['Normalized_Responsivity'] = (df_dut['Photocurrent_A'].abs() / df_dut['Optical_Power_W']) / slope_r
                    fig_lin = px.scatter(df_dut, x="Optical_Power_W", y="Normalized_Responsivity", 
                                        log_x=True, color="DUT_Segment", 
                                        labels={"Optical_Power_W": x_label, "Normalized_Responsivity": "R / <R>"},
                                        title=f"Linearity (R/<R>) vs {x_label.split(' (')[0]}")
                    fig_lin.add_hline(y=1.0, line_dash="dash", line_color="tomato")
                    st.plotly_chart(fig_lin, use_container_width=True)
                 
                c_p3, c_p4 = st.columns(2)
                with c_p3:
                    if 'NEP_W_rtHz' in df_dut.columns:
                        fig_nep = px.scatter(df_dut, x="Optical_Power_W", y="NEP_W_rtHz",
                                             log_x=True, log_y=True, color="DUT_Segment", 
                                             labels={"Optical_Power_W": x_label, "NEP_W_rtHz": "NEP (W/√Hz)"},
                                             title=f"Spectral NEP vs {x_label.split(' (')[0]}")
                        st.plotly_chart(fig_nep, use_container_width=True)
                with c_p4:
                    if 'Detectivity_Jones' in df_dut.columns:
                        fig_dstar = px.scatter(df_dut, x="Optical_Power_W", y="Detectivity_Jones",
                                              log_x=True, log_y=True, color="DUT_Segment", 
                                              labels={"Optical_Power_W": x_label, "Detectivity_Jones": "D* (Jones)"},
                                              title=f"Detectivity (D*) vs {x_label.split(' (')[0]}")
                        st.plotly_chart(fig_dstar, use_container_width=True)
                
                # Data Download
                c_dl1, c_dl2 = st.columns(2)
                
                with c_dl1:
                    # Specialized Export for Origin (Raw XY + Fit XY)
                    x_meas = df_dut['Optical_Power_W'].values
                    y_meas = df_dut['Photocurrent_A'].abs().values
                    x_fit_pts = np.geomspace(df_dut['Optical_Power_W'].min(), df_dut['Optical_Power_W'].max(), 100)
                    y_fit_pts = (10**c_log) * (x_fit_pts**alpha)
                    
                    max_len = max(len(x_meas), len(x_fit_pts))
                    def pad(arr, length):
                        return np.pad(arr.astype(float), (0, length - len(arr)), constant_values=np.nan)

                    # Dynamic Column Names for Origin
                    x_col_name = "Power" if "Power" in x_axis_mode else "Irradiance"
                    x_unit_name = "W" if "Power" in x_axis_mode else "Wcm2"
                    
                    # Create a clean XY XY dataframe for easy copy-paste into Origin
                    df_origin = pd.DataFrame({
                        f"{x_col_name}_Meas_{x_unit_name}": pad(x_meas, max_len),
                        "Photocurrent_Meas_A": pad(y_meas, max_len),
                        f"{x_col_name}_Fit_Line_{x_unit_name}": pad(x_fit_pts, max_len),
                        "Photocurrent_Fit_Line_A": pad(y_fit_pts, max_len)
                    })
                    
                    csv_origin = df_origin.to_csv(index=False)
                    st.download_button("📊 Export for Origin (XY, XY Fit)", 
                                     data=csv_origin, 
                                     file_name=f"ldr_origin_plot_alpha_{alpha:.3f}.csv", 
                                     mime="text/csv",
                                     use_container_width=True,
                                     type="primary",
                                     help="4-column CSV: [Measured Power, Measured Current, Fit Power, Fit Current]. Perfect for Origin.")

                with c_dl2:
                    csv_full = df_dut.to_csv(index=False)
                    st.download_button("📥 Download Full Stitched Results", 
                                     data=csv_full, 
                                     file_name="ldr_full_stitched_results.csv", 
                                     mime="text/csv",
                                     use_container_width=True,
                                     help="Complete dataset with all intermediate columns (SNR, Resistance, etc.)")

                st.info(f"**LDR Model Fit:** $I_{{ph}} = {10**c_log:.2e} \\cdot P^{{{alpha:.3f}}}$")
                
                with st.expander("Combined Data Table"):
                    st.dataframe(df_dut.style.format({
                        "LED_Current_A": "{:.2e}", "Optical_Power_W": "{:.2e}", "Photocurrent_A": "{:.2e}",
                        "Current_Density_A_cm2": "{:.2e}", "Si_Photocurrent_Density_A_cm2": "{:.2e}",
                        "DUT_OD": "{:.1f}", "NEP_W_rtHz": "{:.2e}", "Detectivity_Jones": "{:.2e}",
                        "Noise_Density_V_rtHz": "{:.2e}", "Resistance_Ohms": "{:.0f}"
                    }, na_rep="-"))
            else:
                st.warning("Not enough data points for linearity fit.")
        else:
            st.warning("No calibration active. Plotting raw Photocurrent vs LED Current.")
            fig = px.scatter(df_dut, x="LED_Current_A", y="Photocurrent_A", color="DUT_Segment", log_x=True, log_y=True)
            st.plotly_chart(fig, use_container_width=True)
    
    # --- Advanced: Batch Re-analysis ---
    st.divider()
    with st.expander("🛠️ Advanced: Re-calculate SNR/Amplitude from Raw Traces"):
        st.markdown("""
        If your measurement results have 'Wrong SNR' due to 50Hz pickup, use this tool to re-reprocess the **raw trace files** 
        using a specific **Target Frequency** and **Digital Lock-in**.
        """)
        
        c_re1, c_re2 = st.columns(2)
        with c_re1:
            trace_folder = st.text_input("Folder containing Trace CSVs", value=settings.get("base_save_folder", "data"))
            re_target_f = st.number_input("Target Analysis Frequency (Hz)", value=80.0, key="re_target_f")
            re_resistor = st.number_input("Gain (Resistor) (Ω)", value=47000.0, key="re_resistor")
            save_re = st.checkbox("Save Re-calculated Results to CSV", value=True)
            mask_re = st.checkbox("Mask Power-Line peaks (50/60Hz)", value=True, help="Ignore 50Hz, 60Hz and harmonics from noise floor estimation.")
            
        mask_list_re = [50.0, 60.0] if mask_re else None
            
        if st.button("🚀 Run Batch Re-analysis"):
            if not os.path.exists(trace_folder):
                st.error("Folder not found.")
            else:
                trace_files = glob.glob(os.path.join(trace_folder, "trace_step_*.csv"))
                if not trace_files:
                    st.error("No 'trace_step_*.csv' files found in this folder.")
                else:
                    st.info(f"Checking for results file in `{trace_folder}`...")
                    # Try to load resistor mapping from results file
                    all_results_csvs = glob.glob(os.path.join(trace_folder, "*results*.csv"))
                    results_csvs = [f for f in all_results_csvs if "recalculated" not in os.path.basename(f).lower()]
                    
                    results_df = None
                    if results_csvs:
                        try:
                            # Use the most likely results file (the one that isn't re-calculated)
                            results_csvs.sort(key=os.path.getmtime, reverse=True)
                            target_csv = results_csvs[0]
                            results_df = pd.read_csv(target_csv)
                            
                            if 'LED_Current_A' in results_df.columns and 'Resistance_Ohms' in results_df.columns:
                                st.info(f"📂 Found results file: `{os.path.basename(target_csv)}`. Using **closest-match** strategy to identify gains.")
                            else:
                                results_df = None
                                st.warning("Results file found but lacks 'LED_Current_A' or 'Resistance_Ohms' columns.")
                        except Exception as e:
                            st.warning(f"Could not load resistor mapping from results file: {e}")

                    st.info(f"Processing {len(trace_files)} traces...")
                    re_results = []
                    progress_re = st.progress(0)
                    
                    for f_idx, f_path in enumerate(sorted(trace_files)):
                        try:
                            df_t = pd.read_csv(f_path)
                            t_arr = df_t['time'].values
                            v_arr = df_t['voltage'].values
                            
                            fs_re = 1.0 / np.mean(np.diff(t_arr))
                            
                            # Lock-in Amplitude
                            l_rms = signal_processing.calculate_lockin_amplitude(v_arr, fs_re, re_target_f)
                            # Noise Density
                            n_dens_v = signal_processing.calculate_noise_density_sideband(v_arr, fs_re, re_target_f, mask_freqs=mask_list_re)
                            # Robust SNR
                            snr_rob = signal_processing.calculate_robust_snr(v_arr, fs_re, re_target_f, mask_freqs=mask_list_re)
                            # FFT SNR
                            snr_fft = signal_processing.calculate_snr_fft(v_arr, fs_re, re_target_f)
                            
                            # Extract Current from filename
                            import re
                            match = re.search(r"I([\d\.e\-\+]+)A", os.path.basename(f_path))
                            i_led_target = float(match.group(1)) if match else 0.0
                            
                            # Gain Lookup: Find the LAST row in results_df where LED_Current_A matches i_led_target
                            current_resistor = re_resistor
                            if results_df is not None:
                                # Define a tolerance (1% relative or 1pA absolute)
                                tolerance = max(abs(i_led_target) * 0.01, 1e-12)
                                # Find all rows within tolerance
                                matches = results_df[np.abs(results_df['LED_Current_A'] - i_led_target) < tolerance]
                                if not matches.empty:
                                    # Pick the LAST match (corresponds to successful final measurement)
                                    current_resistor = float(matches['Resistance_Ohms'].iloc[-1])
                            
                            re_results.append({
                                "File": os.path.basename(f_path),
                                "LED_Current_A": i_led_target,
                                "LockIn_Amp_V": l_rms,
                                "Resistance_Ohms": current_resistor,
                                "Photocurrent_A": l_rms / current_resistor,
                                "Noise_Density_V_rtHz": n_dens_v,
                                "SNR_Broadband": snr_rob,
                                "SNR_FFT": snr_fft,
                                "SNR_Status": "OK"
                            })
                            progress_re.progress((f_idx + 1) / len(trace_files))
                        except Exception as e:
                            st.warning(f"Failed to process {f_path}: {e}")
                    
                    if re_results:
                        df_re = pd.DataFrame(re_results).sort_values("LED_Current_A", ascending=False)
                        st.success("Re-analysis Complete!")
                        st.dataframe(df_re.style.format({
                            "LED_Current_A": "{:.3e}",
                            "LockIn_Amp_V": "{:.4e}",
                            "Resistance_Ohms": "{:.1e}",
                            "Photocurrent_A": "{:.4e}",
                            "Noise_Density_V_rtHz": "{:.4e}",
                            "SNR_Broadband": "{:.2f}",
                            "SNR_FFT": "{:.2f}"
                        }))
                        
                        if save_re:
                            out_path = os.path.join(trace_folder, "recalculated_results.csv")
                            df_re.to_csv(out_path, index=False)
                            st.info(f"Saved to: {out_path}")

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
    
    c_t3, c_t4 = st.columns(2)
    with c_t3:
        target_freq = st.number_input("Target Signal Frequency (Hz)", value=settings.get("sweep_freq", 80.0), help="Digital lock-in will lock to this frequency.")
    with c_t4:
        mask_trace = st.checkbox("Mask Power-Line peaks (50/60Hz)", value=True, key="mask_trace", help="Ignore peaks at 50Hz, 60Hz and harmonics from noise floor estimation.")
        
    mask_list_trace = [50.0, 60.0] if mask_trace else None
        
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

                # 5. Digital Lock-in Analysis
                lock_in_rms = signal_processing.calculate_lockin_amplitude(v, fs, target_freq)
                lock_in_current = lock_in_rms / load_res
                
                # Robust SNR (Lock-in / Sideband Noise)
                robust_snr = signal_processing.calculate_robust_snr(v, fs, target_freq, mask_freqs=mask_list_trace)
                
                # Standard FFT SNR
                fft_snr = signal_processing.calculate_snr_fft(v, fs, target_freq)

                # Display Results
                st.subheader("🎯 Digital Lock-in Results")
                k1, k2, k3 = st.columns(3)
                k1.metric("Lock-in Ampltitude (RMS)", f"{lock_in_rms*1000:.3f} mV", f"{lock_in_current:.2e} A")
                k2.metric("Robust SNR", f"{robust_snr:.1f}", help="Signal (Lock-in) / Noise Floor (Sideband)")
                k3.metric("FFT SNR (Peak/Floor)", f"{fft_snr:.1f}")

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
                
                # Add target frequency marker
                fig_psd.add_vline(x=target_freq, line_dash="dash", line_color="green", 
                                  annotation_text=f"Target: {target_freq}Hz")
                # Add 50Hz marker
                fig_psd.add_vline(x=50.0, line_dash="dot", line_color="orange", 
                                  annotation_text="50Hz Noise")
                
                st.plotly_chart(fig_psd, use_container_width=True)
                
            except Exception as e:
                st.error(f"Analysis Failed: {e}")
