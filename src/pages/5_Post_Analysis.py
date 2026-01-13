
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import glob
import time
from scipy import signal
from scipy.interpolate import interp1d

# Ensure we can import from parent directory
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
    
    last_ref = settings.get("last_ref_file")
    
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
                
                temp_global = df_cal[['LED_Current_A', 'Source_Power_W']].rename(columns={'Source_Power_W': 'Optical_Power_W'})
                A_glob, B_glob, r2_glob = cal_mgr.fit_led_power_law(temp_global)
                
                st.session_state.active_calibration = df_cal
                st.session_state.active_calibration_meta = {
                    'wl': cal_wl, 'r_ref': r_val, 
                    'fit_A_glob': A_glob, 'fit_B_glob': B_glob, 'fit_r2_glob': r2_glob, 
                    'ref_area': ref_area,
                    'segment_fits': segment_fits
                }
                
                st.success(f"Calibration Generated! R_ref({cal_wl}nm) = {r_val:.4f} A/W")
                st.info(f"**Global Source Power Model (OD 0):** $P_{{source}} = {A_glob:.4e} \\cdot I_{{LED}}^{{{B_glob:.4f}}}$ ($R^2={r2_glob:.4f}$)")
                
                fig = px.scatter(df_cal, x="LED_Current_A", y="Measured_Power_W", 
                                 color="Source_Segment" if "Source_Segment" in df_cal.columns else None,
                                 log_x=True, log_y=True,
                                 title="Individual LED Calibration Curves (Measured at Diode)")
                
                for seg_id, fit in segment_fits.items():
                    seg_data = df_cal[df_cal['Source_Segment'] == seg_id]
                    x_f = np.geomspace(seg_data['LED_Current_A'].min(), seg_data['LED_Current_A'].max(), 50)
                    y_f = fit['A_meas'] * (x_f ** fit['B'])
                    fig.add_trace(go.Scatter(x=x_f, y=y_f, mode='lines', name=f'Seg {seg_id} (OD {fit["od"]}) Fit'))
                
                st.plotly_chart(fig, use_container_width=True)
                
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
        dev_area = st.number_input("Device Active Area (cm²)", format="%.4f", key="p_dut_area", value=st.session_state.p_dut_area, on_change=sync_setting, args=("p_dut_area", "last_ref_area"))
    with c_dut_mode:
        map_mode = st.radio("Calibration Mapping Mode", options=["Interpolation", "Power Law Fit"], index=0)
    
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
                row_seg = st.selectbox(f"Cal Match {i+1}", options=seg_opts, key=f"dut_seg_{i}")
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
                    df_segment['Current_Density_A_cm2'] = df_segment['Photocurrent_A'] / dev_area
                    
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
                            
                            c_min, c_max = c_cal.min(), c_cal.max()
                            dut_currents = df_segment['LED_Current_A'].values
                            p_meas_at_cal = np.zeros_like(dut_currents)
                            
                            v_mask = (c_cal > 0) & (p_cal > 0)
                            if v_mask.any():
                                f_log = interp1d(np.log10(c_cal[v_mask]), np.log10(p_cal[v_mask]), kind='linear', fill_value='extrapolate')
                                for idx, I in enumerate(dut_currents):
                                    if I <= 0: p_meas_at_cal[idx] = 0.0
                                    elif I >= c_min and I <= c_max: p_meas_at_cal[idx] = 10**float(f_log(np.log10(I)))
                                    else: p_meas_at_cal[idx] = A_extrap * (I ** B_extrap)
                            else:
                                p_meas_at_cal = A_extrap * (dut_currents.clip(min=0) ** B_extrap)
                                
                            p_dut = p_meas_at_cal * (10**(cal_od - dut_od)) * (dev_area / ref_area_val)
                            df_segment['Si_Photocurrent_Density_A_cm2'] = (p_meas_at_cal * meta.get('r_ref', 1.0)) / ref_area_val
                        else:
                            A_fit, B_fit = meta.get('fit_A_glob', 0), meta.get('fit_B_glob', 1)
                            cal_od = 0.0
                            if 'segment_fits' in meta and cal_seg in meta['segment_fits']:
                                A_fit = meta['segment_fits'][cal_seg]['A_meas']
                                B_fit = meta['segment_fits'][cal_seg]['B']
                                cal_od = meta['segment_fits'][cal_seg]['od']
                            p_meas_at_cal = A_fit * (df_segment['LED_Current_A'].values.clip(min=0) ** B_fit)
                            p_dut = p_meas_at_cal * (10**(cal_od - dut_od)) * (dev_area / ref_area_val)
                            df_segment['Si_Photocurrent_Density_A_cm2'] = (p_meas_at_cal * meta.get('r_ref', 1.0)) / ref_area_val
                        df_segment['Optical_Power_W'] = p_dut
                    all_dut_dfs.append(df_segment)
                
                if all_dut_dfs:
                    df_dut = pd.concat(all_dut_dfs).sort_values(by='Optical_Power_W', ascending=False).reset_index(drop=True)
                    valid_log = (df_dut['Optical_Power_W'] > 0) & (df_dut['Photocurrent_A'].abs() > 0)
                    df_log = df_dut[valid_log].copy()
                    if len(df_log) >= 2:
                        lx = np.log10(df_log['Optical_Power_W'].values)
                        ly = np.log10(df_log['Photocurrent_A'].abs().values)
                        alpha, c_log = np.polyfit(lx, ly, 1)
                        p_max, p_min = df_log['Optical_Power_W'].max(), df_log['Optical_Power_W'].min()
                        dr_db = 10 * np.log10(p_max / p_min)
                        x_lin, y_lin = df_log['Optical_Power_W'].values, df_log['Photocurrent_A'].abs().values
                        slope_r = np.sum(x_lin * y_lin) / np.sum(x_lin**2)
                        
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("Linearity Slope (α)", f"{alpha:.4f}")
                        k2.metric("Dyn. Range", f"{dr_db:.1f} dB")
                        k3.metric("Avg Responsivity", f"{slope_r:.4f} A/W")
                        
                        if 'Noise_Density_V_rtHz' in df_dut.columns and 'Resistance_Ohms' in df_dut.columns:
                             df_dut['Current_Noise_A_rtHz'] = df_dut['Noise_Density_V_rtHz'] / df_dut['Resistance_Ohms']
                             df_dut['NEP_W_rtHz'] = df_dut['Current_Noise_A_rtHz'] / slope_r
                             df_dut['Detectivity_Jones'] = np.sqrt(dev_area) / df_dut['NEP_W_rtHz']
                             k4.metric("Best NEP (rtHz)", f"{df_dut['NEP_W_rtHz'].min():.2e} W/√Hz")
                        
                        fig_ldr = px.scatter(df_dut, x="Optical_Power_W", y="Photocurrent_A", color="DUT_Segment", log_x=True, log_y=True)
                        st.plotly_chart(fig_ldr, use_container_width=True)
            except Exception as e:
                st.error(f"Analysis Failed: {e}")

# --- TAB 3: TRACE ANALYSIS ---
with tab_trace:
    st.header("Raw Oscilloscope Trace Analysis")
    last_trace = settings.get("last_trace_file")
    trace_file = st.selectbox("Select Trace File", options=all_files, index=get_index(st.session_state.last_trace_file, all_files), 
                              key='last_trace_file', on_change=sync_setting, args=('last_trace_file', 'last_trace_file'))
    
    if st.button("Analyze Trace"):
        try:
            df_trace = pd.read_csv(trace_file)
            fig_time = px.line(df_trace, x='time', y='voltage', title=f"Time Domain: {os.path.basename(trace_file)}")
            st.plotly_chart(fig_time, use_container_width=True)
        except Exception as e:
            st.error(f"Analysis Failed: {e}")
