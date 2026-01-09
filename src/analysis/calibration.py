import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d
import logging

class CalibrationManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_responsivity_curve(self, filepath: str) -> interp1d:
        """
        Loads a 2-column CSV (Wavelength, Responsivity) and returns an interpolator.
        Assumes no header, or checks if header exists.
        """
        try:
            # Try reading with no header first
            df = pd.read_csv(filepath, header=None)
            
            # Simple heuristic: if first row is strings, it is a header
            if isinstance(df.iloc[0,0], str):
                df = pd.read_csv(filepath) # Reload with header
                cols = df.columns
                x_col, y_col = cols[0], cols[1]
            else:
                x_col, y_col = 0, 1

            # Sort by wavelength
            df = df.sort_values(by=x_col)
            
            x = df[x_col].values
            y = df[y_col].values
            
            # Create interpolator
            interp = interp1d(x, y, kind='linear', fill_value="extrapolate")
            return interp
            
        except Exception as e:
            self.logger.error(f"Failed to load responsivity file: {e}")
            raise

    def generate_led_calibration(self, 
                                measurement_file: str, 
                                reference_responsivity_file: str, 
                                led_wavelength_nm: float) -> tuple[pd.DataFrame, float]:
        """
        Generates an LED Power Calibration table (LED_Current -> Optical_Power).
        """
        return self.generate_multi_led_calibration(
            [{'file': measurement_file, 'od': 0.0}], 
            reference_responsivity_file, 
            led_wavelength_nm
        )

    def generate_multi_led_calibration(self, 
                                     measurements: list[dict], # list of {'file': str, 'od': float}
                                     reference_responsivity_file: str, 
                                     led_wavelength_nm: float) -> tuple[pd.DataFrame, float, dict]:
        """
        Generates a combined LED Power Calibration table from multiple measurements with OD filters.
        Normalizes all to "Source Power" (Power at OD 0) for the combined dataset,
        but also keeps raw measurements for segment-specific fitting.
        
        P_meas = I_pd / R_ref
        P_source = P_meas * 10^OD
        """
        # 1. Get Reference Responsivity at WL
        r_interp = self.load_responsivity_curve(reference_responsivity_file)
        r_ref = float(r_interp(led_wavelength_nm))
        
        all_dfs = []
        segment_fits = {}
        
        for i, m in enumerate(measurements):
            file = m.get('file')
            od = m.get('od', 0.0)
            if not file or not os.path.exists(file):
                self.logger.warning(f"File {file} not found or invalid. Skipping.")
                continue
                
            df = pd.read_csv(file)
            req_cols = ['LED_Current_A', 'Photocurrent_A']
            if not all(c in df.columns for c in req_cols):
                self.logger.warning(f"File {file} missing columns {req_cols}. Skipping.")
                continue
            
            # P_measured = power at the detector through the filter
            df['Measured_Power_W'] = df['Photocurrent_A'].abs() / r_ref
            
            # P_source = power the LED would output at OD 0
            df['Source_Power_W'] = df['Measured_Power_W'] * (10**od)
            
            df['Source_Segment'] = i + 1
            df['OD_Filter'] = od
            
            # Fit this segment individually based on RAW MEASURED POWER
            try:
                # We need a temp df for fit_led_power_law to find 'Optical_Power_W'
                temp_df = df[['LED_Current_A', 'Measured_Power_W']].rename(columns={'Measured_Power_W': 'Optical_Power_W'})
                A, B, r2 = self.fit_led_power_law(temp_df)
                
                # Store fit AND raw points for interpolation
                segment_fits[i + 1] = {
                    'A_meas': A, 
                    'B': B, 
                    'r2': r2, 
                    'od': od,
                    'currents': df['LED_Current_A'].values.tolist(),
                    'powers': df['Measured_Power_W'].values.tolist()
                }
            except Exception as e:
                self.logger.error(f"Failed to fit segment {i+1}: {e}")
                
            all_dfs.append(df[['LED_Current_A', 'Photocurrent_A', 'Measured_Power_W', 'Source_Power_W', 'Source_Segment', 'OD_Filter']])
            
        if not all_dfs:
            raise ValueError("No valid calibration files provided.")
            
        result = pd.concat(all_dfs).sort_values('LED_Current_A').reset_index(drop=True)
        return result, r_ref, segment_fits

    def fit_led_power_law(self, df_cal: pd.DataFrame) -> tuple[float, float, float]:
        """
        Fits a Power Law model P = A * I^B to the calibration data.
        Returns: (A, B, R_squared)
        """
        # Filter positive values for log
        valid = (df_cal['LED_Current_A'] > 0) & (df_cal['Optical_Power_W'] > 0)
        df_log = df_cal[valid].copy()
        
        if len(df_log) < 2:
            return 0.0, 1.0, 0.0
            
        x = np.log(df_log['LED_Current_A'].values)
        y = np.log(df_log['Optical_Power_W'].values)
        
        # Fit log(P) = log(A) + B * log(I)
        # y = c + m*x
        m, c = np.polyfit(x, y, 1)
        
        B = m
        A = np.exp(c)
        
        # R2
        yhat = m*x + c
        ybar = np.mean(y)
        ssreg = np.sum((yhat-ybar)**2)
        sstot = np.sum((y-ybar)**2)
        r2 = ssreg/sstot if sstot!=0 else 0
        
        return A, B, r2

    def fit_responsivity_slope(self, df_dut: pd.DataFrame) -> tuple[float, float, float]:
        """
        Fits a linear slope to I_dut vs P_opt to find Responsivity.
        Returns (Slope, Intercept, R_squared)
        """
        if 'Optical_Power_W' not in df_dut.columns or 'Photocurrent_A' not in df_dut.columns:
            return 0.0, 0.0, 0.0
            
        x = df_dut['Optical_Power_W'].values
        y = df_dut['Photocurrent_A'].abs().values # absolute current
        
        # Remove low/zero power points to avoid noise? Or fit all?
        # Fit linear with Intercept forced to 0 (Physical: I = R * P)
        # Model: y = m * x
        # Least squares solution: m = sum(x*y) / sum(x^2)
        
        # Optional: Use weighted least squares for LDR? 
        # For now, simple zero-forced.
        slope = np.sum(x*y) / np.sum(x**2)
        intercept = 0.0
        
        # R2
        yhat = slope * x 
        ybar = np.sum(y)/len(y)
        ssreg = np.sum((yhat-ybar)**2)
        sstot = np.sum((y - ybar)**2)
        r2 = ssreg / sstot if sstot != 0 else 0
        
        return slope, intercept, r2
