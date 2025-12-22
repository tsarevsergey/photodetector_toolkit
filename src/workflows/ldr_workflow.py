
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Callable
import logging

from hardware.smu_controller import SMUController
from hardware.scope_controller import ScopeController

class ResistorChangeRequiredException(Exception):
    def __init__(self, step_index, snr, current_level, last_results, last_waveforms):
        self.step_index = step_index
        self.snr = snr
        self.current_level = current_level
        self.last_results = last_results
        self.last_waveforms = last_waveforms

class LDRWorkflow:
    def __init__(self, smu: SMUController, scope: ScopeController):
        self.smu = smu
        self.scope = scope
        self.logger = logging.getLogger("workflow.LDR")
        self.stop_requested = False
        
    def run_sweep(self, 
                  start_current: float, 
                  stop_current: float, 
                  steps: int, 
                  frequency: float, 
                  duty_cycle: float,
                  resistor_ohms: float,
                  averages: int = 3,
                  compliance_limit: float = 8.0,
                  scope_range: str = "10V",
                  auto_range: bool = False,
                  ac_coupling: bool = False,
                  min_snr_threshold: float = 25.0,
                  progress_callback: Callable[[float, str], None] = None,
                  start_step_index: int = 0,
                  previous_results: list = None,
                  previous_waveforms: list = None) -> (pd.DataFrame, list):
        
        self.stop_requested = False
        results = previous_results if previous_results else []
        waveforms = previous_waveforms if previous_waveforms else []
        
        # Generate Current Levels (Logspace)
        s = start_current if start_current > 0 else 1e-9
        current_levels_full = np.logspace(np.log10(s), np.log10(stop_current), steps)
        
        # Slice for current run
        # We need to maintain the original index 'i' for progress reporting
        levels_to_run = []
        for idx, lvl in enumerate(current_levels_full):
            if idx >= start_step_index:
                levels_to_run.append((idx, lvl))
        
        # Scope settings
        period = 1.0 / frequency
        # Capture enough for ~4 periods min to ensure good Vpp measure
        capture_duration = max(4 * period, 0.02) 
        
        num_samples = 2000
        tb_index = self.scope.calculate_timebase_index(capture_duration, num_samples)
        
        # 1. Start with Output OFF to ensure clean state
        # Force disable to reset any previous states
        try:
             self.smu.disable_output()
        except: 
             pass
             
        time.sleep(0.5) # Allow relays to open
        
        # Available ranges ordered Low to High
        range_list = ['10MV', '20MV', '50MV', '100MV', '200MV', '500MV', '1V', '2V', '5V', '10V']
        
        # Limits mapping (approx)
        range_limits = {
            '10MV': 0.01, '20MV': 0.02, '50MV': 0.05, '100MV': 0.1, 
            '200MV': 0.2, '500MV': 0.5, '1V': 1.0, '2V': 2.0, '5V': 5.0, '10V': 10.0
        }
        
        # Determine initial index from user setting
        try:
            curr_range_idx = range_list.index(scope_range.upper())
        except:
            curr_range_idx = 9 # Default 10V
            
        self.logger.info(f"Starting LDR Sweep: {steps} steps (Running {len(levels_to_run)}), {start_current:.2e}A -> {stop_current:.2e}A, Freq={frequency}Hz, Start Range={range_list[curr_range_idx]}")
        
        # Loop
        for i, current_level in levels_to_run:
            if self.stop_requested: break
            
            if progress_callback:
                progress_callback((i / steps), f"Step {i+1}/{steps}: {current_level*1000:.2f} mA")
            
            try:
                # --- STEP 1: TEARDOWN / RESET ---
                # "Configure & Arm" button equivalent
                try: self.smu.resource.write("ABOR") 
                except: pass
                self.smu.disable_output()
                # Fast reset
                time.sleep(0.1)
                
                # --- Auto-Range Logic Wrapper ---
                # We wrap the measurement specific logic in a loop to allow re-trying with different scope ranges
                
                retry_count = 0
                max_retries = 15 # Allow full traversal of range stack
                
                # Initialize loop variables to avoid UnboundLocalError
                vpps = []
                step_waveforms = None
                
                while retry_count < max_retries:
                    retry_count += 1
                    current_range_str = range_list[curr_range_idx]
                    
                    # --- STEP 2: CONFIGURE ---
                    self.smu.set_compliance(compliance_limit, "VOLT")
                    
                    # Pulse Logic
                    total_capture_time = (capture_duration + 0.2) * averages
                    req_cycles = int(total_capture_time / period) + 10
                    if req_cycles < 10: req_cycles = 10
                    
                    self.smu.generate_square_wave(current_level, 0.0, period, duty_cycle, req_cycles, "CURR")
                    
                    # Configure Scope
                    coupling_str = "AC" if ac_coupling else "DC"
                    self.scope.configure_channel('A', True, current_range_str, coupling_str)
                    
                    # --- STEP 3: ENABLE & RUN ---
                    self.smu.enable_output()
                    time.sleep(0.5) 
                    self.smu.trigger_list()
                    
                    # --- STEP 4: CAPTURE TRIAL ---
                    # We capture the first block to check ranging
                    time.sleep(0.1) 
                    times, volts = self.scope.capture_block(tb_index, num_samples)
                    
                    # Check Signal Quality
                    if len(volts) > 0:
                        # Determine "Amplitude" relative to Range Limit
                        # Use Robust Statistics (Percentiles) to ignore outliers/spikes
                        
                        v_limit = range_limits.get(current_range_str, 10.0)
                        
                        if ac_coupling:
                            # AC Coupling: Amplitude = Half Peak-to-Peak
                            # Use 99.9th and 0.1st percentiles to reject single-sample transient spikes
                            v_top = np.percentile(volts, 99.9)
                            v_bot = np.percentile(volts, 0.1)
                            v_max_abs = (v_top - v_bot) / 2.0
                        else:
                            # DC Coupled: Absolute max relative to 0V
                            v_top = np.percentile(volts, 99.9)
                            v_bot = np.percentile(volts, 0.1)
                            v_max_abs = max(abs(v_top), abs(v_bot))
                        
                        print(f"[AutoRange Debug] Range: {current_range_str} (Limit {v_limit}V) | Measured Signal: {v_max_abs:.4f}V | Raw Max: {np.max(np.abs(volts)):.4f}V")

                        # Case 1: CLIPPED
                        # If signal hitting >98% of range limit
                        if v_max_abs >= 0.98 * v_limit:
                            if curr_range_idx < len(range_list) - 1:
                                msg = f"RANGE UP: Signal {v_max_abs:.4f}V clipped on {current_range_str}. Switching UP."
                                self.logger.warning(msg)
                                print(f"[AutoRange] {msg}")
                                curr_range_idx += 1
                                # Cleanup and Retry
                                self.smu.disable_output()
                                try: self.smu.resource.write("ABOR") 
                                except: pass
                                time.sleep(0.2)
                                continue # NEXT RETRY
                            else:
                                self.logger.warning("Signal Clipped at Max Range!")
                                # Proceed with what we have
                        
                        # Case 2: UNDER-RANGE (Only if Auto Range is enabled and not at min)
                        # We should switch down if the signal would FIT in the next lower range
                        # with safety margin (e.g. use 80% of next lower range).
                        elif auto_range and (curr_range_idx > 0):
                             next_lower_idx = curr_range_idx - 1
                             next_lower_str = range_list[next_lower_idx]
                             next_lower_limit = range_limits.get(next_lower_str, 1.0)
                             
                             # Limit check: If 20MV is the lowest reliable range, don't go below it.
                             # User observation: "lowest range is 20mV". The "10MV" range might be invalid or unstable on this scope.
                             if next_lower_str == '10MV':
                                 pass # Don't go there
                             
                             # Check if signal fits in 80% of next lower range
                             elif v_max_abs < (0.8 * next_lower_limit):
                                 msg = f"RANGE DOWN: Signal {v_max_abs:.4f}V fits in {next_lower_str} ({next_lower_limit}V). Switching {current_range_str} -> {next_lower_str}."
                                 self.logger.info(msg)
                                 print(f"[AutoRange] {msg}")
                                 curr_range_idx -= 1
                                 
                                 # Cleanup and Retry
                                 self.smu.disable_output()
                                 try: self.smu.resource.write("ABOR") 
                                 except: pass
                                 time.sleep(0.2)
                                 continue # NEXT RETRY
                             
                             # ZERO SIGNAL SPECIAL CASE
                             # If we are effectively 0 (less than 1% of current limit) and not at bottom, forced drop
                             # This handles 8-bit quantization making small signals appear as exactly 0.
                             # But STOP at 20MV.
                             if v_max_abs < 0.01 * v_limit and current_range_str != '20MV':
                                 msg = f"RANGE DOWN (Zero Signal): {v_max_abs:.4f}V is negligible. Forcing drop."
                                 self.logger.info(msg)
                                 print(f"[AutoRange] {msg}")
                                 curr_range_idx -= 1
                                 self.smu.disable_output()
                                 try: self.smu.resource.write("ABOR") 
                                 except: pass
                                 time.sleep(0.2)
                                 continue
                     
                    # If we are here, range is Good or acceptable
                    # Now perform the actual averaging and SNR calculation
                    vpps = []
                    snrs = []
                    step_waveforms = None # Initialize to avoid UnboundLocalError
                    
                    for avg_idx in range(averages):
                         if self.stop_requested: break
                         
                         # If this is the first average, we already have 'times' and 'volts' from the range check
                         if avg_idx == 0:
                             # Use the 'volts' from the range check
                             pass 
                         else:
                             times, volts = self.scope.capture_block(tb_index, num_samples)
                         
                         if len(volts) > 0:
                            # analyze Pulse
                            v_high, v_low, vpp, snr = self.analyze_pulse_snr(volts)
                            vpps.append(vpp)
                            snrs.append(snr)
                            
                            # Store last waveform
                            if avg_idx == averages - 1:
                                step_waveforms = {
                                    "current": current_level,
                                    "times": times[::4], 
                                    "volts": volts[::4],
                                    "snr": snr,
                                    "r_ohms": resistor_ohms
                                }
                         
                         time.sleep(0.05)
                    
                    if step_waveforms:
                        waveforms.append(step_waveforms)

                    break # Break retry loop

                # 2. CRITICAL: Wait for SMU to finish the pulse train naturally
                pulse_train_time = period * req_cycles
                wait_time = pulse_train_time + 1.0 
                elapsed = 0
                while elapsed < wait_time:
                    if self.stop_requested: break
                    time.sleep(0.1)
                    elapsed += 0.1
                
                # --- STEP 5: CLEANUP ---
                self.smu.disable_output()
                try: self.smu.resource.write("ABOR") 
                except: pass
                
                # --- RESULT ---
                if not vpps:
                     self.logger.warning(f"No valid capture for step {i} (Retries exhausted or clipped)")
                     continue
                     
                avg_vpp = np.mean(vpps)
                std_vpp = np.std(vpps)
                avg_snr = np.mean(snrs) if snrs else 0.0
                
                photocurrent = abs(avg_vpp) / resistor_ohms
                
                results.append({
                    "LED_Current_A": current_level,
                    "Scope_Vpp": avg_vpp,
                    "Vpp_Std": std_vpp,
                    "SNR": avg_snr,
                    "Photocurrent_A": photocurrent,
                    "Resistance_Ohms": resistor_ohms,
                    "SNR_Status": "OK" if avg_snr > 10 else "LOW"
                })
                
                print(f"[{i+1}/{steps}] I={current_level:.2e}A -> Vpp={avg_vpp*1000:.1f}mV (SNR={avg_snr:.1f})")
                
                # Check SNR Trigger
                # Threshold from args
                if avg_snr < min_snr_threshold:
                     self.logger.info(f"SNR Low ({avg_snr:.1f} < {min_snr_threshold}). Pausing for Resistor check.")
                     # We raise exception to return control to UI
                     # Pass current state. User will decide whether to Retry (pop last result) or Continue (keep last result).
                     raise ResistorChangeRequiredException(i, avg_snr, current_level, results, waveforms)

            except ResistorChangeRequiredException:
                raise # Re-raise immediately to bubbling up
                
            except Exception as e:
                import traceback
                self.logger.error(f"Error at step {i}: {e}")
                print(f"!!! Error at Step {i+1} (I={current_level:.2e}A): {e}")
                traceback.print_exc()
                
                # Attempt Recovery
                try: 
                    self.smu.disable_output()
                    # If we stuck in ERROR, force back to IDLE so next loop allows configuring
                    # This import is usually at the top, but keeping it here for minimal diff
                    from hardware.smu_controller import InstrumentState 
                    if self.smu.state == InstrumentState.ERROR:
                        self.smu.to_state(InstrumentState.IDLE)
                except: 
                    pass
                
        return pd.DataFrame(results), waveforms

    def analyze_pulse_snr(self, volts: np.ndarray) -> (float, float, float, float):
        """
        Analyzes a square wave pulse to extract High/Low levels and SNR.
        Assumes roughly 50% duty cycle.
        """
        # 1. Determine Histograms/Levels
        # Simple method: Percentiles
        v_high = np.percentile(volts, 95)
        v_low = np.percentile(volts, 5)
        signal = v_high - v_low
        
        # 2. Estimate Noise
        # We assume points > mid_point are "High State" and < mid_point are "Low State"
        mid = (v_high + v_low) / 2
        high_vals = volts[volts > mid]
        low_vals = volts[volts < mid]
        
        # Avoid empty slices
        if len(high_vals) < 10 or len(low_vals) < 10:
            return v_high, v_low, signal, 0.0
            
        # Noise is std dev of the flat parts
        noise_high = np.std(high_vals)
        noise_low = np.std(low_vals)
        avg_noise = (noise_high + noise_low) / 2
        
        # SNR
        if avg_noise < 1e-9: avg_noise = 1e-9 # eps
        snr = signal / avg_noise
        
        return v_high, v_low, signal, snr

    def stop(self):
        self.stop_requested = True
