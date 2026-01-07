import time
import numpy as np
import pandas as pd
from typing import List, Dict, Callable
import logging
import os

from hardware.smu_controller import SMUController, InstrumentState
from hardware.scope_controller import ScopeController
from hardware.scope_controller import ScopeController
import analysis.signal_processing as signal_processing

class ResistorChangeRequiredException(Exception):
    def __init__(self, step_index, snr, current_level, last_results, last_waveforms, last_range_str="10V", step_waveform=None):
        self.step_index = step_index
        self.snr = snr
        self.current_level = current_level
        self.last_results = last_results
        self.last_waveforms = last_waveforms
        self.last_range_str = last_range_str
        self.step_waveform = step_waveform

class LDRWorkflow:
    def __init__(self, smu: SMUController, scope: ScopeController):
        self.smu = smu
        self.scope = scope
        self.logger = logging.getLogger("workflow.LDR")
        self.stop_requested = False
        self.last_range_idx = None

        self.mock = smu.mock or scope.mock
        
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
                  start_delay_cycles: int = 0,
                  min_pulse_cycles: int = 100, 
                  progress_callback: Callable[[float, str, dict], None] = None,
                  start_step_index: int = 0,
                  previous_results: list = None,
                  previous_waveforms: list = None,
                  autosave_path: str = None,
                  # New Params
                  acquisition_mode: str = "Block", # "Block" or "Streaming"
                  sample_rate: float = 100000.0,
                  capture_duration_sec: float = 2.0
                  ) -> tuple[pd.DataFrame, list[dict]]:
        """
        Executes the LDR sweep.
        """
        
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
        
        # Mode Switching
        is_streaming = (acquisition_mode == "Streaming")
        
        if is_streaming:
            # Streaming Setup
            # We respect capture_duration_sec
            capture_duration = capture_duration_sec
            num_samples = int(capture_duration * sample_rate)
            # tb_index not strictly needed for streaming, BU used in range-check capture_block calls
            # Calculate a default for short checks (e.g. 0.1s)
            tb_index = self.scope.calculate_timebase_index(0.1, 2000)
            self.logger.info(f"Mode: Streaming. Duration: {capture_duration}s, Rate: {sample_rate}Hz")
        else:
            # Block Setup (Legacy)
            # Capture enough for ~20 periods to ensure good spectral resolution (PSD)
            # But we respect capture_duration_sec if provided? 
            # Original logic: capture_duration = max(20 * period, 0.05)
            # Let's use user duration if provided, else default logic
            
            if capture_duration_sec > 0.1: # User override
                capture_duration = capture_duration_sec
            else:
                capture_duration = max(20 * period, 0.05) 
            
            num_samples = 16000 # Fixed for block
            tb_index = self.scope.calculate_timebase_index(capture_duration, num_samples)
            self.logger.info(f"Mode: Block. Duration: {capture_duration}s, Samples: {num_samples}")
        
        # 1. Start with Output OFF to ensure clean state
        try:
             self.smu.set_current(0.0) # Set to 0A first
             self.smu.disable_output()
        except: 
             pass
             
        time.sleep(0.5) 
        
        # 2. Enable output ONCE before the sweep starts
        try:
            self.smu.set_compliance(compliance_limit, "VOLT")
            self.smu.enable_output()
            self.logger.info("SMU Output enabled for entire sweep.")
        except Exception as e:
            self.logger.error(f"Failed to enable SMU output: {e}")
            raise e
        
        # Available ranges ordered Low to High
        range_list = ['10MV', '20MV', '50MV', '100MV', '200MV', '500MV', '1V', '2V', '5V', '10V']
        
        # Limits mapping (approx)
        range_limits = {
            '10MV': 0.01, '20MV': 0.02, '50MV': 0.05, '100MV': 0.1, 
            '200MV': 0.2, '500MV': 0.5, '1V': 1.0, '2V': 2.0, '5V': 5.0, '10V': 10.0
        }
        
        # Determine initial index
        # If Resuming (start_step_index > 0) AND we have a memory of last range, use it.
        # Otherwise use user setting.
        
        start_idx_candidate = None
        if start_step_index > 0 and self.last_range_idx is not None:
             start_idx_candidate = self.last_range_idx
             self.logger.info(f"Resuming sweep: Using last known range index {start_idx_candidate} ({range_list[start_idx_candidate]})")
        
        if start_idx_candidate is not None:
            curr_range_idx = start_idx_candidate
        else:
            try:
                curr_range_idx = range_list.index(scope_range.upper())
            except:
                curr_range_idx = 9 # Default 10V
            
        self.logger.info(f"Starting LDR Sweep: {steps} steps (Running {len(levels_to_run)}), {start_current:.2e}A -> {stop_current:.2e}A, Freq={frequency}Hz, Start Range={range_list[curr_range_idx]}")
        print(f"DEBUG: autosave_path received = '{autosave_path}'")
        
        # Loop
        for i, current_level in levels_to_run:
            if self.stop_requested: break
            
            if progress_callback:
                progress_callback((i / steps), f"Step {i+1}/{steps}: {current_level*1000:.2f} mA", {})
            
            # WORKFLOW SAFETY CHECK: Monotonicity
            # Ensure we are not increasing current (going backwards) which could be dangerous
            # Note: We skip this check for re-runs of the same point (i.e. resistor change retry)
            # but here we iterate through distinct levels.
            # This relies on usage of 'levels_to_run' list.
            if i > 0: # Check against previous STEP (original_index - 1)
               # We need to find the previous level in the FULL list.
               # current_levels_full was used to generate this.
               # But locally, we can just ensure:
               # If this is a descending sweep, verify current <= previous_in_loop
               # Wait, if we resume components, levels_to_run might be a slice. 
               pass # Logic is tricky with Resumes.
               # Simpler check: If start > stop, ensure current level <= start_current
               if start_current > stop_current and current_level > (start_current * 1.01):
                    raise ValueError(f"Safety Violation: Current {current_level} exceeds Start Current {start_current}!")

            # Check vs Previous iteration loop value
            # Since we iterate levels_to_run sequentially:
            # We can store 'last_executed_current' variable.
            if hasattr(self, 'last_executed_current') and self.last_executed_current is not None:
                # Allow small floating point tolerance or equality (re-measure)
                # But strictly NO INCREASE > 1%
                if start_current >= stop_current: # Downward Sweep
                    if current_level > self.last_executed_current * 1.01:
                         raise ValueError(f"Safety Violation: Current increased from {self.last_executed_current} to {current_level} during downward sweep!")
            
            self.last_executed_current = current_level
            
            try:
                # --- STEP 1: TEARDOWN / RESET ---
                # We stay in "OUTP ON" state but stop any running sequences
                try: 
                    self.smu.resource.write("ABOR") 
                    self.smu.set_current(0.0) # Ensure 0A between steps
                except: pass
                
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
                    if self.stop_requested: break
                    retry_count += 1
                    current_range_str = range_list[curr_range_idx]
                    
                    # --- STEP 2: CONFIGURE ---
                    self.smu.set_compliance(compliance_limit, "VOLT")
                    
                    # Pulse Logic
                    
                    # If streaming, we might need continuous pulses for a LONG time.
                    # Cycles = Duration / Period
                    # For 10s at 80Hz = 800 cycles.
                    # Ensure pulse train covers the capture duration
                    
                    # Capture Time
                    # Streaming: duration exactly. Block: duration + overhead.
                    
                    # Capture Time
                    # Must cover all averages + overheads
                    # Streaming takes exact duration, but Block mode overhead is higher.
                    # We add a generous 2.0s buffer + 10% to be safe.
                    
                    req_time = (capture_duration + 0.2) * averages + 2.0 
                    req_cycles = int(req_time / period) + 20
                    
                    if req_cycles < 10: req_cycles = 10
                    if req_cycles < min_pulse_cycles: req_cycles = min_pulse_cycles
                    
                    self.smu.generate_square_wave(current_level, 0.0, period, duty_cycle, req_cycles, "CURR")
                    
                    # Configure Scope (with retry for State Errors)
                    try:
                        coupling_str = "AC" if ac_coupling else "DC"
                        self.scope.configure_channel('A', True, current_range_str, coupling_str)
                    except RuntimeError as e:
                        if "InstrumentState" in str(e) or "state" in str(e):
                             self.logger.warning(f"Scope state error detected: {e}. Attempting reset...")
                             self.scope.reset_device()
                             time.sleep(0.5)
                             # Retry config
                             self.scope.configure_channel('A', True, current_range_str, coupling_str)
                        else:
                            raise e
                    
                    # --- STEP 3: ENABLE & RUN ---
                    self.smu.enable_output()
                    self.smu.trigger_list() # Starts the train
                    
                    # Optional user delay (Warm-up / Settle)
                    if start_delay_cycles > 0:
                        # Sleep for N cycles before starting capture
                        # SMU is already running in background
                        delay_time = start_delay_cycles * period
                        time.sleep(delay_time)
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
                                self.smu.set_current(0.0)
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
                                 self.smu.set_current(0.0)
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
                                 self.smu.set_current(0.0)
                                 try: self.smu.resource.write("ABOR") 
                                 except: pass
                                 time.sleep(0.2)
                                 continue # NEXT RETRY
                    
                    # --- UPDATE PREVIEW after trial capture ---
                    if progress_callback and len(volts) > 0:
                        _, _, trial_vpp, trial_snr = self.analyze_pulse_snr(volts)
                        progress_callback((i/steps), f"Step {i+1}/{steps}: Measuring...", {
                            "vpp": trial_vpp,
                            "snr": trial_snr,
                            "range": current_range_str,
                            "times": times[::4], 
                            "volts": volts[::4]
                        })

                    # If we are here, range is Good or acceptable
                    # Now perform the actual averaging and SNR calculation
                    vpps = []
                    lockin_amps = [] # New
                    noise_densities = [] # New
                    snrs_fft = [] # Spectral (One shot)
                    snrs_time = [] # Time Domain (Averaged)
                    step_waveforms = {} # Initialize to avoid UnboundLocalError
                    
                    for avg_idx in range(averages):
                         if self.stop_requested: break
                         
                         # Capture
                         if acquisition_mode == "Streaming":
                             times, volts = self.scope.capture_streaming(capture_duration_sec, sample_rate)
                         else:
                             times, volts = self.scope.capture_block(tb_index, num_samples)
                         
                         if len(volts) > 0:


                         
                             # analyze Pulse (Time Domain) - used for averaging Vpp
                             v_high, v_low, vpp, s_time = self.analyze_pulse_snr(volts)
                             vpps.append(vpp)
                             snrs_time.append(s_time)
                             
                             if len(times) > 1:
                                 fs = 1.0 / (times[1] - times[0])
                                 
                                 # Calculate Robust Metrics
                                 l_amp = signal_processing.calculate_lockin_amplitude(volts, fs, frequency)
                                 lockin_amps.append(l_amp)
                                 
                                 n_dens = signal_processing.calculate_noise_density_sideband(volts, fs, frequency)
                                 noise_densities.append(n_dens)

                             
                             # analyze Pulse (Frequency Domain)
                             # IMPORTANT: We only calculate Noise/SNR from the FIRST block (or result averaging)
                             # to avoid "artificially reducing noise" via block averaging.
                             if avg_idx == 0:
                                 if len(times) > 1:
                                     fs = 1.0 / (times[1] - times[0])
                                     s_fft = signal_processing.calculate_snr_fft(volts, fs, frequency)
                                     snrs_fft.append(s_fft)
                                     
                                     # Prep FFT for Storage
                                     win = np.hamming(len(volts))
                                     ft = np.fft.rfft((volts - np.mean(volts)) * win)
                                     # Standard normalization: 2.0 / sum(window)
                                     mag = np.abs(ft) * (2.0 / np.sum(win))
                                     ffreqs = np.fft.rfftfreq(len(volts), 1/fs)
                                 else:
                                     snrs_fft.append(s_time)
                                     ffreqs, mag = np.array([]), np.array([])
                             
                             # Store last waveform (Raw)
                             if avg_idx == 0:
                                 step_waveforms = {
                                     "current": current_level,
                                     "times": times[::2], # Light downsample for UI speed
                                     "volts": volts[::2],
                                     "fft_freqs": ffreqs, # Keep full FFT for accuracy
                                     "fft_mag": mag,
                                     "snr_fft": snrs_fft[0],
                                     "snr_time": snrs_time[0],
                                     "r_ohms": resistor_ohms
                                 }
                         
                         # Autosave Trace (If enabled and path provided)
                         if autosave_path and len(times) > 0:
                             try:
                                 # Ensure directory exists (workflow safety)
                                 # Although UI creates it, we double check.
                                 if not os.path.exists(autosave_path):
                                     os.makedirs(autosave_path, exist_ok=True)
                                 
                                 # Filename: trace_step_{i}_{Current}.csv
                                 # Use 'i' (step index) and 'current_level'
                                 filename = f"trace_step_{i+1}_I{current_level:.2e}A_avg{avg_idx+1}.csv"
                                 # Sanitize filename if needed (scientific notation is mostly safe on windows, wait, colon is not safe!)
                                 # .2e gives 1.00e-03 which is safe.
                                 
                                 full_path = os.path.join(autosave_path, filename)
                                 
                                 # Save DataFrame
                                 # Use light downsample? No, raw trace requested.
                                 df_trace = pd.DataFrame({'time': times, 'voltage': volts})
                                 df_trace.to_csv(full_path, index=False)
                                 # self.logger.info(f"Saved trace: {filename}")
                             except Exception as exc:
                                 self.logger.error(f"Failed to autosave trace {filename}: {exc}")
                         
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
                self.smu.set_current(0.0)
                try: self.smu.resource.write("ABOR") 
                except: pass
                
                # --- RESULT ---
                if not vpps:
                     self.logger.warning(f"No valid capture for step {i} (Retries exhausted or clipped)")
                     continue
                     
                avg_vpp = np.mean(vpps)
                std_vpp = np.std(vpps)
                avg_snr_time = np.mean(snrs_time)
                avg_snr_fft = snrs_fft[0] if snrs_fft else 0
                
                # Lock-In Results
                avg_lockin_amp = np.mean(lockin_amps) if lockin_amps else 0
                avg_noise_dens = np.mean(noise_densities) if noise_densities else 0
                
                # Adjusted SNR: Signal (LockIn) / NoiseFloor (Density * sqrt(BW?))
                # For direct SNR ratio: LockIn / (NoiseDens * sqrt(1Hz)) = Signal/Noise_in_1Hz
                # Technically SNR = Power ratio? 
                # Let's report "SNR_1Hz" = (V_sig / V_noise_rtHz)**2 ??
                # Or just Amplitude SNR: V_sig / V_noise_rtHz
                
                calc_snr_1hz = 0
                if avg_noise_dens > 0:
                     calc_snr_1hz = avg_lockin_amp / avg_noise_dens
                
                # Photocurrent from LockIn (More accurate than Vpp)
                photocurrent = avg_lockin_amp / resistor_ohms
                
                # Calculate Johnson Noise (Thermal Limit)
                johnson_noise_dens = signal_processing.calculate_johnson_noise(resistor_ohms)
                
                # Excess Noise Factor (Measured / Thermal)
                excess_noise = 0
                if johnson_noise_dens > 0:
                    excess_noise = avg_noise_dens / johnson_noise_dens
                
                # Quantization Check
                # 8-bit scope = 256 levels.
                # If Vpp < 5 levels (approx 2% of range), we are in quantization noise.
                scope_range_v = range_limits.get(current_range_str, 10.0)
                lsb_v = scope_range_v / 256.0 # Approx full scale / 256? Or half scale? 
                # PicoScope ranges are usually +/- Range (so Span = 2*Range).
                # LSB = (2 * Range) / 256 = Range / 128
                lsb_v = scope_range_v / 128.0 
                
                quantization_warning = False
                if avg_vpp < 5 * lsb_v:
                    self.logger.warning(f"Signal Vpp ({avg_vpp:.2e}V) is < 5 LSBs of Range {current_range_str}. Results may be quantization limited.")
                    quantization_warning = True
                
                results.append({
                    "LED_Current_A": current_level,
                    "Scope_Vpp": avg_vpp,
                    "Vpp_Std": std_vpp,
                    "SNR_Time": avg_snr_time,
                    "SNR_FFT": avg_snr_fft,
                    "Photocurrent_A": photocurrent,
                    "Resistance_Ohms": resistor_ohms,
                    "SNR_Status": "OK" if avg_snr_time > min_snr_threshold else "LOW",
                    "LockIn_Amp_V": avg_lockin_amp,
                    "Noise_Density_V_rtHz": avg_noise_dens,
                    "SNR_Broadband": calc_snr_1hz,
                    "Johnson_Noise_V_rtHz": johnson_noise_dens,
                    "Excess_Noise_Factor": excess_noise,
                    "Quantization_Limited": quantization_warning
                })
                
                print(f"[{i+1}/{steps}] I={current_level:.2e}A | Vpp={avg_vpp:.2e}V | Noise={avg_noise_dens:.2e} V/rtHz (Thermal {johnson_noise_dens:.2e} V/rtHz) | Q-Lim: {quantization_warning}")
                
                # Save state for persistence
                self.last_range_idx = curr_range_idx
                
                # Check SNR Trigger (Using Time Domain SNR as requested)
                # If signal is noisy in time domain, we likely need a larger resistor to get more voltage.
                if avg_snr_time < min_snr_threshold:
                    self.logger.warning(f"SNR_Time ({avg_snr_time:.1f}) below threshold ({min_snr_threshold})")
                    # Safely disable SMU output before raising pause exception
                    try:
                        self.smu.set_current(0.0)
                        self.smu.disable_output()
                    except:
                        pass
                    raise ResistorChangeRequiredException(i, avg_snr_time, current_level, results, waveforms, range_list[curr_range_idx], step_waveform=step_waveforms)

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
                
        # Final Cleanup: Always ensure output is off when finished
        try:
            self.smu.set_current(0.0)
            self.smu.disable_output()
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
