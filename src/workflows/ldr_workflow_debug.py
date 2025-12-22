
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Callable
import logging

from hardware.smu_controller import SMUController
from hardware.scope_controller import ScopeController

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
                  progress_callback: Callable[[float, str], None] = None) -> pd.DataFrame:
        
        self.stop_requested = False
        results = []
        
        # ... (Logspace gen)
        s = start_current if start_current > 0 else 1e-9
        current_levels = np.logspace(np.log10(s), np.log10(stop_current), steps)
        
        # Scope settings
        period = 1.0 / frequency
        # Capture enough for ~3 periods min, or 20ms default
        # If averaging, we still capture one block per average loop, or one long block?
        # Better to capture separate blocks to reduce sync noise.
        capture_duration = max(3 * period, 0.02) 
        
        # Calculate optimal timebase
        num_samples = 2000
        tb_index = self.scope.calculate_timebase_index(capture_duration, num_samples)
        
        # 1. Start with Output OFF to ensure clean state
        # Force disable to reset any previous states
        try:
             self.smu.disable_output()
        except: 
             pass
             
        time.sleep(0.5) # Allow relays to open
        
        self.logger.info(f"Starting LDR Sweep: {steps} steps (Avg {averages}), {start_current:.2e}A -> {stop_current:.2e}A")
        
        # Turn Output ON *once* for the whole sweep (prevents relay clicking)
        # We will just update the pulse list dynamically while output is ON? 
        # Some SMUs don't allow re-config while ON.
        # But looping Enable/Disable is definitely slow (relays).
        # Let's try: Enable -> [Config List -> Trigger -> Capture] -> Disable (only at very end)
        
        # Actually, B2900 needs Output OFF to change Trigger config usually.
        # But maybe we can keep it ON if we just change Source Memory?
        # For safety/reliability with your specific clicking issue:
        # We must accept the relay click time. We just need to WAIT for it.
        
        for i, current_level in enumerate(current_levels):
            # ... (Stop check)
            if self.stop_requested: break
                
            if progress_callback:
                progress_callback((i / steps), f"Step {i+1}/{steps}: {current_level*1000:.2f} mA (Avg x{averages})")
                
            try:
                # 1. Setup SMU Pulse
                req_cycles = int(capture_duration / period) + 50 
                if current_level > 0.1: current_level = 0.1
                
                # We typically must be OFF to re-write the list source.
                self.smu.disable_output() 
                # Relay click OFF
                
                self.smu.set_compliance(compliance_limit, "VOLT") 
                self.smu.generate_square_wave(current_level, 0.0, period, duty_cycle, req_cycles, "CURR")
                
                # Averaging Loop
                vpps = []
                v_lows = []
                
                # Relay click ON
                self.smu.enable_output()
                time.sleep(0.5) # Wait for Relay bounce/settling
                
                for avg_idx in range(averages):
                     if self.stop_requested: break
                     
                     # Trigger Pulse (Software Trigger for List)
                     self.smu.trigger_list()
                     
                     # Wait for pulse train to stabilize
                     time.sleep(0.1) 
                     
                     # --- Removed intrusive SMU Validation ---
                     # Trust the Scope. If Vpp is detected, SMU was running.
                     # ----------------------------------------

                     # Capture
                     times, volts = self.scope.capture_block(tb_index, num_samples)
                     
                     # Analyze Single Shot
                     if len(volts) > 0:
                        v_high = np.percentile(volts, 95)
                        v_low = np.percentile(volts, 5)
                        vpps.append(v_high - v_low)
                        v_lows.append(v_low)
                        
                     time.sleep(0.02) # Short cooldown
                
                # We keep Output ON until we need to reconfigure
                # Actually, loop restarts -> reconfig -> needs OFF.
                # So we must Disable here.
                self.smu.disable_output()
                
                if not vpps:
                    self.logger.warning(f"No valid captures for step {i}")
                    continue
                    
                avg_vpp = np.mean(vpps)
                std_vpp = np.std(vpps)
                v_low_avg = np.mean(v_lows) if 'v_lows' in locals() else 0 
                
                photocurrent = abs(avg_vpp) / resistor_ohms
                
                results.append({
                    "LED_Current_A": current_level,
                    "Scope_Vpp": avg_vpp,
                    "V_High_Avg": np.mean(vpps) + v_low_avg, 
                    "Vpp_Std": std_vpp,
                    "Photocurrent_A": photocurrent,
                    "Resistance_Ohms": resistor_ohms
                })
                
                # Console feedback per step
                print(f"[{i+1}/{steps}] I_led={current_level:.2e}A -> Vpp={avg_vpp*1000:.1f}mV")
                
                # Delay between steps (Requested by User)
                time.sleep(0.5)
                
            except Exception as e:
                import traceback
                self.logger.error(f"Error at step {i}: {e}")
                print(f"!!! Error at Step {i+1} (I={current_level:.2e}A): {e}")
                traceback.print_exc()
                
                # Attempt Recovery
                try: 
                    self.smu.disable_output()
                    # If we stuck in ERROR, force back to IDLE so next loop allows configuring
                    if self.smu.state == InstrumentState.ERROR:
                        self.smu.to_state(InstrumentState.IDLE)
                except: 
                    pass
                
        return pd.DataFrame(results)

    def stop(self):
        self.stop_requested = True
