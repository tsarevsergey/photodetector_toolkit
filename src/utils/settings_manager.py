import json
import os
import logging
from typing import Any, Dict

DEFAULT_SETTINGS = {
    "base_save_folder": "data",
    "save_raw_traces": False,
    "last_led_wavelength": 461.0,
    "last_ref_file": "data/SiDiodeResponsivity.csv",
    "last_meas_file": "data/Si_cal1.csv",
    "snr_threshold": 25.0,
    "min_cycles": 100,
    "led_compliance": 5.0,
    "averages": 3,
    "resistor": 47000.0,
    "sample_name": "Sample_1",
    "sweep_start": 10e-3,
    "sweep_stop": 10e-6,
    "sweep_steps": 10,
    "capture_delay_cycles": 0,
    "sweep_freq": 40.0,
    "duty_cycle": 0.5,
    "acquisition_mode": "Block",
    "global_amp_type": "Passive Resistor",
    "global_resistor_val": 47000.0,
    "global_tia_gain": 1e6,
    "global_safety_max_v": 9.5,
    "last_dut_file": "",
    "capture_duration": 0.5,
    "sample_rate": 100000.0,
    "scope_range_idx": 6, # 1V default
    "auto_range": True,
    "ac_coupling": False,
    "suppress_info_logs": True,
    # Pulse Generator
    "pulse_mode": "Current",
    "pulse_high": 1e-3,
    "pulse_low": 0.0,
    "pulse_compliance": 8.0,
    "pulse_freq": 10.0,
    "pulse_duty": 0.5,
    "pulse_cycles": 100,
    "pulse_measure_resistor": 47000.0,
    "pulse_scope_range": "10V",
    "pulse_duration": 0.05,
    "pulse_samples": 2000,
    "pulse_ac_coupling": False,
    "pulse_delay_cycles": 0,
    # IV Sweep
    "iv_start": -1.0,
    "iv_stop": 1.0,
    "iv_steps": 21,
    "iv_mode": "Linear",
    "iv_direction": "Single",
    "iv_nplc": 1.0,
    "iv_compliance": 0.01,
    # Scope Commissioning
    "scope_ch_a_range": "1V",
    "scope_ch_a_coupling": "DC",
    "scope_ch_a_enabled": True,
    "scope_ch_b_range": "1V",
    "scope_ch_b_coupling": "DC",
    "scope_ch_b_enabled": False,
    "scope_acq_mode": "Block",
    "scope_block_duration_ms": 20.0,
    "scope_streaming_duration_s": 2.0,
    "scope_num_samples": 2000,
    "scope_sample_rate": 100000.0,
    # Scope Commissioning - Quick Overrides
    "com_quick_coupling": "DC",
    "com_quick_range": "2V",
    "com_quick_tia_gain": 1000.0,
    # Scope Commissioning - Noise
    "noise_source_r": 1000.0,
    "noise_cal_gain": 1000.0,
    "noise_cal_range": "10MV",
    "noise_cal_coupling": "AC",
    "noise_cal_duration": 1.0,
    "noise_f_start": 10.0,
    "noise_f_stop": 10000.0,
    # Scope Commissioning - Detectivity
    "det_area": 1.0,
    "det_resp": 0.5,
    "det_gain": 1e6,
    "det_input_mode": "Live Capture",
    "det_range": "10MV",
    "det_coupling": "AC",
    "det_duration": 1.0,
    "det_f_start": 10.0,
    "det_f_end": 1000.0,
    "det_exp_name": "Detectivity_Run_1",
    "det_manual_v": 1e-6
}

class SettingsManager:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = os.path.abspath(config_file)
        self.logger = logging.getLogger(__name__)
        self.settings = self.load_settings()

    def load_settings(self) -> Dict[str, Any]:
        """Loads settings from JSON file, falling back to defaults."""
        if not os.path.exists(self.config_file):
            return DEFAULT_SETTINGS.copy()
        
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
            # Merge with defaults to ensure all keys exist
            merged = {**DEFAULT_SETTINGS, **data}
            return self._sanitize_settings(merged)
        except Exception as e:
            self.logger.error(f"Failed to load settings from {self.config_file}: {e}")
            return DEFAULT_SETTINGS.copy()

    def _sanitize_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Fixes invalid values that might have been saved (e.g. 0.0 resistor)."""
        # Fix Resistor 0.0
        if settings.get("global_resistor_val", 0.0) <= 0:
            settings["global_resistor_val"] = DEFAULT_SETTINGS["global_resistor_val"]
        
        # Fix TIA Gain 0.0
        if settings.get("global_tia_gain", 0.0) <= 0:
            settings["global_tia_gain"] = DEFAULT_SETTINGS["global_tia_gain"]
            
        # Fix Pulse Frequency 0.0
        if settings.get("pulse_freq", 0.0) <= 0:
            settings["pulse_freq"] = DEFAULT_SETTINGS["pulse_freq"]

        return settings

    def save_settings(self, new_settings: Dict[str, Any]) -> None:
        """Updates and saves settings to JSON file."""
        self.settings.update(new_settings)
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            self.logger.error(f"Failed to save settings: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.settings[key] = value
        self.save_settings({key: value}) # Auto-save on set? Or manual save? Let's auto-save.
