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
    "sweep_start": 10e-3,
    "sweep_stop": 10e-6,
    "sweep_steps": 10,
    "capture_delay_cycles": 0,
    "sweep_freq": 40.0,
    "acquisition_mode": "Block",
    "capture_duration": 0.5,
    "sample_rate": 100000.0,
    "scope_range_idx": 4
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
            return {**DEFAULT_SETTINGS, **data}
        except Exception as e:
            self.logger.error(f"Failed to load settings from {self.config_file}: {e}")
            return DEFAULT_SETTINGS.copy()

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
