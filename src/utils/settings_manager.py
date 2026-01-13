import os
import logging
import configparser
from typing import Any, Dict

DEFAULT_SETTINGS = {
    "base_save_folder": "data",
    "save_raw_traces": False,
    "snr_threshold": 25.0,
    "min_cycles": 100,
    "led_compliance": 5.0,
    "averages": 3,
    "resistor": 47000.0,
    "sweep_start": 0.01,
    "sweep_stop": 1e-05,
    "sweep_steps": 10,
    "capture_delay_cycles": 0,
    "sweep_freq": 40.0,
    "duty_cycle": 0.5,
    "acquisition_mode": "Block",
    "capture_duration": 0.5,
    "sample_rate": 100000.0,
    "scope_range_idx": 6, # 1V default
    "auto_range": True,
    "ac_coupling": False,
    "suppress_info_logs": True,
    "global_amp_type": "Passive Resistor",
    "global_resistor_val": 1000.0,
    "global_tia_gain": 1e9,
    "global_safety_max_v": 10.0,
    "last_led_wavelength": 461.0,
    "last_ref_file": "data/SiDiodeResponsivity.csv",
    "last_meas_file": "data/Si_cal1.csv",
    "last_dut_file": "",
    "last_trace_file": "",
    # Scope Commissioning defaults
    "scope_comm_range_a": "2V",
    "scope_comm_coupling_a": "DC",
    "scope_comm_acq_mode": "Block",
    "scope_comm_duration_ms": 20.0,
    "scope_comm_num_samples": 2000,
    "scope_comm_sample_rate": 100000.0,
    "scope_comm_quick_range": "2V",
    "scope_comm_quick_coupling": "DC",
    "scope_comm_tia_gain": 1000.0,
    # Pulse Generator defaults
    "pulse_gen_mode": "Current",
    "pulse_gen_high_level": 0.001,
    "pulse_gen_low_level": 0.0,
    "pulse_gen_compliance": 8.0,
    "pulse_gen_frequency": 10.0,
    "pulse_gen_duty_cycle": 0.5,
    "pulse_gen_cycles": 100,
    "pulse_gen_resistor": 47000.0,
    "pulse_gen_scope_range": "10V",
    "pulse_gen_capture_duration": 0.05,
    "pulse_gen_samples": 2000,
    "pulse_gen_delay_cycles": 0,
    "smu_visa_address": "USB0::0x0957::0xCD18::MY51143841::0::INSTR",
    "sample_name": "Sample_1",
    "scope_comm_gain": 1000.0,
    "ldr_resistor": 47000.0,
    "ldr_tia_gain": 1000000000.0,
    "smu_mock_mode": False,
    "scope_mock_mode": False,
    "ldr_save_data": True,
    "smu_sweep_start": -1.0,
    "smu_sweep_stop": 1.0,
    "smu_sweep_steps": 21,
    "smu_sweep_mode": "Linear",
    "smu_sweep_dir": "Single",
    "smu_sweep_nplc": 1.0,
    "smu_sweep_comp": 0.01,
    "smu_manual_mode": "Voltage",
    "smu_manual_val": 0.0,
    "smu_manual_comp": 0.1,
    "last_led_wavelength": 461.0,
    "last_ref_area": 1.0,
    "last_dut_area": 1.0,
    "last_ref_file": "",
    "last_meas_file": "",
    "last_dut_file": "",
    "last_trace_file": ""
}

# Mapping of keys to sections for INI organization
SECTION_MAP = {
    "base_save_folder": "GENERAL",
    "save_raw_traces": "GENERAL",
    "suppress_info_logs": "GENERAL",
    "snr_threshold": "LDR_MEASUREMENT",
    "min_cycles": "LDR_MEASUREMENT",
    "acquisition_mode": "LDR_MEASUREMENT",
    "capture_duration": "LDR_MEASUREMENT",
    "sample_rate": "LDR_MEASUREMENT",
    "sweep_start": "LDR_MEASUREMENT",
    "sweep_stop": "LDR_MEASUREMENT",
    "sweep_steps": "LDR_MEASUREMENT",
    "sweep_freq": "LDR_MEASUREMENT",
    "duty_cycle": "LDR_MEASUREMENT",
    "capture_delay_cycles": "LDR_MEASUREMENT",
    "led_compliance": "LDR_MEASUREMENT",
    "averages": "LDR_MEASUREMENT",
    "resistor": "LDR_MEASUREMENT",
    "scope_range_idx": "LDR_MEASUREMENT",
    "auto_range": "LDR_MEASUREMENT",
    "ac_coupling": "LDR_MEASUREMENT",
    "scope_comm_range_a": "SCOPE_COMMISSIONING",
    "scope_comm_coupling_a": "SCOPE_COMMISSIONING",
    "scope_comm_acq_mode": "SCOPE_COMMISSIONING",
    "scope_comm_duration_ms": "SCOPE_COMMISSIONING",
    "scope_comm_num_samples": "SCOPE_COMMISSIONING",
    "scope_comm_sample_rate": "SCOPE_COMMISSIONING",
    "scope_comm_quick_range": "SCOPE_COMMISSIONING",
    "scope_comm_quick_coupling": "SCOPE_COMMISSIONING",
    "scope_comm_tia_gain": "SCOPE_COMMISSIONING",
    "pulse_gen_mode": "PULSE_GENERATOR",
    "pulse_gen_high_level": "PULSE_GENERATOR",
    "pulse_gen_low_level": "PULSE_GENERATOR",
    "pulse_gen_compliance": "PULSE_GENERATOR",
    "pulse_gen_frequency": "PULSE_GENERATOR",
    "pulse_gen_duty_cycle": "PULSE_GENERATOR",
    "pulse_gen_cycles": "PULSE_GENERATOR",
    "pulse_gen_resistor": "PULSE_GENERATOR",
    "pulse_gen_scope_range": "PULSE_GENERATOR",
    "pulse_gen_capture_duration": "PULSE_GENERATOR",
    "pulse_gen_samples": "PULSE_GENERATOR",
    "pulse_gen_ac_coupling": "PULSE_GENERATOR",
    "pulse_gen_delay_cycles": "PULSE_GENERATOR",
    "global_amp_type": "GLOBAL_DEVICE",
    "global_resistor_val": "GLOBAL_DEVICE",
    "global_tia_gain": "GLOBAL_DEVICE",
    "global_safety_max_v": "GLOBAL_DEVICE",
    "last_led_wavelength": "POST_PROCESSING",
    "last_ref_file": "POST_PROCESSING",
    "last_meas_file": "POST_PROCESSING",
    "last_dut_file": "POST_PROCESSING",
    "last_trace_file": "POST_PROCESSING",
    "smu_visa_address": "GLOBAL_DEVICE",
    "sample_name": "GENERAL",
    "scope_comm_gain": "SCOPE_COMMISSIONING",
    "ldr_resistor": "LDR_MEASUREMENT",
    "ldr_tia_gain": "LDR_MEASUREMENT",
    "ldr_save_data": "LDR_MEASUREMENT",
    "smu_mock_mode": "GLOBAL_DEVICE",
    "scope_mock_mode": "GLOBAL_DEVICE",
    "smu_sweep_start": "SMU_CONTROL",
    "smu_sweep_stop": "SMU_CONTROL",
    "smu_sweep_steps": "SMU_CONTROL",
    "smu_sweep_mode": "SMU_CONTROL",
    "smu_sweep_dir": "SMU_CONTROL",
    "smu_sweep_nplc": "SMU_CONTROL",
    "smu_sweep_comp": "SMU_CONTROL",
    "smu_manual_mode": "SMU_CONTROL",
    "smu_manual_val": "SMU_CONTROL",
    "smu_manual_comp": "SMU_CONTROL"
}

# To switch to beta, change CONFIG_FILENAME to "settings.ini" or "config_beta.ini"
CONFIG_FILENAME = "settings.ini"
ALT_CONFIG_FILENAME = "config_beta.ini"

# Calculate absolute path relative to the 'src' directory (parent of 'utils')
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(UTILS_DIR)
DEFAULT_CONFIG_PATH = os.path.join(SRC_DIR, CONFIG_FILENAME)

class SettingsManager:
    def __init__(self, config_file: str = None):
        if config_file is None:
            # Try primary, then secondary
            if os.path.exists(DEFAULT_CONFIG_PATH):
                config_file = DEFAULT_CONFIG_PATH
            else:
                alt_path = os.path.join(SRC_DIR, ALT_CONFIG_FILENAME)
                if os.path.exists(alt_path):
                    config_file = alt_path
                else:
                    config_file = DEFAULT_CONFIG_PATH # Fallback to primary for creation
            
        self.config_file = os.path.abspath(config_file)
        self.logger = logging.getLogger(__name__)
        self._last_loaded_time = 0
        self.settings = DEFAULT_SETTINGS.copy()
        self.load_settings()

    def disk_is_newer(self):
        """Checks if the configuration file on disk has been modified since we last loaded it."""
        if not os.path.exists(self.config_file):
            return False
        try:
            return os.path.getmtime(self.config_file) > self._last_loaded_time
        except:
            return False

    def get_config_path(self) -> str:
        """Returns the absolute path to the current config file."""
        return self.config_file

    def _cast_value(self, key: str, value: str) -> Any:
        """Casts string from INI to the correct type based on DEFAULT_SETTINGS."""
        if key not in DEFAULT_SETTINGS:
            return value
        
        default_val = DEFAULT_SETTINGS[key]
        value = value.strip()
        if isinstance(default_val, bool):
            return value.lower() in ('true', 'yes', '1', 'on')
        if isinstance(default_val, int):
            try: return int(value)
            except: return default_val
        if isinstance(default_val, float):
            try: return float(value)
            except: return default_val
        return value

    def reload(self) -> None:
        """Force a fresh load from the config file."""
        self.settings = self.load_settings()

    def load_settings(self) -> Dict[str, Any]:
        """Loads settings from INI file, falling back to defaults."""
        if not os.path.exists(self.config_file):
            print(f"⚠️ WARNING: Config file not found at {self.config_file}. Using hardcoded defaults.")
            return DEFAULT_SETTINGS.copy()
        
        try:
            config = configparser.ConfigParser()
            config.read(self.config_file)
            
            data = {}
            for section in config.sections():
                for key, value in config.items(section):
                    data[key] = self._cast_value(key, value)
            
            self._last_loaded_time = os.path.getmtime(self.config_file)
            # Merge with defaults
            self.settings = {**DEFAULT_SETTINGS, **data}
            return self.settings
        except Exception as e:
            print(f"❌ ERROR: Failed to load {self.config_file}: {e}. Using defaults.")
            self.logger.error(f"Failed to load settings: {e}")
            return DEFAULT_SETTINGS.copy()

    def save_settings(self, new_settings: Dict[str, Any]) -> None:
        """Updates and saves settings to INI file, attempting to preserve comments."""
        self.settings.update(new_settings)
        
        # If file doesn't exist, create it from scratch using ConfigParser
        if not os.path.exists(self.config_file):
            config = configparser.ConfigParser()
            for key, val in self.settings.items():
                section = SECTION_MAP.get(key, "OTHER")
                if section not in config:
                    config.add_section(section)
                config.set(section, key, str(val))
            
            try:
                with open(self.config_file, 'w') as f:
                    config.write(f)
            except Exception as e:
                self.logger.error(f"Failed to create config file: {e}")
            return

        # Smart Save: Try to preserve comments by reading lines and replacing values
        try:
            with open(self.config_file, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            keys_to_update = set(new_settings.keys())
            
            for line in lines:
                found_key = None
                strip_line = line.strip()
                if strip_line and not strip_line.startswith((';', '#', '[')):
                    if '=' in line:
                        k = line.split('=')[0].strip().lower()
                        if k in keys_to_update:
                            found_key = k
                
                if found_key:
                    # Preserve indentation and trailing comments if possible
                    prefix = line.split('=')[0]
                    suffix = ""
                    if '#' in line: suffix = " #" + line.split('#', 1)[1].rstrip()
                    elif ';' in line: suffix = " ;" + line.split(';', 1)[1].rstrip()
                    
                    val = self.settings[found_key]
                    # Format float/bool correctly for INI
                    if isinstance(val, bool): val = str(val).lower()
                    
                    new_line = f"{prefix}= {val}{suffix}\n"
                    updated_lines.append(new_line)
                    keys_to_update.discard(found_key)
                else:
                    updated_lines.append(line)
            
            # If some keys were NOT found in the file, we append them at the end or in sections
            # (Better to just use ConfigParser for those, but for now let's keep it simple)
            # Append missing keys to their sections if they weren't found
            if keys_to_update:
                final_lines = []
                # Distribute missing keys into sections
                missing_by_section = {}
                for k in keys_to_update:
                    sec = SECTION_MAP.get(k, "OTHER")
                    if sec not in missing_by_section: missing_by_section[sec] = []
                    missing_by_section[sec].append(k)
                
                current_section = None
                processed_sections = set()
                
                for line in updated_lines:
                    final_lines.append(line)
                    if line.strip().startswith('[') and line.strip().endswith(']'):
                        current_section = line.strip()[1:-1]
                        # Don't add yet, add at the end of the section or before next section
                    
                    # If we are about to enter a new section or it's the end of file, dump missing keys for previous section
                    # Actually, simple way: if it's the start of a next section, dump the previous one's missing keys
                
                # RE-ALGORITHM: Rebuild from scratch but keep comments if found
                # Actually, let's just use the existing updated_lines and just append to the end for simplicity
                # OR properly insert.
                
                # Quick and dirty: Append missing sections at the end
                for sec, keys in missing_by_section.items():
                    # Check if section exists in file
                    section_exists = any(line.strip() == f"[{sec}]" for line in updated_lines)
                    if not section_exists:
                        updated_lines.append(f"\n[{sec}]\n")
                        for k in keys:
                            val = self.settings[k]
                            if isinstance(val, bool): val = str(val).lower()
                            updated_lines.append(f"{k}= {val}\n")
                    else:
                        # Append to existing section (find last line of section)
                        found_sec = False
                        insert_idx = len(updated_lines)
                        for idx, line in enumerate(updated_lines):
                            if line.strip() == f"[{sec}]":
                                found_sec = True
                            elif found_sec and line.strip().startswith('['):
                                insert_idx = idx
                                break
                        
                        for k in keys:
                            val = self.settings[k]
                            if isinstance(val, bool): val = str(val).lower()
                            updated_lines.insert(insert_idx, f"{k}= {val}\n")
                            insert_idx += 1

            with open(self.config_file, 'w') as f:
                f.writelines(updated_lines)
                
        except Exception as e:
            self.logger.error(f"Smart save failed, falling back to ConfigParser: {e}")
            # Fallback to standard save (loses comments)
            config = configparser.ConfigParser()
            for key, val in self.settings.items():
                section = SECTION_MAP.get(key, "OTHER")
                if section not in config: config.add_section(section)
                config.set(section, key, str(val))
            with open(self.config_file, 'w') as f:
                config.write(f)

    def get(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Update a setting, save it, and ensure internal state is fresh."""
        # Optional: Load latest from file before saving to prevent overwriting manual edits?
        # For now, just update internal dict and save.
        self.settings[key] = value
        self.save_settings({key: value})
