from typing import Optional, Dict, Any
import time
from .base_controller import BaseInstrumentController, InstrumentState

try:
    import pyvisa
except ImportError:
    pyvisa = None

class SMUController(BaseInstrumentController):
    """
    Controller for SCPI-compliant SMUs (Keysight B2900 series, Keithley 2400 series).
    
    Attributes:
        address (str): VISA resource address (e.g., 'USB0::0x0957::...::INSTR').
        resource (pyvisa.resources.MessageBasedResource): The actual VISA resource object.
    """

    def __init__(self, address: str, name: str = "SMU", mock: bool = False):
        super().__init__(name, mock)
        self.address = address
        self.resource = None
        self.rm = None
        
        # Cache for current settings to avoid unnecessary hardware queries
        self._current_source_amps = 0.0
        self._voltage_limit_volts = 2.0  # Safe default
        self._output_enabled = False
        
        # Software Safety Limit (None = Disabled)
        self.software_current_limit = None 

    def set_software_current_limit(self, limit: float = None):
        """
        Sets a software-level high-water mark for Current Source.
        Any attempt to set a current higher than this (abs) will raise ValueError.
        Set to None to disable.
        """
        self.software_current_limit = abs(limit) if limit is not None else None
        if self.software_current_limit is not None:
             self.logger.info(f"SAFETY: Software Current Limit set to {self.software_current_limit:.2e} A")
        else:
             self.logger.info("SAFETY: Software Current Limit DISABLED")

    def _check_current_limit(self, amps: float):
        """Internal helper to verify current against software limit."""
        if self.software_current_limit is not None:
            if abs(amps) > self.software_current_limit * 1.001: # 0.1% tolerance
                raise ValueError(f"SAFETY INTERLOCK: Requested {amps:.2e} A exceeds Software Limit {self.software_current_limit:.2e} A")

    def connect(self) -> None:
        """Connects to the instrument via PyVISA or mocks the connection."""
        if self.mock:
            self.logger.info(f"MOCK: Connected to SMU at {self.address}")
            self.to_state(InstrumentState.IDLE)
            return

        if pyvisa is None:
            self.handle_error("PyVISA not installed, cannot connect to real hardware.")
            return

        try:
            self.rm = pyvisa.ResourceManager()
            
            # Open with a timeout (default 20s for long lists)
            self.resource = self.rm.open_resource(self.address, open_timeout=20000)
            self.resource.timeout = 20000 # Set communication timeout to 20s
            
            # Attempt to clear the bus if possible (Fix for some lock states)
            try:
                self.resource.clear()
            except:
                pass

            # Basic SCPI identification to verify connection
            try:
                idn = self.resource.query("*IDN?")
            except Exception as e:
                # If query fails, try to clear again and retry once
                self.logger.warning(f"IDN query failed ({e}), retrying after clear...")
                self.resource.clear()
                time.sleep(0.5)
                idn = self.resource.query("*IDN?")

            self.logger.info(f"Connected to SMU: {idn.strip()}")
            
            # Reset to known state
            self.resource.write("*RST")
            self.resource.write("SOUR:FUNC:MODE CURR") # Current source mode
            
            self.to_state(InstrumentState.IDLE)
            
        except Exception as e:
            # Check specifically for NCIC which implies a locked interface
            msg = str(e)
            if "VI_ERROR_NCIC" in msg:
                 self.handle_error(f"SMU Locked (NCIC). Please Power-Cycle Instrument. Details: {msg}")
            else:
                 self.handle_error(f"Failed to connect to SMU: {msg}")

    def disconnect(self) -> None:
        """Safely disables output and closes connection."""
        # Attempt minimal shutdown
        if self._state not in [InstrumentState.OFF, InstrumentState.ERROR]:
            try:
                self.disable_output()
            except Exception:
                pass 

        # Force Close Resource
        if self.resource:
            try:
                self.resource.close()
                self.logger.info("Closed SMU VISA resource.")
            except Exception as e:
                self.logger.warning(f"Error closing resource: {e}")
        
        # Explicit cleanup
        self.resource = None
        if self.rm:
            try:
                self.rm.close()
            except:
                pass
            self.rm = None
            
        self.to_state(InstrumentState.OFF)

    def configure(self, settings: Dict[str, Any]) -> None:
        """
        Configures source settings.
        
        Expected keys:
            - compliance_voltage (float): Voltage limit in Volts.
            - current_range (float, optional): Measurement/Source range.
        """
        self.require_state([InstrumentState.IDLE, InstrumentState.CONFIGURED, InstrumentState.ARMED, InstrumentState.RUNNING])
        
        compliance = settings.get("compliance_voltage", 2.0)
        
        if self.mock:
            self.logger.info(f"MOCK: Configured SMU. Compliance={compliance}V")
            self._voltage_limit_volts = compliance
            self.to_state(InstrumentState.CONFIGURED)
            return

        try:
             # Set generic SCPI compliance (Voltage Protection)
            self.set_compliance(compliance, "VOLT")
            self.to_state(InstrumentState.CONFIGURED)
        except Exception as e:
            self.handle_error(f"Configuration failed: {e}")

    def set_compliance(self, limit: float, limit_type: str) -> None:
        """
        Sets compliance limit.
        Args:
            limit (float): value
            limit_type (str): 'VOLT' or 'CURR'
        """
        self.require_state([InstrumentState.IDLE, InstrumentState.CONFIGURED, InstrumentState.ARMED, InstrumentState.RUNNING])
        limit_type = limit_type.upper()
        if limit_type not in ['VOLT', 'CURR']:
            raise ValueError("Limit type must be VOLT or CURR")
            
        if self.mock:
            self.logger.info(f"MOCK: Set {limit_type} Compliance to {limit}")
            if limit_type == 'VOLT':
                self._voltage_limit_volts = limit
            return

        try:
             # SCPI: SENS:<Type>:PROT <Value>
            self.resource.write(f"SENS:{limit_type}:PROT {limit}")
        except Exception as e:
            self.handle_error(f"Failed to set compliance: {e}")

    def set_nplc(self, nplc: float) -> None:
        """
        Sets the measurement speed in Number of Power Line Cycles (NPLC).
        0.01 (Fast) to 100 (High Accuracy). Default is usually 1.0.
        """
        self.require_state([InstrumentState.IDLE, InstrumentState.CONFIGURED, InstrumentState.ARMED, InstrumentState.RUNNING])
        
        if self.mock:
            self.logger.info(f"MOCK: Set NPLC to {nplc}")
            return

        try:
            # Set for both voltage and current sensing
            self.resource.write(f"SENS:VOLT:NPLC {nplc}")
            self.resource.write(f"SENS:CURR:NPLC {nplc}")
        except Exception as e:
            self.handle_error(f"Failed to set NPLC: {e}")

    def set_current(self, amps: float) -> None:
        """Sets the DC source current immediately."""
        # Allow recovery from ERROR if we are just trying to zero the output
        self.require_state([InstrumentState.IDLE, InstrumentState.CONFIGURED, InstrumentState.ARMED, InstrumentState.RUNNING, InstrumentState.ERROR])
        
        if self.mock:
            self._check_current_limit(amps)
            self.logger.info(f"MOCK: Set Current {amps} A")
            self._current_source_amps = amps
            return

        try:
            self._check_current_limit(amps)
            self.resource.write(f"SOUR:FUNC:MODE CURR") 
            self.resource.write(f"SOUR:CURR {amps}")
            self._current_source_amps = amps
            # If we succeed, we are at least IDLE or CONFIGURED. If we were ERROR, we might be OK now?
            if self.state == InstrumentState.ERROR:
                self.to_state(InstrumentState.IDLE)
        except Exception as e:
            self.handle_error(f"Failed to set current: {e}")

    def set_source_mode(self, mode: str) -> None:
        """
        Sets the source mode: 'VOLT' or 'CURR'.
        Args:
            mode (str): 'VOLT' for Voltage Source, 'CURR' for Current Source.
        """
        self.require_state([InstrumentState.IDLE, InstrumentState.CONFIGURED, InstrumentState.ARMED, InstrumentState.RUNNING])
        
        mode = mode.upper()
        if mode not in ['VOLT', 'CURR']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'VOLT' or 'CURR'.")

        if self.mock:
            self.logger.info(f"MOCK: Set Source Mode to {mode}")
            return

        try:
            self.resource.write(f"SOUR:FUNC:MODE {mode}")
            self.logger.info(f"Set Source Mode to {mode}")
        except Exception as e:
            self.handle_error(f"Failed to set source mode: {e}")

    def set_voltage(self, volts: float) -> None:
        """Sets the DC source voltage immediately."""
        self.require_state([InstrumentState.IDLE, InstrumentState.CONFIGURED, InstrumentState.ARMED, InstrumentState.RUNNING])

        if self.mock:
            self.logger.info(f"MOCK: Setting voltage to {volts} V")
            return

        try:
            self.resource.write(f"SOUR:VOLT {volts}")
        except Exception as e:
            self.handle_error(f"Failed to set voltage: {e}")


    def enable_output(self) -> None:
        """Turns the SMU output ON."""
        # IDLE allowed to just turn on. ERROR allowed for force recovery attempts.
        self.require_state([InstrumentState.CONFIGURED, InstrumentState.ARMED, InstrumentState.IDLE, InstrumentState.ERROR]) 

        if self.mock:
            self.logger.info("MOCK: Output ENABLED")
            self._output_enabled = True
            self.to_state(InstrumentState.RUNNING)
            return

        try:
            self.resource.write("OUTP ON")
            self._output_enabled = True
            # Double check
            if not self.mock and "1" not in self.resource.query("OUTP?"):
                 self.logger.warning("SMU did not report Output ON after command!")
            self.to_state(InstrumentState.RUNNING)
        except Exception as e:
            self.handle_error(f"Failed to enable output: {e}")

    def disable_output(self) -> None:
        """Turns the SMU output OFF."""
        # Allowed from almost any state except OFF
        # Also allowed from ERROR to try and safe the device
        
        if self.mock:
             self.logger.info("MOCK: Output DISABLED")
             self._output_enabled = False
             self.to_state(InstrumentState.IDLE)
             return

        try:
            # Try to abort first if we are stuck in a trigger wait
            try: self.resource.write("ABOR") 
            except: pass
            
            self.resource.write("OUTP OFF")
            self._output_enabled = False
            self.to_state(InstrumentState.IDLE)
        except Exception as e:
            # If we fail to disable, we are in big trouble.
            # Try a low level interface clear if it's a timeout
            if "VI_ERROR_TMO" in str(e) or "Timeout" in str(e):
                self.logger.critical("VISA Timeout during disable. Attempting Interface Clear.")
                try:
                    self.resource.clear() # VISA `viClear`
                except:
                    pass
            self.handle_error(f"Failed to disable output: {e}")


    def setup_list_sweep(self, points: list[float], source_mode: str, time_per_step: float, trigger_count: int = 1) -> None:
        """
        Configures a List Sweep (Arbitrary Waveform).
        
        Args:
            points: List of values (Volts or Amps).
            source_mode: 'VOLT' or 'CURR'.
            time_per_step: Duration of each point in seconds.
            trigger_count: Number of times to repeat the list (or length of list logic). 
                           Usually 1 sweep through the list.
        """
        self.require_state([InstrumentState.IDLE, InstrumentState.CONFIGURED, InstrumentState.RUNNING, InstrumentState.ERROR])
        
        source_mode = source_mode.upper()
        if source_mode not in ['VOLT', 'CURR']:
             raise ValueError("Mode must be VOLT or CURR")
             
        # Validation
        if len(points) == 0:
            raise ValueError("List points cannot be empty")
            
        if source_mode == 'CURR':
            for p in points:
                self._check_current_limit(p)
            
        # Convert list to string
        points_str = ",".join([f"{x:.6e}" for x in points])
        
        if self.mock:
            self.logger.info(f"MOCK: Configured List Sweep ({source_mode}). Points={len(points)}, Step={time_per_step}s")
            self.to_state(InstrumentState.ARMED)
            return

        try:
             # 1. Select Function
            self.resource.write(f"SOUR:FUNC:MODE {source_mode}")
            
            # 2. Set Mode to LIST
            self.resource.write(f"SOUR:{source_mode}:MODE LIST")
            
            # --- CLEAR PREVIOUS LIST & TRACE ---
            # Ideally we clear the source memory, but B2900 overwrites usually.
            # But let's clear the trace buffer so we don't carry over old trigger events?
            self.resource.write("TRAC:CLE")
            
            # 3. Upload Points (B2900 limit is 2500-100k depending on model check manual if list is huge)
            self.resource.write(f"SOUR:LIST:{source_mode} {points_str}")
            
            # 4. Timing Config (Timer Trigger)
            # Source Trigger: Timer
            self.resource.write("TRIG:TRAN:SOUR TIM")
            self.resource.write(f"TRIG:TRAN:TIM {time_per_step}")
            
            # 5. Length of sequence
            self.resource.write(f"TRIG:TRAN:COUN {len(points)}")
            
            # 6. Repetitions
            self.resource.write(f"ARM:TRAN:COUN {trigger_count}")
            
            # Ensure Trigger Timer is valid by setting Source shape? 
            # Default is continuous via list, so SOUR:FUNC:SHAP PULS might be wrong if we use list.
            # List mode usually overrides shape.
            
            self.to_state(InstrumentState.ARMED)
            self.logger.info("SMU Armed for List Sweep")
            
        except Exception as e:
            self.handle_error(f"Failed to setup list sweep: {e}")

    def generate_square_wave(self, high_level: float, low_level: float, period: float, duty_cycle: float, total_cycles: int, mode: str = "CURR"):
        """
        Generates a square wave using List Sweep.
        
        Args:
            high_level: Value during ON phase
            low_level: Value during OFF phase
            period: Total period in seconds
            duty_cycle: 0.0 to 1.0 (e.g. 0.5 = 50%)
            total_cycles: Number of full periods to generate
            mode: 'CURR' or 'VOLT'
        """
        if not (0 < duty_cycle < 1):
             raise ValueError("Duty cycle must be between 0 and 1")
             
        if mode == 'CURR':
            self._check_current_limit(high_level)
            self._check_current_limit(low_level)
             
        # Calculate time steps
        t_high = period * duty_cycle
        t_low = period * (1 - duty_cycle)
        
        # B2900 List Sweep supports constant interval per step typically (TRIG:TRAN:TIM is global).
        # To achieve variable duty cycle with fixed step, we might need multiple points.
        # OR: We see if B2900 supports per-step delay. (SOUR:LIST:DEL ?)
        # A simpler robust way for "Any Duty Cycle" with fixed generic timer: 
        # Create a finer grid (e.g. 100 points per cycle).
        
        # However, for 10ms pulses, 100 points is 0.1ms (ok for SMU).
        # Let's try to find a Common Denominator or use 2 points if 50% duty.
        
        # Strategy: Use 50 points per cycle to give 2% duty cycle resolution.
        res = 50
        on_points = int(res * duty_cycle)
        off_points = res - on_points
        
        cycle_points = [high_level] * on_points + [low_level] * off_points
        full_list = cycle_points # We can just loop this list using ARM count?
        
        # Time per step
        dt = period / res
        
        self.logger.info(f"Generating Square Wave: {on_points} High / {off_points} Low steps. dt={dt*1000:.2f}ms")
        
        # We set trigger count to 1 (run list once) but ARM count to total_cycles
        # Wait, setup_list_sweep uses ARM for repetitions.
        
        self.setup_list_sweep(full_list, mode, time_per_step=dt, trigger_count=total_cycles)

    def trigger_list(self):
        """Starts the configured list sweep."""
        # We allow ARMED (normal flow) or RUNNING (if output was just enabled manually or via sequence)
        # Also allowed from ERROR to attempt retry/recovery commands
        self.require_state([InstrumentState.ARMED, InstrumentState.RUNNING, InstrumentState.ERROR])
        
        if self.mock:
            self.logger.info("MOCK: Trigger List Sequence")
            self.to_state(InstrumentState.RUNNING)
            return
            
        try:
            # Use *WAI to ensure any previous setup commands are fully processed
            self.resource.write("*WAI")
            self.resource.write("INIT (@1)") # Init channel 1
            self.to_state(InstrumentState.RUNNING)
        except Exception as e:
            self.handle_error(f"Trigger failed: {e}")

    def setup_pulse(self, high_amps: float, low_amps: float, pulse_width: float, period: float):
        """
        Configures a pulse train.
        Note: Precise pulsing often requires specific model-dependent commands (List sweep vs Pulse mode).
        This implementation assumes a Keysight B2900 style list sweep for broad compatibility or Mocking.
        """
        self.require_state([InstrumentState.IDLE, InstrumentState.CONFIGURED, InstrumentState.RUNNING])
        
        if self.mock:
            self.logger.info(f"MOCK: Configured Pulse. High={high_amps}, Low={low_amps}, Width={pulse_width}, Period={period}")
            return
            
        # TODO: Implement B2900 specific pulse list SCPI commands here.
        # This is complex and depends heavily on the exact model.
        # For now, we will log a warning that non-mock pulse is semi-implemented.
        self.logger.warning("Real hardware pulse generation requires specific model implementation. Using DC fallback.")

    def measure(self) -> Dict[str, float]:
        """
        Performs a spot measurement of Voltage and Current.
        Returns:
            dict: {'voltage': float, 'current': float}
        """
        self.require_state([InstrumentState.RUNNING, InstrumentState.ARMED, InstrumentState.CONFIGURED, InstrumentState.IDLE])

        
        if self.mock:
            # Return plausible values based on settings
            import random
            noise = random.gauss(0, 1e-9)
            v_meas = self._voltage_limit_volts * 0.1 # Dummy value
            i_meas = self._current_source_amps + noise
            self.logger.info(f"MOCK: Measured V={v_meas:.4f}, I={i_meas:.4e}")
            return {'voltage': v_meas, 'current': i_meas}

        try:
            # Measure both (B2900 supports generic SCPI MEAS?)
            # Usually MEAS:VOLT? and MEAS:CURR? works. 
            # Or MEAS:VOLT,CURR? for both?
            # Safe approach: Read individually or use :MEAS? if supported.
            # B2900: :MEAS:VOLT? returns voltage. :MEAS:CURR? returns current.
            
            v_str = self.resource.query("MEAS:VOLT?").strip()
            i_str = self.resource.query("MEAS:CURR?").strip()
            
            result = {
                'voltage': float(v_str),
                'current': float(i_str)
            }
            return result
        except Exception as e:
            self.handle_error(f"Measurement failed: {e}")
            return {'voltage': 0.0, 'current': 0.0}
