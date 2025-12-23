from typing import Dict, Optional, Any, Tuple
import numpy as np
import time

try:
    from picosdk.ps2000a import ps2000a as ps
    from picosdk.functions import adc2mV, assert_pico_ok
    import ctypes
except ImportError:
    ps = None

from .base_controller import BaseInstrumentController, InstrumentState

class ScopeController(BaseInstrumentController):
    """
    Controller for PicoScope 2000 Series (ps2000a driver for 2208B).
    """
    def __init__(self, name="Scope", mock=False):
        super().__init__(name, mock)
        self.chandle = ctypes.c_int16()
        self.status = {}
        self.timebase = 8 # Default timebase index
        self.sample_interval_ns = 0
        self.max_samples = 10000 
        self._streaming = False
        
        # Buffer storage
        self.buffer_a = None
        self.buffer_b = None

    def __del__(self):
        """Destructor to ensure handle is closed if object is garbage collected."""
        self.disconnect()

    def connect(self):
        if self.mock:
            self.logger.info("MOCK: Connected to PicoScope 2208B")
            self.to_state(InstrumentState.IDLE)
            return

        if ps is None:
            self.handle_error("PicoSDK not installed.")
            return

        # Check for conflicting software
        try:
             import subprocess
             output = subprocess.check_output("tasklist", shell=True).decode()
             if "PicoScope.exe" in output or "PicoScope7.exe" in output:
                 self.logger.warning("PicoScope software detected running. This may block connection!")
                 st.warning("⚠️ 'PicoScope' software is running in the background. Please close it completely.")
        except:
             pass

        try:
            # Open Unit
            self.status["openunit"] = ps.ps2000aOpenUnit(ctypes.byref(self.chandle), None)
            
            # Check specifically for PICO_NOT_FOUND (3)
            # assert_pico_ok raises generic error if != 0
            if self.status["openunit"] == 3:
                self.logger.warning("PICO_NOT_FOUND detected. Attempting brute-force driver reset...")
                self.force_close_all()
                time.sleep(1.0)
                # Retry
                self.chandle = ctypes.c_int16()
                self.status["openunit"] = ps.ps2000aOpenUnit(ctypes.byref(self.chandle), None)
            
            assert_pico_ok(self.status["openunit"])
            
            self.logger.info(f"Connected to PicoScope Handle {self.chandle.value}")
            self.to_state(InstrumentState.IDLE)
            
            # Default Channel Setup (A=ON, B=OFF)
            # Use 10V range to safely cover LDR sweeps (potentially up to 8-9V signal)
            self.configure_channel('A', True, "10V")
            self.configure_channel('B', False)
            
        except Exception as e:
            # If it failed, ensure we strip state
            self.chandle = None
            self.handle_error(f"Failed to open scope: {e}")

    def reset_connection(self):
        """Closes and re-opens connection."""
        self.disconnect()
        self.connect()

    def configure_channel(self, channel_name: str, enabled: bool, range_str: str = "2V", coupling: str = "DC"):
        """
        Configures a channel.
        Args:
            channel_name: 'A' or 'B'
            enabled: True/False
            range_str: '10mv', '20mv', '50mv', '100mv', '200mv', '500mv', '1V', '2V', '5V', '10V', '20V'
            coupling: 'DC' or 'AC'
        """
        self.require_state([InstrumentState.IDLE, InstrumentState.CONFIGURED])
        
        if self.mock:
            self.logger.info(f"MOCK: Config Channel {channel_name} {enabled} {range_str}")
            return

        ch_map = {'A': ps.PS2000A_CHANNEL['PS2000A_CHANNEL_A'], 'B': ps.PS2000A_CHANNEL['PS2000A_CHANNEL_B']}
        
        # Valid ranges map
        ranges = {
            '10ML': ps.PS2000A_RANGE['PS2000A_10MV'],
            '20MV': ps.PS2000A_RANGE['PS2000A_20MV'],
            '50MV': ps.PS2000A_RANGE['PS2000A_50MV'],
            '100MV': ps.PS2000A_RANGE['PS2000A_100MV'],
            '200MV': ps.PS2000A_RANGE['PS2000A_200MV'],
            '500MV': ps.PS2000A_RANGE['PS2000A_500MV'],
            '1V': ps.PS2000A_RANGE['PS2000A_1V'],
            '2V': ps.PS2000A_RANGE['PS2000A_2V'],
            '5V': ps.PS2000A_RANGE['PS2000A_5V'],
            '10V': ps.PS2000A_RANGE['PS2000A_10V'],
            '20V': ps.PS2000A_RANGE['PS2000A_20V'],
        }
        
        range_enum = ranges.get(range_str.upper(), ps.PS2000A_RANGE['PS2000A_2V'])
        coupling_enum = ps.PS2000A_COUPLING['PS2000A_DC'] if coupling.upper() == 'DC' else ps.PS2000A_COUPLING['PS2000A_AC']
        
        try:
            self.status[f"setCh{channel_name}"] = ps.ps2000aSetChannel(
                self.chandle,
                ch_map[channel_name],
                1 if enabled else 0,
                coupling_enum,
                range_enum,
                0 # Analog offset
            )
            assert_pico_ok(self.status[f"setCh{channel_name}"])
            
            # Store state for ADC conversion
            if channel_name == 'A':
                self.current_range_enum = range_enum
                
            self.to_state(InstrumentState.CONFIGURED)
        except Exception as e:
            self.handle_error(f"Error configuring Channel {channel_name}: {e}")

    def capture_block(self, timebase_index: int = 8, samples: int = 2000):
        """
        Captures a single block of data.
        """
        self.require_state([InstrumentState.IDLE, InstrumentState.CONFIGURED])
        
        if self.mock:
            self.logger.info("MOCK: Capture Block")
            time.sleep(0.1)
            t = np.linspace(0, 0.01, samples)
            v = np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.05, samples)
            return t, v

        try:
            # 0. Stop any existing capture
            ps.ps2000aStop(self.chandle)
            
            # --- Setup Buffers (mimic test.py pattern) ---
            # Create buffers
            bufferAMax = (ctypes.c_int16 * samples)()
            bufferAMin = (ctypes.c_int16 * samples)() # Not used but required for call
            
            # Set Data Buffers (Plural)
            # Channel A = 0
            self.status["setDataBuffersA"] = ps.ps2000aSetDataBuffers(
                self.chandle,
                0,
                ctypes.byref(bufferAMax),
                ctypes.byref(bufferAMin),
                samples,
                0,
                0 # Ratio mode none
            )
            assert_pico_ok(self.status["setDataBuffersA"])

            # 1. Get Timebase
            timeIntervalns = ctypes.c_float()
            returnedMaxSamples = ctypes.c_int32()
            oversample = 0 # As per test.py
            
            self.status["getTimebase2"] = ps.ps2000aGetTimebase2(self.chandle, timebase_index, samples, ctypes.byref(timeIntervalns), oversample, ctypes.byref(returnedMaxSamples), 0)
            assert_pico_ok(self.status["getTimebase2"])

            # 2. Run Block
            # Args: handle, nPre, nPost, timebase, oversample, timeIndisposedMs, segmentIndex, lpReady, pParameter
            timeIndisposedMs = ctypes.c_int32()
            self.status["runBlock"] = ps.ps2000aRunBlock(
                self.chandle,
                0,          # Pre-trigger samples
                samples,    # Post-trigger samples
                timebase_index,
                oversample,
                ctypes.byref(timeIndisposedMs),
                0, None, None
            )
            assert_pico_ok(self.status["runBlock"])
            self.logger.info("RunBlock started.")
            
            # 3. Wait for ready (Timeout 5s)
            ready = ctypes.c_int16(0)
            start_wait = time.time()
            
            while ready.value == 0:
                self.status["isReady"] = ps.ps2000aIsReady(self.chandle, ctypes.byref(ready))
                if time.time() - start_wait > 5.0:
                    raise TimeoutError("Scope capture timed out (IsReady never returned true)")
                time.sleep(0.01)
                
            # 4. Get Data
            overflow = ctypes.c_int16()
            cmaxSamples = ctypes.c_int32(samples)
            
            self.status["getValues"] = ps.ps2000aGetValues(self.chandle, 0, ctypes.byref(cmaxSamples), 0, 0, 0, ctypes.byref(overflow))
            assert_pico_ok(self.status["getValues"])
            
            # 5. Conversion
            # Get Max ADC
            maxADC = ctypes.c_int16()
            self.status["maximumValue"] = ps.ps2000aMaximumValue(self.chandle, ctypes.byref(maxADC))
            assert_pico_ok(self.status["maximumValue"])
            
            # Convert to mV using helper
            # Use stored range from configuration
            current_range = self.current_range_enum if hasattr(self, 'current_range_enum') else ps.PS2000A_RANGE['PS2000A_10V']
            
            # Use picosdk.functions.adc2mV
            raw_data_mv = adc2mV(bufferAMax, current_range, maxADC)
            
            # Convert to Volts
            volts = np.array(raw_data_mv) / 1000.0
            
            # Time axis
            dt = timeIntervalns.value * 1e-9
            times = np.linspace(0, dt * samples, samples)
             
            return times, volts

        except Exception as e:
            self.handle_error(f"Capture failed: {e}")
            raise e
        finally:
            if ps and self.chandle:
                try: 
                    ps.ps2000aStop(self.chandle)
                except: 
                    pass

    def configure(self, settings: Dict[str, Any]) -> None:
        """
        Generic configuration entry point.
        Currently just logs/placeholders for specific internal configs.
        """
        # Can route to configure_channel here if needed
        self.to_state(InstrumentState.CONFIGURED)

    def disconnect(self):
        if self.chandle:
            try:
                ps.ps2000aCloseUnit(self.chandle)
            except:
                pass
        self.chandle = None
        self.to_state(InstrumentState.OFF)

    def calculate_timebase_index(self, duration_s: float, samples: int) -> int:
        """
        Calculates the required timebase index for a desired duration and sample count.
        Formula for PS2000A (TB >= 3):
            Interval (s) = (TB - 2) / 62,500,000
            Total Time = Interval * Samples
            
            => Interval = Duration / Samples
            => (TB - 2) / 62.5e6 = Duration / Samples
            => TB - 2 = (Duration / Samples) * 62.5e6
            => TB = ((Duration * 62.5e6) / Samples) + 2
        """
        if duration_s <= 0 or samples <= 0:
            return 8 # safe default
            
        target_interval = duration_s / samples
        
        # Initial estimate assuming TB >= 3
        tb = int((target_interval * 62_500_000) + 2)
        
        # Clamp to valid range (driver dependent, usually 2^32 max, but practically limited)
        if tb < 3: tb = 3 # We ignore the super fast modes 0-2 for general physics use
        
        return tb

    @staticmethod
    def force_close_all():
        """
        Attempts to close potentially stuck handles.
        Checks specific known 'magic' handles and standard range.
        """
        if ps is None: return
        
        # Range of potential handles
        # ps2000a driver usually issues small int handles, but we check more.
        # Also check 16384 as per some support forum suggestions for 'stuck' units.
        targets = list(range(100)) + [16384]
        
        for i in targets:
            try:
                h = ctypes.c_int16(i)
                ps.ps2000aCloseUnit(h)
            except:
                pass

    def capture_streaming(self, duration_s: float, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Captures data in Streaming Mode.
        Args:
            duration_s: Total time to record.
            sample_rate: Desired samples per second.
        Returns:
            times, volts
        """
        self.require_state([InstrumentState.IDLE, InstrumentState.CONFIGURED])
        
        needed_samples = int(duration_s * sample_rate)
        
        if self.mock:
            self.logger.info(f"MOCK: Capture Streaming {duration_s}s @ {sample_rate}Hz")
            time.sleep(min(duration_s, 2.0)) # Don't sleep full time in mock if long
            t = np.linspace(0, duration_s, needed_samples)
            v = 0.5 * np.sin(2 * np.pi * 80 * t) + np.random.normal(0, 0.01, needed_samples)
            return t, v

        try:
            self.to_state(InstrumentState.RUNNING)
            # 0. Stop keys
            ps.ps2000aStop(self.chandle)
            
            # 1. Setup Buffers for Overview (Driver requires these even if we copy out)
            # We use a small buffer for the driver ring buffer
            overview_buffer_size = 10000 
            bufferAMax = (ctypes.c_int16 * overview_buffer_size)()
            bufferAMin = (ctypes.c_int16 * overview_buffer_size)()
            
            self.status["setDataBuffersA"] = ps.ps2000aSetDataBuffers(
                self.chandle,
                0, # ChA
                ctypes.byref(bufferAMax),
                ctypes.byref(bufferAMin),
                overview_buffer_size,
                0, # Seg index
                0 # Ratio mode none
            )
            assert_pico_ok(self.status["setDataBuffersA"])
            
            # 2. Configure Streaming
            # Sample Interval
            # PS2000A_US = 2, PS2000A_NS = 1.
            # We want sample_rate (Hz). interval = 1e9 / sample_rate (ns).
            # But specific drivers have limits.
            # Let's use US (microseconds) for stability if rate < 1MS/s
            
            if sample_rate > 1_000_000:
                # Use NS
                sample_interval = int(1e9 / sample_rate)
                time_units = ps.PS2000A_TIME_UNITS['PS2000A_NS']
            else:
                # Use US
                sample_interval = int(1e6 / sample_rate)
                time_units = ps.PS2000A_TIME_UNITS['PS2000A_US']
                
            if sample_interval < 1: sample_interval = 1
            
            sampleInterval = ctypes.c_int32(sample_interval)
            
            # We need a big application buffer to copy data into
            app_buffer = np.zeros(needed_samples, dtype=np.int16)
            samples_collected = 0
            
            # 3. Define Callback (Closure style to capture locals? No, ctypes callbacks need to be global or standard)
            # Actually, we can use a class method if wrapped, but usually a simple function is safer.
            # We will use the polling loop approach with GetStreamingLatestValues which is easier in Python than C-callbacks.
            # Wait, ps2000aGetStreamingLatestValues REQUIRES a callback.
            
            # We define callback here
            def streaming_callback(handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):
                nonlocal samples_collected
                if noOfSamples > 0:
                     # Copy data
                     # Where is it coming from? It comes from the buffers passed to SetDataBuffers.
                     # The driver fills 'bufferAMax' circularly.
                     # startIndex tells us where in bufferAMax the new data is.
                     
                     # Indices in circular buffer
                     src_start = startIndex
                     src_end = startIndex + noOfSamples
                     
                     # Indices in target app_buffer
                     dest_start = samples_collected
                     dest_end = samples_collected + noOfSamples
                     
                     # Safety check bounds
                     if dest_end > needed_samples:
                         noOfSamples = needed_samples - samples_collected
                         dest_end = needed_samples
                         src_end = startIndex + noOfSamples
                     
                     if noOfSamples > 0:
                         # Handle wrap around of driver buffer? 
                         # The SDK usually assures linear blocks in callback? 
                         # "The driver copies the data... to the overview buffers..."
                         # If bufferAMax size is sufficient for each callback chunk, we are good.
                         # We set overview 10000, usually valid.
                         
                         # Copy
                         # We need to access the c-array as numpy or list
                         # bufferAMax is c_int16_Array.
                         # Slicing c_array works
                         chunk = bufferAMax[src_start:src_end]
                         app_buffer[dest_start:dest_end] = chunk
                         
                         samples_collected += noOfSamples
                         
            # Convert to CType
            c_callback = ps.StreamingReadyType(streaming_callback)
            
            # 4. Start Streaming
            maxPreTrigger = 0
            autoStop = 1 # We manage stop manually or via count? 
            # If we set autoStop=1 and pass needed_samples as maxPostTrigger, driver stops automatically.
            
            self.status["runStreaming"] = ps.ps2000aRunStreaming(
                self.chandle,
                ctypes.byref(sampleInterval),
                time_units,
                maxPreTrigger, 
                needed_samples, # maxPostTrigger
                1, # autoStop
                1, # downSampleRatio
                ps.PS2000A_RATIO_MODE['PS2000A_RATIO_MODE_NONE'],
                overview_buffer_size
            )
            assert_pico_ok(self.status["runStreaming"])
            
            actual_sample_interval = sampleInterval.value
            actual_rate = (1e9 if time_units == ps.PS2000A_TIME_UNITS['PS2000A_NS'] else 1e6) / actual_sample_interval
            self.logger.info(f"Streaming started: {needed_samples} samples @ ~{actual_rate:.1f}Hz")
            
            # 5. Collection Loop
            start_time = time.time()
            while samples_collected < needed_samples:
                # Poll
                ps.ps2000aGetStreamingLatestValues(self.chandle, c_callback, None)
                
                if time.time() - start_time > (duration_s + 5.0):
                    raise TimeoutError("Streaming capture timed out.")
                
                time.sleep(0.01)
                
            # 6. Stop
            self.status["stop"] = ps.ps2000aStop(self.chandle)
            self.to_state(InstrumentState.IDLE)
            
            # 7. Convert
            # Use stored range
            current_range = self.current_range_enum if hasattr(self, 'current_range_enum') else ps.PS2000A_RANGE['PS2000A_10V']
            
            # Helper: We have int16 array. Need MaxADC.
            maxADC = ctypes.c_int16()
            ps.ps2000aMaximumValue(self.chandle, ctypes.byref(maxADC))
            
            # Convert to Volts
            # adc2mV expects c_array, but we have numpy int16.
            # Optimised conversion:
            # V = (raw / maxADC) * range_mv
            # Ranges are enumerations, need to map numeric value.
            
            # Reverse map enum to mV
            # 10V=10000mV. 
            # We need a robust map.
            range_mv_map = {
                ps.PS2000A_RANGE['PS2000A_10MV']: 10,
                ps.PS2000A_RANGE['PS2000A_20MV']: 20,
                ps.PS2000A_RANGE['PS2000A_50MV']: 50,
                ps.PS2000A_RANGE['PS2000A_100MV']: 100,
                ps.PS2000A_RANGE['PS2000A_200MV']: 200,
                ps.PS2000A_RANGE['PS2000A_500MV']: 500,
                ps.PS2000A_RANGE['PS2000A_1V']: 1000,
                ps.PS2000A_RANGE['PS2000A_2V']: 2000,
                ps.PS2000A_RANGE['PS2000A_5V']: 5000,
                ps.PS2000A_RANGE['PS2000A_10V']: 10000,
                ps.PS2000A_RANGE['PS2000A_20V']: 20000,
            }
            v_range_mv = range_mv_map.get(current_range, 10000)
            
            # Numpy calc (faster than iterating adc2mV)
            volts = (app_buffer.astype(np.float32) / maxADC.value) * (v_range_mv / 1000.0)
            
            # Time axis
            # Total duration = actual_sample_interval * samples?
            # actual_sample_interval is in units (e.g. us).
            dt = 1.0 / actual_rate
            times = np.linspace(0, dt * samples_collected, samples_collected)
            
            return times, volts
            
        except Exception as e:
            self.handle_error(f"Streaming failed: {e}")
            # Ensure we sort of cleanup
            try: ps.ps2000aStop(self.chandle)
            except: pass
            raise e
        finally:
            if self.state == InstrumentState.RUNNING:
                self.to_state(InstrumentState.IDLE)

    def reset_device(self):
        """Attempts to soft-reset the scope state."""
        try:
             ps.ps2000aStop(self.chandle)
        except: pass
        self.to_state(InstrumentState.IDLE)

