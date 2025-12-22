
import sys
import os
import logging
import time

# Add src to path so we can import hardware
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hardware.smu_controller import SMUController, InstrumentState

# Setup logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ADDRESS = "USB0::0x0957::0xCD18::MY51143841::0::INSTR"

def test_hardware():
    logger.info(f"Attempting to connect to SMU at {ADDRESS}")
    
    smu = SMUController(address=ADDRESS, mock=False)
    
    try:
        logger.info("Connecting...")
        smu.connect()
        
        if smu.state == InstrumentState.OFF:
            logger.error("Failed to connect (State is still OFF)")
            return

        logger.info("Successfully connected!")
        
        # Configure
        logger.info("Configuring compliance voltage to 2.0V...")
        smu.configure({"compliance_voltage": 2.0})
        
        # Set Current to 1uA (safe low value)
        logger.info("Setting current to 1e-6 A...")
        smu.set_current(1e-6)
        
        # Enable Output
        logger.info("Enabling Output...")
        smu.enable_output()
        time.sleep(1) # Settling time
        
        # Measure
        logger.info("Measuring...")
        meas = smu.measure()
        logger.info(f"Measurement Result: {meas}")
        
        if abs(meas['current']) < 1e-9:
             logger.warning("Current reading is suspiciously low!")

        
        # Disable Output
        logger.info("Disabling Output...")
        smu.disable_output()
        
        logger.info("Test Sequence Complete.")
        
    except Exception as e:
        logger.error(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        logger.info("Disconnecting...")
        smu.disconnect()

if __name__ == "__main__":
    test_hardware()
