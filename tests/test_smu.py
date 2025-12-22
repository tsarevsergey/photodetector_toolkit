import pytest
import logging
from src.hardware.smu_controller import SMUController, InstrumentState

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)

def test_smu_mock_lifecycle():
    """Verifies the full lifecycle of the SMU Controller in Mock mode."""
    
    # 1. Initialize
    smu = SMUController(address="USB::MOCK", mock=True)
    assert smu.state == InstrumentState.OFF
    
    # 2. Connect
    smu.connect()
    assert smu.state == InstrumentState.IDLE
    
    # 3. Configure
    smu.configure({"compliance_voltage": 2.5})
    assert smu._voltage_limit_volts == 2.5
    assert smu.state == InstrumentState.CONFIGURED
    
    # 4. Set Current (valid state)
    smu.set_current(0.01) # 10mA
    assert smu._current_source_amps == 0.01
    
    # 5. Enable Output
    smu.enable_output()
    assert smu.state == InstrumentState.RUNNING
    assert smu._output_enabled is True
    
    # 6. Disable Output
    smu.disable_output()
    assert smu.state == InstrumentState.IDLE
    assert smu._output_enabled is False
    
    # 7. Disconnect
    smu.disconnect()
    assert smu.state == InstrumentState.OFF

def test_smu_safety_checks():
    """Verifies that operations are rejected in invalid states."""
    smu = SMUController(address="USB::MOCK", mock=True)
    
    # Cannot configure while OFF
    with pytest.raises(RuntimeError):
        smu.configure({"compliance_voltage": 1.0})
        
    smu.connect()
    # Cannot enable output while IDLE (needs config? Actually my code allows IDLE->RUNNING but check implementation)
    # The code says: self.require_state([InstrumentState.CONFIGURED, InstrumentState.ARMED, InstrumentState.IDLE])
    # So enabling from IDLE is allowed (assuming defaults).
    
    smu.enable_output()
    assert smu.state == InstrumentState.RUNNING
    
    # Cannot disconnect without disabling?
    # Disconnect implementation says it attempts to disable output.
    smu.disconnect() 
    assert smu.state == InstrumentState.OFF
