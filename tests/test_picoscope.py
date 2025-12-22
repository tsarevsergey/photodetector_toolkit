import pytest
import logging
import numpy as np
from src.hardware.picoscope_controller import PicoScopeController, InstrumentState

logging.basicConfig(level=logging.DEBUG)

def test_picoscope_mock_acquisition():
    """Verifies mock acquisition flow."""
    scope = PicoScopeController(mock=True)
    
    # Connect
    scope.connect()
    assert scope.state == InstrumentState.IDLE
    
    # Configure
    scope.configure({"samples": 500})
    assert scope.post_trigger_samples == 500
    assert scope.state == InstrumentState.CONFIGURED
    
    # Acquire
    t, v = scope.acquire()
    
    assert len(t) == 500
    assert len(v) == 500
    assert isinstance(t, np.ndarray)
    assert isinstance(v, np.ndarray)
    
    # State should return to IDLE (or previous) after blocking acquire
    # In my implementation, it goes IDLE -> RUNNING -> IDLE
    # But wait, acquire() returns values, so it must be finished.
    # The method explicitly sets self.to_state(InstrumentState.IDLE) at end.
    
    # Disconnect
    scope.disconnect()
    assert scope.state == InstrumentState.OFF
