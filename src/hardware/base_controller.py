from abc import ABC, abstractmethod
import logging
from enum import Enum
from typing import Optional, Dict, Any, List, Union

class InstrumentState(Enum):
    """
    Standard states for hardware instruments.
    """
    OFF = "OFF"         # Not connected or powered down
    IDLE = "IDLE"        # Connected, ready for config
    CONFIGURED = "CONFIGURED" # Parameters set, ready to arm/run
    ARMED = "ARMED"       # Ready for trigger (specific to scopes/acquisition)
    RUNNING = "RUNNING"     # Actively acquiring or outputting
    ERROR = "ERROR"       # Fault state

class BaseInstrumentController(ABC):
    """
    Abstract base class for all hardware instrument controllers.
    Enforces the MCP (Model-Controller-Peripheral) pattern requirements:
    - Finite state machine
    - Mock mode support
    - Logging
    - Safety checks
    """

    def __init__(self, name: str, mock: bool = False):
        self.name = name
        self.mock = mock
        self._state = InstrumentState.OFF
        self.logger = logging.getLogger(f"instrument.{name}")
        self.config: Dict[str, Any] = {}
        
        # Ensure logging is set up
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.logger.info(f"Initialized {self.name} controller (Mock Mode: {self.mock})")

    @property
    def state(self) -> InstrumentState:
        return self._state

    def to_state(self, new_state: InstrumentState):
        """Safe state transition with logging."""
        self.logger.info(f"State transition: {self._state.value} -> {new_state.value}")
        self._state = new_state

    @abstractmethod
    def connect(self) -> None:
        """
        Establish connection to the physical or mock instrument.
        Must transition from OFF to IDLE on success.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Safely shutdown and disconnect.
        Transition to OFF.
        """
        pass

    @abstractmethod
    def configure(self, settings: Dict[str, Any]) -> None:
        """
        Apply configuration.
        Transition from IDLE to CONFIGURED.
        """
        pass

    def require_state(self, allowed_states: "Union[List[InstrumentState], InstrumentState]"):
        """
        Enforce that the instrument is in one of the allowed states.
        Raises RuntimeError otherwise.
        """
        if isinstance(allowed_states, InstrumentState):
            allowed_states = [allowed_states]
        
        if self._state not in allowed_states:
            msg = f"Operation rejected for {self.name}: Current state {self._state.value} not in {[s.value for s in allowed_states]}"
            self.logger.error(msg)
            raise RuntimeError(msg)

    def handle_error(self, error_msg: str):
        """
        Standard error handler: logs error and forces into ERROR state.
        """
        self.logger.error(f"HARDWARE ERROR: {error_msg}")
        self.to_state(InstrumentState.ERROR)
        # In a real app, might want to trigger a safe shutdown here
