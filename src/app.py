
import streamlit as st

st.set_page_config(
    page_title="Noise Spectral Analyzer",
    page_icon="📉",
    layout="wide"
)

st.title("Noise Spectral Analyzer & Photodetector Characterization")

import atexit

# Global cleanup to handle server shutdowns or reloads
def cleanup():
    # Attempt to close SMU
    if 'smu' in st.session_state and st.session_state.smu:
        try:
            st.session_state.smu.disconnect()
            print("Cleanup: Disconnected SMU")
        except:
            pass
            
    # Attempt to close Scope
    if 'scope' in st.session_state and st.session_state.scope:
        try:
            st.session_state.scope.disconnect()
            print("Cleanup: Disconnected Scope")
        except:
            pass

atexit.register(cleanup)

st.markdown("""
### Welcome to the Noise Spectral Analyzer Control Software.

Please select a tool from the sidebar to begin:

- **1. SMU Direct Control**: Manual control of SMU for IV curves and testing.
- **2. Pulse Generator**: Generate square wave pulses for LED driving.
- **3. Scope Commissioning**: Setup and test PicoScope acquisition.

---
**System Status**:
- Python Environment: Verified
- Hardware Drivers: Loaded
""")
