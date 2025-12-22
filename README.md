# Noise Spectrum Analyser

A professional Python-based instrument control and analysis software for measuring **Linear Dynamic Range (LDR)** and **Noise Equivalent Power (NEP)** of photodetectors. Built with Streamlit for a rich, interactive user interface.

## 🚀 Overview

This toolkit provides a layered architecture (UI -> Workflow -> Acquisition -> Hardware Drivers) to perform high-precision electronic measurements. It supports:
- **LDR Sweeps**: Automated sweeps of LED current vs. Photodetector response.
- **Noise Analysis**: Power Spectral Density (PSD) measurements using PicoScope.
- **Instrument Control**: Direct control for SMUs and Oscilloscopes.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tsarevsergey/photodetector_toolkit.git
   cd photodetector_toolkit
   ```

2. **Set up a Virtual Environment** (Recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🔌 Hardware Requirements & PicoSDK

This software interacts with laboratory hardware. To use the physical instruments, you must install the appropriate drivers and libraries.

### PicoScope Setup

For oscilloscope control (PicoScope 2000 or 4000 series), the `picosdk` Python package is required.

1. **Pico Technology SDK**: You must download and install the official PicoSDK C-libraries for your operating system (Windows/Linux/macOS) from the [Pico Technology website](https://www.picotech.com/downloads).
2. **Python Wrappers**: These are installed via `requirements.txt` (`pip install picosdk`). 
   - The software imports drivers from `picosdk.ps2000a` or `picosdk.ps4000a` depending on the device used.
   - If you are developing features for the wrappers, you can clone the [picosdk-python-wrappers](https://github.com/picotech/picosdk-python-wrappers) repository locally (this directory is ignored by Git in this project).

### SMU & Other Instruments
- Uses **PyVISA** for communication with Keysight/Keithley SMUs.
- Ensure the **NI-VISA** or **Keysight IO Libraries** are installed on your system.

## 🏃 Running the App

Start the Streamlit application:
```bash
streamlit run src/app.py
```

## 📂 Project Structure
- `src/app.py`: Main Dashboard.
- `src/pages/`: Individual analysis and control pages.
- `src/hardware/`: Drivers for SMU, PicoScope, and other peripherals.
- `src/workflows/`: Complex measurement logic (e.g., LDR sweeps).
- `tests/`: Hardware verification scripts.
