# Technical Task  
## Noise Spectral Analyzer & Photodetector Characterization Software

**Version:** v1.0  
**Language:** Python  
**Target user:** Experimental physics / optoelectronics lab  
**Audience of this document:** Software developer (scientific instrumentation)

---

## 1. Project Goal

Develop a Python-based software package to control laboratory instruments and perform **noise spectral density**, **NEP**, and **linear dynamic range (LDR)** measurements of photodetectors using:

- FEMTO transimpedance amplifier (manual gain)
- PicoScope 4262 (16-bit oscilloscope)
- Current-driven LED source
- SMU (Keysight B2901 or Keithley via SCPI)

The software must be **reproducible**, **calibration-aware**, and suitable for **scientific publications**.

---

## 2. Supported Hardware

### 2.1 Instruments

| Instrument | Model | Control Method |
|-----------|------|----------------|
| Oscilloscope | PicoScope 4262 | PicoSDK |
| SMU | Keysight B2901 / Keithley | SCPI via PyVISA |
| Amplifier | FEMTO OE-200-IN1 | Manual (user input) |
| Light source | LED | SMU current control |
| Calibration detector | Si photodiode | FEMTO + scope |

---

## 3. Software Architecture (Required)

The implementation must follow a layered architecture:

```
UI Layer
└── Workflow Layer
    └── Acquisition Layer
        └── Hardware Drivers
```

Each layer must be independent and testable.

---

## 4. Technology Stack

- **Python ≥ 3.10**
- `numpy`, `scipy`
- `pyvisa`
- PicoScope Python SDK
- `h5py`
- `yaml` or `json`
- UI (MVP): `streamlit`

---

## 5. Hardware Driver Layer

### 5.1 SMU Driver

Implement a class that supports Keysight and Keithley SMUs via SCPI.

**Required functionality:**
- Set DC current
- Generate square-wave current pulses
- Set compliance voltage
- Enable/disable output

---

### 5.2 PicoScope Driver (4262)

**Required functionality:**
- Configure sampling parameters
- Block-mode acquisition
- Return time-domain voltage traces

---

### 5.3 Amplifier Configuration (Manual)

Amplifier gain is not controlled in software.

The user must manually enter:
- Nominal gain (V/A)
- Bandwidth mode
- Filter state

This metadata must be saved with every measurement.

---

## 6. Acquisition Layer

### 6.1 Pulse Response Acquisition

**Goal:** Measure detector response to pulsed LED illumination.

**Outputs:**
- Raw time traces
- Pulse metrics (peak, charge, baseline noise)

---

### 6.2 Noise Acquisition

**Goal:** Measure input-referred current noise.

**Method:**
- Long untriggered acquisition
- Welch PSD (windowed, overlapped)
- Convert voltage PSD → current ASD

---

## 7. Signal Processing Requirements

### PSD Estimation
- Window: Hann
- Overlap: 50%
- Method: Welch averaging

---

## 8. Calibration Workflow (Si Diode)

Map LED current to optical power using calibrated Si photodiode responsivity.

---

## 9. LDR Measurement Workflow

Multi-gain segmented measurement stitched in post-processing.

---

## 10. Data Storage

```
YYYYMMDD_RunName/
├── manifest.yaml
├── data.h5
├── notes.txt
└── exports/
```

---

## 11. User Interface (MVP)

- Streamlit-based UI
- Forms for parameters
- Time trace, PSD, and LDR plots

---

## 12. Validation & Self-Test

- Scope noise floor
- Amplifier noise
- Thermal noise of known resistor
- Calibration repeatability

---

## 13. Deliverables

- Modular Python package
- Instrument drivers
- Measurement workflows
- Streamlit UI
- Example dataset
- README

---

**End of technical task**

---

## 17. Development by AI Agent & MCP-Style Hardware Interaction

### 17.1 AI Agent Development Context

The software will be **developed primarily by an AI coding agent** under human supervision.  
Therefore, the codebase must be:

- Highly modular and self-describing
- Explicit in hardware assumptions
- Defensive against partial or ambiguous hardware states
- Easy to iterate, refactor, and extend autonomously

All modules must include:
- Clear docstrings
- Explicit input/output contracts
- Minimal hidden state

---

### 17.2 MCP-Style Hardware Interaction Requirement

The AI agent must also implement **MCP-style (Model–Controller–Peripheral) programs** that interact with **real physical laboratory equipment** connected to the host computer.

This implies:

- Hardware control logic must be **strictly separated** from:
  - UI
  - signal processing
  - experiment logic
- Each physical instrument must be wrapped as a **deterministic, state-aware controller**

---

### 17.3 MCP Conceptual Mapping

| Layer | Responsibility |
|------|---------------|
| Model | Measurement state, metadata, calibration data |
| Controller | Instrument command logic, sequencing, validation |
| Peripheral | Physical instruments (SMU, PicoScope, FEMTO amp) |

---

### 17.4 Hardware Controller Requirements

Each instrument controller must:

- Expose a **finite set of explicit states**
  - e.g. `IDLE`, `CONFIGURED`, `ARMED`, `ACQUIRING`, `ERROR`
- Validate state transitions
- Fail loudly on invalid sequences
- Provide a software-emulated **mock mode** for AI testing without hardware

Example:
```python
scope = Pico4262Controller(mock=True)
scope.configure(...)
scope.arm()
data = scope.acquire()
```

---

### 17.5 Safety & Robustness Constraints

Because real hardware is involved, the AI agent must implement:

- Current and voltage limit enforcement
- Safe default states on exceptions
- Automatic output shutdown on error
- Explicit user confirmation for destructive actions

---

### 17.6 Logging & Traceability

All MCP-style controllers must:
- Log every command sent to hardware
- Timestamp hardware interactions
- Save logs alongside measurement data

This is mandatory to allow post-mortem debugging of AI-driven experiments.

---

### 17.7 Acceptance Criteria (AI + MCP)

The implementation will be accepted only if:

- The AI agent can run the software in **mock mode** end-to-end
- The same code can switch to **real hardware mode** without changes
- No hardware action occurs without an explicit controller call
- Hardware failures do not crash the UI or corrupt saved data

---
