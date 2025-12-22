import numpy as np
from scipy import signal
from typing import Tuple, Optional

def calculate_psd(
    voltage: np.ndarray, 
    fs: float, 
    window: str = 'hann', 
    nperseg: Optional[int] = None, 
    overlap_percent: float = 50.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Power Spectral Density (PSD) using Welch's method.
    
    Args:
        voltage: 1D array of voltage samples.
        fs: Sampling frequency in Hz.
        window: Window function to use (default 'hann').
        nperseg: Length of each segment. If None, defaults to 256 or appropriate size.
        overlap_percent: Overlap between segments (0-100).
    
    Returns:
        f: Array of sample frequencies.
        pxx: Power Spectral Density (V**2/Hz).
    """
    if nperseg is None:
        nperseg = min(len(voltage), 4096) # Default to a reasonable chunk size
        
    noverlap = int(nperseg * (overlap_percent / 100.0))
    
    f, pxx = signal.welch(voltage, fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return f, pxx

def calculate_asd(
    voltage: np.ndarray, 
    fs: float, 
    transimpedance_gain: float,
    window: str = 'hann', 
    nperseg: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Amplitude Spectral Density (ASD) in Amps/sqrt(Hz).
    
    Args:
        voltage: 1D array of voltage samples.
        fs: Sampling frequency in Hz.
        transimpedance_gain: Amplifier gain in V/A.
        
    Returns:
        f: Frequency array.
        asd: Current ASD in A/sqrt(Hz).
    """
    f, pxx_voltage = calculate_psd(voltage, fs, window=window, nperseg=nperseg)
    
    # Convert Voltage PSD (V^2/Hz) to Voltage ASD (V/sqrt(Hz))
    asd_voltage = np.sqrt(pxx_voltage)
    
    # Convert to Current ASD (A/sqrt(Hz))
    # V = I * R -> I = V / R
    asd_current = asd_voltage / transimpedance_gain
    
    return f, asd_current
