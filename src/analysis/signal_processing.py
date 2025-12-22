import numpy as np
from scipy import signal
from typing import Tuple, Optional, Dict

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

def calculate_snr_fft(
    voltage: np.ndarray, 
    fs: float, 
    target_freq: float,
    rbw_bins: int = 5
) -> float:
    """
    Calculates SNR in the frequency domain.
    SNR = (Peak Power at target_freq) / (Average Noise Power in vicinity)
    
    Args:
        voltage: Time domain signal.
        fs: Sampling frequency.
        target_freq: The expected frequency of the LED signal.
        rbw_bins: Number of bins around peak to consider as "Signal".
        
    Returns:
        snr: Linear ratio of peak power to noise floor power.
    """
    # Use Hamming window to reduce leakage
    n = len(voltage)
    win = np.hamming(n)
    v_win = (voltage - np.mean(voltage)) * win
    
    # FFT
    ft = np.fft.rfft(v_win)
    freqs = np.fft.rfftfreq(n, 1/fs)
    mag_sq = np.abs(ft)**2
    
    # Find bin closest to target_freq
    idx = np.argmin(np.abs(freqs - target_freq))
    
    # Signal Power: Sum around the peak
    # (Using a few bins to account for minor frequency drift or finite window resolution)
    start_bin = max(0, idx - rbw_bins)
    end_bin = min(len(mag_sq), idx + rbw_bins + 1)
    signal_power = np.sum(mag_sq[start_bin:end_bin])
    
    # Noise Floor Power: Average level in the vicinity (excluding the signal peak)
    # Take a 10% frequency span or at least 50 bins
    window_half = max(25, int(len(mag_sq) * 0.05))
    noise_start = max(0, idx - window_half)
    noise_end = min(len(mag_sq), idx + window_half + 1)
    
    # Create mask to exclude signal area from noise calculation
    mask = np.ones(noise_end - noise_start, dtype=bool)
    # Relative indices in the noise window
    mask_start = start_bin - noise_start
    mask_end = end_bin - noise_start
    mask[max(0, mask_start):min(len(mask), mask_end)] = False
    
    noise_bins = mag_sq[noise_start:noise_end][mask]
    
    if len(noise_bins) == 0:
        return 0.0
        
    avg_noise_power = np.median(noise_bins)
    
    # Scale correction: For Rayleigh distributed noise (magnitude), median is lower than mean.
    # For Power (Chi-squared 2-DOF), Mean = Median / ln(2) approx?
    # Actually, for estimating the floor level of white noise, median is a robust estimator.
    # Let's stick to median as "Representative Noise Floor Level".
    
    if avg_noise_power < 1e-15: # Floor
        return signal_power / 1e-15
        
    # We want SNR = Signal Energy / Noise Energy per bin? 
    # Or Peak/Floor? The user likely wants Peak/Floor ratio.
    # Signal Power calculated above is Sum of bins (Total Signal Energy).
    # Noise Power above is per-bin median.
    
    snr = signal_power / avg_noise_power
    return float(snr)
