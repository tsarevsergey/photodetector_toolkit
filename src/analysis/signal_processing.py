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
    n_bins_signal = end_bin - start_bin
    total_peak_power = np.sum(mag_sq[start_bin:end_bin])
    
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
        
    avg_noise_power_per_bin = np.median(noise_bins)
    
    if avg_noise_power_per_bin < 1e-25: # Floor
        return total_peak_power / 1e-25
        
    # --- REFINE SNR CALCULATION ---
    # The user observed that summing M bins of noise gives SNR ~ M if we don't normalize.
    # Correct Integrated SNR = (Signal Energy) / (Expected Noise Energy in same band)
    # Signal Energy = Sum(Peak Bins) - M * (Noise per bin) [Optional: Subtracting baseline makes it 0 for noise]
    # SNR (Ratio) = Total Peak Energy / (M * Noise floor per bin)
    
    expected_noise_energy = n_bins_signal * avg_noise_power_per_bin
    
    # We use a cautious "Excess Power" approach:
    # SNR = (Integrated_Power) / (Expected_Noise_Power_in_Band)
    # If it's just noise, this ratio will be approximately 1.0.
    snr = total_peak_power / expected_noise_energy
    
    return float(snr)

def calculate_lockin_amplitude(voltage: np.ndarray, fs: float, target_freq: float) -> float:
    """
    Calculates signal amplitude using digital lock-in (IQ demodulation).
    Robust against noise and slight frequency drift.
    Returns RMS voltage of the signal component.
    """
    n = len(voltage)
    t = np.arange(n) / fs
    
    # Reference signals (I and Q)
    ref_i = np.sin(2 * np.pi * target_freq * t)
    ref_q = np.cos(2 * np.pi * target_freq * t)
    
    # Demodulate
    sig_i = voltage * ref_i
    sig_q = voltage * ref_q
    
    # Low-pass filter (Mean over integer periods)
    # Ideally simpler: Just mean works if N is large and non-coherent errors avg out
    mean_i = np.mean(sig_i)
    mean_q = np.mean(sig_q)
    
    # Magnitude
    # V_peak = 2 * sqrt(I^2 + Q^2) 
    # (Factor of 2 comes from sin^2 average = 0.5)
    v_peak = 2 * np.sqrt(mean_i**2 + mean_q**2)
    
    # Return RMS
    return v_peak / np.sqrt(2)

def calculate_noise_density_sideband(voltage: np.ndarray, fs: float, target_freq: float) -> float:
    """
    Estimates noise density (V/rtHz) by looking at sidebands around target freq.
    """
    # Use Full Length for resolution
    f, pxx = signal.welch(voltage, fs, nperseg=len(voltage), window='hann')
    
    # Define sideband regions (e.g. +/- 10% to 20% away)
    # Avoid the peak itself
    idx_target = np.argmin(np.abs(f - target_freq))
    
    # Window of exclusion (e.g. +/- 5Hz or 5 bins)
    bin_width = f[1] - f[0]
    exclusion_width = max(5.0, target_freq * 0.05)
    exclusion_bins = int(exclusion_width / max(bin_width, 1e-6))
    
    # Region of interest: target +/- 5*exclusion
    roi_width_bins = max(exclusion_bins * 5, 10)
    
    start_bin = max(0, idx_target - roi_width_bins)
    end_bin = min(len(f), idx_target + roi_width_bins)
    
    # Mask out the signal peak
    mask = np.ones(end_bin - start_bin, dtype=bool)
    # Center relative
    center = idx_target - start_bin
    
    m_start = max(0, center - exclusion_bins)
    m_end = min(len(mask), center + exclusion_bins + 1)
    mask[m_start : m_end] = False
    
    roi_pxx = pxx[start_bin:end_bin][mask]
    
    if len(roi_pxx) == 0:
        return 1e-9 # Fallback
        
    avg_noise_power_density = np.median(roi_pxx) # V^2/Hz
    return np.sqrt(avg_noise_power_density)

def calculate_robust_snr(voltage: np.ndarray, fs: float, target_freq: float) -> float:
    """
    Calculates SNR using Lock-In Amplitude / Sideband Noise.
    SNR = (Signal_RMS**2) / (Noise_Density**2 * BinWidth)
    """
    if len(voltage) < 10: return 0.0
    
    # 1. Signal RMS
    v_rms = calculate_lockin_amplitude(voltage, fs, target_freq)
    
    # 2. Noise Density
    v_n_density = calculate_noise_density_sideband(voltage, fs, target_freq)
    
    # 3. Bandwidth (ENBW)
    # The relevant bandwidth is the Effective Noise Bandwidth of the Lock-In Filter.
    # For a complex IQ demodulation followed by a mean over duration T:
    # The noise variance in V_rms^2 is 2 * (n^2 / (2T)) + ...?
    # As derived: E[V_rms^2] from noise density n is (2 * n^2) / T.
    
    duration = len(voltage) / fs
    enbw = 2.0 / duration # Effective noise bandwidth for the RMS estimator
    
    if v_n_density < 1e-18: return 1000.0 # High SNR Cap
    
    # Power Ratio
    signal_power_total = v_rms**2
    noise_power_in_band = (v_n_density**2) * enbw
    
    # SNR = Total Power / Noise Power
    # For pure noise, this should hover around 1.0.
    snr = signal_power_total / noise_power_in_band
    
    return float(snr)

def calculate_johnson_noise(resistance_ohms: float, temp_k: float = 300.0) -> float:
    """
    Calculates thermal (Johnson) noise voltage density in V/rtHz.
    V_n = sqrt(4 * k_B * T * R)
    """
    if resistance_ohms <= 0: return 0.0
    k_B = 1.380649e-23
    return np.sqrt(4 * k_B * temp_k * resistance_ohms)

def extract_valid_pulse_train(
    times: np.ndarray, 
    volts: np.ndarray, 
    frequency: float,
    min_duration_cycles: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyzes the waveform to find the longest continuous segment of valid pulses.
    Useful for cleaning up streaming captures where the SMU might have paused or glitched.
    
    Args:
        times: Time array
        volts: Voltage array
        frequency: Expected pulse frequency
        min_duration_cycles: Minimum length to consider valid
        
    Returns:
        (valid_times, valid_volts) - Slice of the original arrays. 
        Returns ([], []) if no valid segment found.
    """
    if len(volts) < 2: return np.array([]), np.array([])
    
    fs = 1.0 / (times[1] - times[0])
    period_samples = int(fs / frequency)
    if period_samples < 2: period_samples = 2
    
    # 1. Calculate Activity Envelope (Rolling Std Dev)
    # Use simple uniform filter approximation for speed
    # envelope ~ std dev over 1 period
    
    # Remove DC offset first
    v_ac = volts - np.mean(volts)
    
    # Rolling variance ~ correlate v^2 with boxcar? 
    # Or just use scipy.ndimage.uniform_filter1d
    from scipy.ndimage import uniform_filter1d
    
    # Rectified envelope is easier/faster than std
    # Low pass filter the rectified signal
    rectified = np.abs(v_ac)
    envelope = uniform_filter1d(rectified, size=period_samples * 2)
    
    # 2. Threshold
    # Active region should have envelope > 15% of median envelope (or max?)
    # If the signal dropped to 0, envelope goes to close to 0 (noise floor).
    # Use robust max (95th percentile) to set scale
    max_env = np.percentile(envelope, 95)
    threshold = 0.15 * max_env
    
    # If threshold is too close to noise floor? 
    # Check noise floor (5th percentile)
    noise_floor = np.percentile(envelope, 5)
    if threshold < 2 * noise_floor:
        threshold = 2 * noise_floor # Ensure we are above noise
        
    mask = envelope > threshold
    
    # 3. Find Longest Continuous Region
    # Identify changes
    # pad with False to find edges
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    if len(starts) == 0:
        return np.array([]), np.array([])
        
    durations = ends - starts
    best_idx = np.argmax(durations)
    
    s = starts[best_idx]
    e = ends[best_idx]
    
    # Check minimum duration
    cycles_found = (e - s) / period_samples
    if cycles_found < min_duration_cycles:
        return np.array([]), np.array([])
        
    # 4. Return Slice
    return times[s:e], volts[s:e]

