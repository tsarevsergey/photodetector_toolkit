import numpy as np
import sys
import os

# Mock path for testing
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))
import analysis.signal_processing as sig

def test_snr_with_50hz():
    fs = 10000.0
    t = np.arange(0, 2.0, 1/fs)
    target_f = 85.0
    
    # Signal (85 Hz)
    signal = 0.01 * np.sin(2 * np.pi * target_f * t)
    
    # Strong 50Hz Noise
    noise_50hz = 0.05 * np.sin(2 * np.pi * 50.0 * t)
    
    # White Noise
    white_noise = 0.005 * np.random.randn(len(t))
    
    voltage = signal + noise_50hz + white_noise
    
    # Analysis
    lockin_rms = sig.calculate_lockin_amplitude(voltage, fs, target_f)
    print(f"Target Freq: {target_f} Hz")
    print(f"True Signal RMS: {0.01 / np.sqrt(2):.6f}")
    print(f"Lock-in RMS: {lockin_rms:.6f}")
    
    fft_snr = sig.calculate_snr_fft(voltage, fs, target_f)
    robust_snr = sig.calculate_robust_snr(voltage, fs, target_f)
    
    print(f"FFT SNR: {fft_snr:.2f}")
    print(f"Robust SNR (Lock-in/Sideband): {robust_snr:.2f}")

    # Now test what happens if we mistakenly look for "max peak"
    # (Simulated by picking 50Hz as target)
    fft_snr_50 = sig.calculate_snr_fft(voltage, fs, 50.0)
    print(f"FFT SNR at 50Hz (Wrong target): {fft_snr_50:.2f}")

def test_no_signal():
    fs = 10000.0
    t = np.arange(0, 2.0, 1/fs)
    target_f = 85.0
    
    # Only noise, NO SIGNAL
    white_noise = 0.005 * np.random.randn(len(t))
    voltage = white_noise
    
    # Analysis
    fft_snr = sig.calculate_snr_fft(voltage, fs, target_f)
    robust_snr = sig.calculate_robust_snr(voltage, fs, target_f)
    
    print(f"\n--- No Signal Case (Target: {target_f} Hz) ---")
    print(f"FFT SNR: {fft_snr:.2f} (Should be ~1.0)")
    print(f"Robust SNR (Lock-in/Sideband): {robust_snr:.2f} (Should be ~1.0)")

if __name__ == "__main__":
    test_snr_with_50hz()
    test_no_signal()
