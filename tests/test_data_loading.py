
import os
import sys
import pandas as pd
import numpy as np
import io

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from analysis.signal_processing import load_noise_trace_csv

def test_load_noise_trace_csv():
    # 1. Create a dummy CSV with metadata
    csv_content = """# Timestamp: 2026-01-21 15:10:48
# TIA_Gain_VA: 1.0e+06
# Active_Area_cm2: 0.5
time,voltage
0.0,0.001
0.1,0.002
0.2,0.0015
"""
    
    file_obj = io.BytesIO(csv_content.encode('utf-8'))
    # Mocking streamlit uploaded file object properties if needed, but load_noise_trace_csv uses getvalue()
    class MockUploadedFile:
        def getvalue(self):
            return csv_content.encode('utf-8')
            
    df, meta = load_noise_trace_csv(MockUploadedFile())
    
    print("Metadata:", meta)
    print("DataFrame Header:\n", df.head())
    
    assert meta['TIA_Gain_VA'] == 1.0e6
    assert meta['Active_Area_cm2'] == 0.5
    assert len(df) == 3
    assert 'time' in df.columns
    assert 'voltage' in df.columns
    
    print("Test Passed!")

if __name__ == "__main__":
    test_load_noise_trace_csv()
