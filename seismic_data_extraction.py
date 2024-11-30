import os
import matplotlib.pyplot as plt
import segyio
import numpy as np
import pandas as pd

def seismic_extraction(segyfile, wellname):
    # Open seismic file
    with segyio.open(segyfile, "r", ignore_geometry=True) as f:
        # Access traces as a numpy array
        traces = segyio.tools.collect(f.trace.raw[:])  # All traces
        depths = np.arange(traces.shape[1])  # Depth levels

    # Compute amplitude attributes
    avg_amplitude = np.mean(np.abs(traces), axis=0)  # Mean absolute amplitude per depth
    max_amplitude = np.max(np.abs(traces), axis=0)   # Max amplitude per depth

    # FFT for dominant frequency
    from scipy.fftpack import fft
    fft_spectrum = fft(traces, axis=1)
    power_spectrum = np.abs(fft_spectrum)**2
    dominant_frequency = np.argmax(power_spectrum, axis=1)  # Dominant frequency per trace

    lengths = [
        len(depths),
        len(avg_amplitude),
        len(max_amplitude),
        len(dominant_frequency)
    ]
    max_length = max(lengths)

    def pad_with_nan(array, max_length):
        array = np.asarray(array, dtype=float)
        return np.pad(array, (0, max_length - len(array)), constant_values=np.nan)

    dominant_frequency = pad_with_nan(dominant_frequency, max_length)
    avg_amplitude = pad_with_nan(avg_amplitude, max_length)
    max_amplitude = pad_with_nan(max_amplitude, max_length)
    depths = pad_with_nan(depths, max_length)



    features = pd.DataFrame({
        "Depth": depths,
        "AvgAmplitude": avg_amplitude,
        "MaxAmplitude": max_amplitude,
        "Dominant Frequency": dominant_frequency
    })

    features.to_csv(f"seismic_features_{wellname}.csv", index=False)
    print(f"Features saved for {wellname}.")

root_dir = "/Users/vegardhatleli/Library/Mobile Documents/com~apple~CloudDocs/NTNU/09 - I&IKT Høst 2024/GeoHackathon/Data/2D Seismic"  # Update to the actual folder path

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".sgy"):  
            file_path = os.path.join(subdir, file)

            folder_name = os.path.basename(subdir)
            file_name = os.path.splitext(file)[0] 
            wellname = f"{folder_name}_{file_name}"
            seismic_extraction(file_path, wellname)
