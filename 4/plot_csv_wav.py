import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import os

def plot_csv_and_wav(csv_path, wav_path, emg_channels=['CH5', 'CH6']):
    """
    Plots EMG data from a CSV file and Audio data from a WAV file on the same graph.
    
    Args:
        csv_path (str): Path to the CSV file containing EMG data.
        wav_path (str): Path to the WAV file containing audio data.
        emg_channels (list): List of column names to plot from the CSV (e.g., ['CH1', 'CH2']).
    """
    print(f"Loading CSV: {csv_path}")
    # Load CSV Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Process CSV Time
    # Assuming 'Timestamp' column exists based on previous files. 
    # If not, we might need to create a time array based on sampling rate if known, 
    # but 'plot_rms.py' used 'Timestamp'.
    if 'Timestamp' in df.columns:
        csv_time = df['Timestamp'] - df['Timestamp'].iloc[0]
    else:
        print("Warning: 'Timestamp' column not found in CSV. Using index as time (assuming 1ms sampling if not specified).")
        csv_time = df.index * 0.001 # Fallback assumption

    print(f"Loading WAV: {wav_path}")
    # Load WAV Data
    try:
        fs, wav_data = wavfile.read(wav_path)
    except FileNotFoundError:
        print(f"Error: WAV file not found at {wav_path}")
        return

    # Process WAV Time
    # wav_data can be stereo (N, 2) or mono (N,)
    wav_duration = len(wav_data) / fs
    wav_time = np.linspace(0, wav_duration, len(wav_data))
    
    # If stereo, just pick the first channel for plotting transparency
    if wav_data.ndim > 1:
        wav_amplitude = wav_data[:, 0]
        print("WAV is stereo, plotting first channel only.")
    else:
        wav_amplitude = wav_data

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot EMG on the left axis
    color_idx = 0
    colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:purple']
    
    for ch in emg_channels:
        if ch in df.columns:
            color = colors[color_idx % len(colors)]
            ax1.plot(csv_time, df[ch], label=f'EMG {ch}', color=color, alpha=0.7, linewidth=1)
            color_idx += 1
        else:
            print(f"Warning: Channel {ch} not found in CSV columns.")

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('EMG Amplitude', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # Create a second y-axis for the WAV data
    ax2 = ax1.twinx() 
    
    # Plot WAV on the right axis
    ax2.plot(wav_time, wav_amplitude, label='Audio (WAV)', color='tab:blue', alpha=0.5, linewidth=0.5)
    ax2.set_ylabel('Audio Amplitude', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    # Align x-axis to the union of both durations or just one?
    # Usually we want to see the overlapping part or the whole thing.
    # Let's set the limit to the max duration of either.
    max_time = max(csv_time.iloc[-1], wav_time[-1])
    ax1.set_xlim(0, max_time)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title(f"EMG and Audio Signal Synchronization\nCSV: {os.path.basename(csv_path)} | WAV: {os.path.basename(wav_path)}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Default paths - CHANGE THESE AS NEEDED
    # Using raw strings (r"...") to handle backslashes on Windows
    
    # Example file from the user's directory
    default_csv = r"C:\Users\hrsyn\Desktop\masterPY\emg_data_13\emgraw_big_20251222113439.csv"
    
    # Example wav file
    default_wav = r"C:\Users\hrsyn\Desktop\masterPY\5\義手.wav"
    
    # Check if files exist before running default
    if os.path.exists(default_csv) and os.path.exists(default_wav):
        plot_csv_and_wav(default_csv, default_wav, emg_channels=['CH1', 'CH2'])  # Adjust channels as needed
    else:
        print("Default files not found. Please edit the 'default_csv' and 'default_wav' variables in the script.")
        print(f"Checked CSV: {default_csv}")
        print(f"Checked WAV: {default_wav}")
