import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import glob
import os

import sys


def main():
    # Determine the directory to search
    # Priority: 1. Command line argument, 2. Script's directory
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Searching for .wav files in: {target_dir}")
    wav_pattern = os.path.join(target_dir, "*.wav")
    wav_files = glob.glob(wav_pattern)

    if not wav_files:
        print(f"No .wav files found in {target_dir}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all wav files
    for f in wav_files:
        try:
            sample_rate, data = wavfile.read(f)
            # If stereo, take only one channel for simplicity
            if len(data.shape) > 1:
                data = data[:, 0]

            duration = len(data) / sample_rate
            time = np.linspace(0, duration, len(data))

            ax.plot(time, data, label=os.path.basename(f), alpha=0.7)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    ax.set_title("WAV File Visualization")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)

    # Click handling
    click_coords = []

    def on_click(event):
        # Check if click is inside the axes
        if event.inaxes != ax:
            return

        # Record time (x-axis)
        t_click = event.xdata
        click_coords.append(t_click)

        # Draw vertical line
        ax.axvline(x=t_click, color="r", linestyle="--")
        plt.draw()

        print(f"Point {len(click_coords)} selected: {t_click:.4f} s")

        # If 2 points selected, calculate diff
        if len(click_coords) == 2:
            dt = abs(click_coords[1] - click_coords[0])
            print("-" * 30)
            print(f"Time Difference: {dt:.6f} seconds")
            print("-" * 30)
            # Reset for next measurement
            click_coords.clear()

    fig.canvas.mpl_connect("button_press_event", on_click)
    print("Click on the plot to select points.")
    plt.show()


if __name__ == "__main__":
    main()
