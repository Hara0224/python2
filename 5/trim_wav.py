import numpy as np
from scipy.io import wavfile
import glob
import os
import sys

def main():
    # Determine directory
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = os.path.dirname(os.path.abspath(__file__))

    wav_pattern = os.path.join(target_dir, '*.wav')
    wav_files = glob.glob(wav_pattern)

    if not wav_files:
        print(f"No .wav files found in {target_dir}")
        return

    print(f"\nFound {len(wav_files)} files in {target_dir}:")
    for i, f in enumerate(wav_files):
        print(f"[{i}] {os.path.basename(f)}")

    # Select file
    while True:
        try:
            selection = input("\nSelect file index (0-N): ")
            idx = int(selection)
            if 0 <= idx < len(wav_files):
                selected_file = wav_files[idx]
                break
            else:
                print("Invalid index. Try again.")
        except ValueError:
            print("Please enter a number.")

    # Get trim parameters
    try:
        start_trim = float(input("Seconds to remove from START: "))
        end_trim = float(input("Seconds to remove from END: "))
    except ValueError:
        print("Invalid input. Please enter numbers (e.g., 0.5).")
        return

    # Process file
    try:
        sample_rate, data = wavfile.read(selected_file)
        
        # Calculate indices
        n_start = int(start_trim * sample_rate)
        n_end = int(end_trim * sample_rate)
        total_frames = data.shape[0]

        # Validations
        if n_start + n_end >= total_frames:
            print("Error: Trim duration longer than file duration.")
            return

        # Slice
        if n_end > 0:
            trimmed_data = data[n_start : -n_end]
        else:
            trimmed_data = data[n_start:]

        # Create output filename
        base, ext = os.path.splitext(selected_file)
        output_file = f"{base}_trimmed{ext}"

        # Save
        wavfile.write(output_file, sample_rate, trimmed_data)
        
        original_duration = total_frames / sample_rate
        new_duration = trimmed_data.shape[0] / sample_rate

        print(f"\nSuccess! Trimmed file saved as: {os.path.basename(output_file)}")
        print(f"Original duration: {original_duration:.2f}s")
        print(f"New duration:      {new_duration:.2f}s")

    except Exception as e:
        print(f"An error occurred processing the file: {e}")

if __name__ == "__main__":
    main()
