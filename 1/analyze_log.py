
import pandas as pd
import numpy as np
import glob
import os

# Find the latest CSV file
list_of_files = glob.glob('c:/Users/hrsyn/Desktop/masterPY/1/emg_data_*.csv') 
latest_file = max(list_of_files, key=os.path.getctime)
print(f"Analyzing: {latest_file}")

df = pd.read_csv(latest_file)

# Columns: timestamp, ch1..8, ard_micros, vib1_z, vib2_z, state
# State: 0=Idle, 1=Attacking, 2=Cooldown

# Identify Attack events
# Find continuous segments of state=1
df['group'] = (df['state'] != df['state'].shift()).cumsum()
attacks = df[df['state'] == 1]

if attacks.empty:
    print("No attacks found in logs.")
    exit()

attack_groups = attacks.groupby('group')

print(f"Found {len(attack_groups)} attack events.")

# Channels for Down (Strong) detection: CH5, CH6 (indices 4, 5 in 0-indexed, or ch5, ch6 in csv)
# V19.py: DOWN_CH = [5, 6] -> This corresponds to array indices 5 and 6, which are CH6 and CH7 in 1-based notation?
# Let's check V19.py: 
#   ema_val = np.zeros(8)
#   rms_buf index 0..7
#   DOWN_CH = [5, 6]
#   So indices 5 and 6.
#   CSV header is ch1..ch8. So indices 5 and 6 correspond to 'ch6' and 'ch7'.

down_channels = ['ch6', 'ch7']

peak_values = []
peak_ratios_sigma = []

# To calculate sigma ratio, we need mean and std.
# We can estimate them from the period before the FIRST attack or just generally from IDLE state.
# Ideally, we calculate it from the calibration period, but that's not explicitly marked in state (maybe state 0?).
# Calibration is usually at the start.
# Let's simple take all State 0 data as "Idle/Calibration" for rough stats, or just the first few seconds.

# Better: use the user's calibration logic. 
# "mean + 10sigma"
# Let's calculate mean and std from the entire IDLE (state 0) data for simplicity? 
# Or just the first 3 seconds of data if available.

idle_data = df[df['state'] == 0]
if idle_data.empty:
    print("No idle data found.")
    mean = np.zeros(8)
    std = np.ones(8)
else:
    # Use global mean/std for simplification
    mean = idle_data[['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']].mean()
    std = idle_data[['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']].std()

print("Estimated Baseline (Idle):")
print(f"Mean CH6: {mean['ch6']:.4f}, Std CH6: {std['ch6']:.4f}")
print(f"Mean CH7: {mean['ch7']:.4f}, Std CH7: {std['ch7']:.4f}")

print("\n--- Attack Events ---")
for _, group in attack_groups:
    # Get peak of the combined or max of individual? 
    # V19 logic:
    #   measure_buffer.append(down_level)
    #   down_level = np.max([ema_val[ch] for ch in DOWN_CH])
    #   peak_val = np.max(measure_buffer)
    
    # Let's replicate this
    group_max_per_row = group[down_channels].max(axis=1)
    peak_val = group_max_per_row.max()
    
    # Calculate how many sigmas this is
    # Using the channel that contributed to the peak? 
    # Or just taking the max z-score relative to its own channel?
    
    # Replicate V19 logic roughly:
    # strong_thr is mean + 10*std.
    # So ratio = (peak - mean) / std.
    # But peak comes from max of ch6/ch7. 
    # Let's compute Z-score for both and take max.
    
    z_scores_list = []
    for ch in down_channels:
        m = mean[ch]
        s = std[ch]
        peak_ch = group[ch].max() # Peak of this channel in this event
        z = (peak_ch - m) / s if s > 0 else 0
        z_scores_list.append(z)
        
    max_z = max(z_scores_list)
    
    peak_values.append(peak_val)
    peak_ratios_sigma.append(max_z)
    
    print(f"Peak: {peak_val:.4f}, Max Z-Score: {max_z:.2f}")

print("\n--- Summary ---")
print(f"Average Peak Z-Score: {np.mean(peak_ratios_sigma):.2f}")
print(f"Max Peak Z-Score: {np.max(peak_ratios_sigma):.2f}")
print(f"Min Peak Z-Score: {np.min(peak_ratios_sigma):.2f}")

# Suggestion
print(f"\nCurrent Threshold setting in V19 is 'mean + 10.0 * sigma'.")
print(f"Observed peaks are around {np.mean(peak_ratios_sigma):.2f} sigma.")

