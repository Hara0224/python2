import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, simpledialog
import os


def calculate_rms(data, window_size=50):
    """
    Calculate Root Mean Square (RMS) using a rolling window.
    """
    return np.sqrt(data.pow(2).rolling(window=window_size).mean())


def main():
    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw()

    # Open file dialog to select CSV
    print("Please select a CSV file...")
    file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")], initialdir=os.getcwd())

    if not file_path:
        print("No file selected.")
        return

    print(f"File selected: {file_path}")

    try:
        # Read CSV
        df = pd.read_csv(file_path)

        time_col = None
        if "Timestamp" in df.columns:
            time_col = df["Timestamp"]
            # Reset time to start at 0
            time_col = time_col - time_col.iloc[0]
            non_target_cols = ["Timestamp"]
        else:
            time_col = None
            non_target_cols = []

        # Identify potential data columns (numeric)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        data_cols_candidates = [c for c in numeric_cols if c not in non_target_cols]

        if not data_cols_candidates:
            print("No numeric data columns found.")
            return

        # Input Window Size
        window_size = simpledialog.askinteger("Input", "Enter RMS Window Size:", initialvalue=50, minvalue=1)
        if not window_size:
            window_size = 50  # Default if cancelled or empty? askinteger returns None if cancelled.
            # If cancelled, maybe we should stop? Or strictly default.
            print("Window size input cancelled or invalid, using default: 50")
            window_size = 50

        # Let user select channels
        selected_cols = []

        def on_ok(dialog, lb):
            selection = lb.curselection()
            for i in selection:
                selected_cols.append(lb.get(i))
            dialog.destroy()

        dialog = tk.Toplevel(root)
        dialog.title("Select Channels")
        dialog.geometry("300x400")

        tk.Label(dialog, text="Select channels to plot (Ctrl+Click for multiple):").pack(pady=5)

        lb = tk.Listbox(dialog, selectmode=tk.MULTIPLE, width=40, height=15)
        for col in data_cols_candidates:
            lb.insert(tk.END, col)
        lb.pack(padx=10, pady=5, expand=True, fill=tk.BOTH)

        tk.Button(dialog, text="OK", command=lambda: on_ok(dialog, lb), width=10).pack(pady=10)

        print("Please select channels from the dialog...")
        dialog.wait_window()

        if not selected_cols:
            print("No channels selected. Exiting.")
            return

        print(f"Channels selected: {selected_cols}")

        # Input Y-axis Max
        y_max = simpledialog.askfloat("Input", "Enter Y-axis Max Limit (Min=0):", initialvalue=100.0, minvalue=0.1)
        if not y_max:
            print("Y-axis Max input cancelled, using default: Auto")
            y_max = None

        # Calculate RMS for each data column
        rms_df = pd.DataFrame()
        if time_col is not None:
            rms_df["Time"] = time_col

        for col in selected_cols:
            rms_df[col] = calculate_rms(df[col], window_size)

        # Plotting with Subplots
        num_channels = len(selected_cols)
        fig, axes = plt.subplots(num_channels, 1, figsize=(10, 3 * num_channels), sharex=True)

        # If there's only one channel, axes is not a list, so make it iterable
        if num_channels == 1:
            axes = [axes]

        for i, col in enumerate(selected_cols):
            ax = axes[i]
            if time_col is not None:
                ax.plot(rms_df["Time"], rms_df[col], label=f"{col} (RMS)")
            else:
                ax.plot(rms_df[col], label=f"{col} (RMS)")

            ax.set_ylabel("Amplitude")
            ax.set_title(f"Channel: {col}")
            if y_max is not None:
                ax.set_ylim(0, y_max)
            ax.legend(loc="upper right")
            ax.grid(True)

        if time_col is not None:
            axes[-1].set_xlabel("Time [s]")
        else:
            axes[-1].set_xlabel("Sample")

        fig.suptitle(f"RMS Analysis of {os.path.basename(file_path)} (Window={window_size})", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Adjust for suptitle
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
