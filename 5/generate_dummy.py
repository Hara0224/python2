import numpy as np
from scipy.io import wavfile

def generate_wav(filename, freq, duration=2.0, rate=44100):
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    # Generate sine wave
    data = 0.5 * np.sin(2 * np.pi * freq * t)
    # Convert to 16-bit integer PCM
    scaled = np.int16(data * 32767)
    wavfile.write(filename, rate, scaled)
    print(f"Generated {filename}")

if __name__ == "__main__":
    generate_wav("test_signal_440Hz.wav", 440)
    generate_wav("test_signal_880Hz.wav", 880)
