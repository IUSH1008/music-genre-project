
import os
import time

# Measure library initialization
start = time.perf_counter()

import librosa
import librosa.display
import numpy as np
import matplotlib

# Save graphs without opening display windows
matplotlib.use("Agg")

import matplotlib.pyplot as plt

print("Libraries imported in:", round(time.perf_counter() - start, 2), "seconds")

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_PATH = "data/raw/Data/genres_original"

audio_path = os.path.join(
    DATASET_PATH,
    "classical",
    "classical.00000.wav"
)

os.makedirs("outputs", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ==========================================
# 2. LOAD AUDIO
# ==========================================

start = time.perf_counter()

audio, sample_rate = librosa.load(
    audio_path,
    sr=22050,
    mono=True
)

print("\nAUDIO INFORMATION")
print("-----------------")
print("Sample Rate:", sample_rate)
print("Duration:", round(len(audio) / sample_rate, 2))
print("Total Samples:", len(audio))

print("Audio loading time:",
      round(time.perf_counter() - start, 2), "seconds")

# ==========================================
# 3. WAVEFORM
# ==========================================

start = time.perf_counter()

time_axis = np.arange(len(audio)) / sample_rate

# Plot every 10th sample
plt.figure(figsize=(10, 3))

plt.plot(
    time_axis[::10],
    audio[::10],
    linewidth=0.5
)

plt.title("Classical Music - Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.savefig("outputs/classical_waveform.png", dpi=100)
plt.close()

print("Waveform generation:",
      round(time.perf_counter() - start, 2), "seconds")

# ==========================================
# 4. SPECTROGRAM
# ==========================================

start = time.perf_counter()

stft = librosa.stft(
    audio,
    n_fft=1024,
    hop_length=1024
)

spectrogram = librosa.amplitude_to_db(
    np.abs(stft),
    ref=np.max
)

plt.figure(figsize=(10, 4))

librosa.display.specshow(
    spectrogram,
    sr=sample_rate,
    hop_length=1024,
    x_axis="time",
    y_axis="hz"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Classical Music - Spectrogram")
plt.tight_layout()

plt.savefig("outputs/classical_spectrogram.png", dpi=100)
plt.close()

print("Spectrogram generation:",
      round(time.perf_counter() - start, 2), "seconds")

# ==========================================
# 5. MFCC EXTRACTION
# ==========================================

start = time.perf_counter()

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=13,
    n_fft=2048,
    hop_length=1024
)

print("\nMFCC INFORMATION")
print("----------------")
print("MFCC Shape:", mfcc.shape)
print("Number of coefficients:", mfcc.shape[0])

# Save numerical features
np.save(
    "data/processed/classical_mfcc.npy",
    mfcc
)

print("MFCC extraction and saving:",
      round(time.perf_counter() - start, 2), "seconds")

# ==========================================
# 6. MFCC VISUALIZATION
# ==========================================

start = time.perf_counter()

plt.figure(figsize=(10, 4))

librosa.display.specshow(
    mfcc,
    sr=sample_rate,
    hop_length=1024,
    x_axis="time"
)

plt.colorbar()
plt.title("Classical Music - MFCC Features")
plt.tight_layout()

plt.savefig("outputs/classical_mfcc.png", dpi=100)
plt.close()

print("MFCC visualization:",
      round(time.perf_counter() - start, 2), "seconds")

# ==========================================
# 7. COMPLETION
# ==========================================

print("\nALL OPERATIONS COMPLETED!")
print("Check outputs/ and data/processed/")

print("\nTotal execution time:",
      round(time.perf_counter() - start, 2), "seconds")
