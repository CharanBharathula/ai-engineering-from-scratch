# Phase 06: Speech and Audio

Audio processing involves converting sound waves into digital signals that machine learning models can understand, often translating between speech and text.

## Roadmap

| Lesson | Description | Status |
|--------|-------------|--------|
| 01. Audio Data Foundations | Waveforms, Sampling Rates, and Spectrograms. | ✅ |
| 02. Feature Extraction | MFCCs (Mel-Frequency Cepstral Coefficients) using Librosa. | ⬚ |
| 03. Speech-to-Text (ASR) | Using Whisper for transcription. | ⬚ |
| 04. Text-to-Speech (TTS) | Generating human-like audio. | ⬚ |
| 05. Audio Classification | Detecting environmental sounds or speaker emotion. | ⬚ |

## Code Example: Audio Feature Extraction (Librosa)

To feed audio into a neural network, we usually convert the raw waveform into a visual representation called a Spectrogram.

```python
# pip install librosa matplotlib numpy
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_spectrogram(audio_path):
    # 1. Load the audio file
    # sr=None preserves original sampling rate
    y, sr = librosa.load(audio_path, sr=None) 
    
    # 2. Extract Mel-frequency cepstral coefficients (MFCCs)
    # These are the standard features used in speech recognition
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # 3. Visualize the features
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC (Audio Features)')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('mfcc_plot.png')
    print("Spectrogram saved as mfcc_plot.png")

# Usage:
# plot_spectrogram("sample_audio.wav")
```