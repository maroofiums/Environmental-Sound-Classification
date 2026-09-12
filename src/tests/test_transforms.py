import torch

from src.config import SAMPLE_RATE
from src.transforms import (
    MelSpectrogramTransform,
    LogMelSpectrogramTransform,
    SpectrogramToImage
)

waveform = torch.randn(
    1,
    220500,
)

mel_transform = MelSpectrogramTransform(
    sample_rate=SAMPLE_RATE
)

mel = mel_transform(waveform)

print("Shape of Waveform: ", waveform.shape)
print("Shape of Mel Spectrogram: ", mel.shape)

log_mel_transform = LogMelSpectrogramTransform(
    sample_rate=SAMPLE_RATE
)

log_mel = log_mel_transform(waveform)

print("Shape of Log Mel Spectrogram: ", log_mel.shape)

image_transform = SpectrogramToImage()

image = image_transform(log_mel)

print("Image: ", image.shape)