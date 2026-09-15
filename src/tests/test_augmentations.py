import torch

from src.augmentations import (
    ComposeAudio,
    RandomGain,
    AddGaussianNoise,
    RandomTimeShift,
)

waveform = torch.randn(1, 220_500)

augmentations = ComposeAudio(
    [
        RandomGain(
            gain_range=(0.8, 1.2),
            probability=1.0,
        ),
        AddGaussianNoise(
            noise_factor=0.005,
            probability=1.0,
        ),
        RandomTimeShift(
            max_shift_seconds=0.5,
            sample_rate=44_100,
            probability=1.0,
        ),
    ]
)

augmented_waveform = augmentations(waveform)

print("Original shape:", waveform.shape)
print("Augmented shape:", augmented_waveform.shape)