from src.config import (
    AUDIO_DIR,
    METADATA_PATH,
    SAMPLE_RATE,
    NUM_SAMPLES,
)

from src.dataset import ESC50Dataset


dataset = ESC50Dataset(
    audio_dir=AUDIO_DIR,
    metadata_path=METADATA_PATH,
    folds=[1, 2, 3],
    target_sample_rate=SAMPLE_RATE,
    num_samples=NUM_SAMPLES,
)

print("Dataset size:", len(dataset))

waveform, label = dataset[0]

print("Waveform shape:", waveform.shape)
print("Label:", label)
print("Waveform dtype:", waveform.dtype)