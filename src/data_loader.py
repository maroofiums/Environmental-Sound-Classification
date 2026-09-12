from torch.utils.data import DataLoader

from src.config import (
    AUDIO_DIR,
    METADATA_PATH,
    SAMPLE_RATE,
    NUM_SAMPLES,
    BATCH_SIZE,
    NUM_WORKERS,
)

from src.dataset import ESC50Dataset

def create_dataloaders():

    train_dataset = ESC50Dataset(
        audio_dir=AUDIO_DIR,
        metadata_path=METADATA_PATH,
        folds=[1, 2, 3],
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES
    )

    validation_dataset = ESC50Dataset(
        audio_dir=AUDIO_DIR,
        metadata_path=METADATA_PATH,
        folds=[4],
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES
    )

    test_dataset = ESC50Dataset(
        audio_dir=AUDIO_DIR,
        metadata_path=METADATA_PATH,
        folds=[5],
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        # pin_memory=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        # pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        # pin_memory=True
    )


    return train_loader, validation_loader, test_loader