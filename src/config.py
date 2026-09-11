from pathlib import Path


# Project Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ESC-50"
AUDIO_DIR = DATA_DIR / "audio"
METADATA_PATH = DATA_DIR / "meta" / "esc50.csv"


# Dataset Configuration

NUM_CLASSES = 50
SAMPLE_RATE = 44_100
AUDIO_DURATION = 5
NUM_SAMPLES = SAMPLE_RATE * AUDIO_DURATION


# Training Configuration

BATCH_SIZE = 16
NUM_WORKERS = 2
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10


# Checkpoints

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)