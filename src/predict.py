from pathlib import Path

import torch
import torchaudio

from src.config import (
    NUM_CLASSES,
    SAMPLE_RATE,
    NUM_SAMPLES,
    CHECKPOINT_DIR,
)

from src.models import AudioCNN
from src.transforms import LogMelSpectrogramTransform


# ESC-50 class names
CLASS_NAMES = [
    "dog",
    "rooster",
    "pig",
    "cow",
    "frog",
    "cat",
    "hen",
    "insects",
    "sheep",
    "crow",
    "rain",
    "sea_waves",
    "crackling_fire",
    "crickets",
    "chirping_birds",
    "water_drops",
    "wind",
    "pouring_water",
    "toilet_flush",
    "thunderstorm",
    "crying_baby",
    "sneezing",
    "clapping",
    "breathing",
    "coughing",
    "footsteps",
    "laughing",
    "brushing_teeth",
    "snoring",
    "drinking_sipping",
    "door_wood_knock",
    "mouse_click",
    "keyboard_typing",
    "door_wood_creaks",
    "can_opening",
    "washing_machine",
    "vacuum_cleaner",
    "clock_alarm",
    "clock_tick",
    "glass_breaking",
    "helicopter",
    "chainsaw",
    "siren",
    "car_horn",
    "engine",
    "train",
    "church_bells",
    "airplane",
    "fireworks",
    "hand_saw",
]


def load_audio(
    audio_path,
    target_sample_rate=SAMPLE_RATE,
    num_samples=NUM_SAMPLES,
):
    """
    Load an audio file and prepare it for the model.
    """

    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo/multi-channel audio to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )

    # Resample if necessary
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )

        waveform = resampler(waveform)

    # Pad or crop to the expected length
    current_length = waveform.shape[-1]

    if current_length < num_samples:
        padding_amount = num_samples - current_length

        waveform = torch.nn.functional.pad(
            waveform,
            (0, padding_amount),
        )

    elif current_length > num_samples:
        waveform = waveform[..., :num_samples]

    return waveform


def load_model(device):
    """
    Create the model and load the best checkpoint.
    """

    model = AudioCNN(
        num_classes=NUM_CLASSES
    ).to(device)

    checkpoint_path = (
        CHECKPOINT_DIR / "best_audio_cnn.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation accuracy: "
        f"{checkpoint['validation_accuracy']:.2f}%"
    )

    return model



def predict(
    model,
    audio_path,
    transform,
    device,
    top_k=5,
):
    """
    Predict the top-k classes for one audio file.
    """

    waveform = load_audio(audio_path)

    # Convert waveform into log-mel spectrogram
    spectrogram = transform(waveform)

    # [1, 128, time] -> [1, 1, 128, time]
    spectrogram = spectrogram.unsqueeze(0)
    spectrogram = spectrogram.to(device)

    with torch.no_grad():
        outputs = model(spectrogram)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        top_probabilities, top_indices = (
            torch.topk(
                probabilities,
                k=top_k,
                dim=1,
            )
        )

    top_probabilities = top_probabilities[0].cpu()
    top_indices = top_indices[0].cpu()

    predictions = []

    for probability, index in zip(
        top_probabilities,
        top_indices,
    ):
        predictions.append(
            {
                "class": CLASS_NAMES[index.item()],
                "confidence": probability.item() * 100,
            }
        )

    return predictions



def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Change this path to the audio file you want to classify
    audio_path = (
        Path("data/ESC-50/audio/1-100032-A-0.wav")
    )

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    transform = LogMelSpectrogramTransform(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
    )

    model = load_model(device)

    predictions = predict(
        model=model,
        audio_path=audio_path,
        transform=transform,
        device=device,
        top_k=5,
    )

    print("\nTop Predictions")
    print("-" * 30)

    for rank, prediction in enumerate(
        predictions,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{prediction['class']:<20} "
            f"{prediction['confidence']:.2f}%"
        )

if __name__ == "__main__":
    main()