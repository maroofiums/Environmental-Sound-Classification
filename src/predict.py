import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import (
    NUM_CLASSES,
    CHECKPOINT_DIR,
)

from src.data_loader import create_dataloaders
from src.models import AudioCNN
from src.transforms import LogMelSpectrogramTransform


def evaluate_model(model, test_loader, loss_fn, device):
    """
    Evaluate the model on the test dataset.
    """

    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(
            test_loader,
            desc="Testing",
        )

        for waveforms, labels in progress_bar:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            outputs = model(waveforms)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * waveforms.size(0)

            predictions = outputs.argmax(dim=1)

            correct_predictions += (
                (predictions == labels).sum().item()
            )

            total_samples += labels.size(0)

            current_accuracy = (
                correct_predictions / total_samples
            ) * 100

            progress_bar.set_postfix(
                accuracy=f"{current_accuracy:.2f}%"
            )

    average_loss = running_loss / total_samples
    accuracy = (
        correct_predictions / total_samples
    ) * 100

    return average_loss, accuracy


def main():
    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Create the same transform used during training
    transform = LogMelSpectrogramTransform(
        sample_rate=44_100,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
    )

    # We only need the test DataLoader
    _, _, test_loader = create_dataloaders(
        transform=transform
    )

    # Create model architecture
    model = AudioCNN(
        num_classes=NUM_CLASSES
    ).to(device)

    # Checkpoint path
    checkpoint_path = (
        CHECKPOINT_DIR / "best_audio_cnn.pt"
    )

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation accuracy: "
        f"{checkpoint['validation_accuracy']:.2f}%"
    )

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Evaluate
    test_loss, test_accuracy = evaluate_model(
        model=model,
        test_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print("\nTest Results")
    print("-" * 30)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()