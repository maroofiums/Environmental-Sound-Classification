import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    NUM_CLASSES,
    LEARNING_RATE,
    NUM_EPOCHS,
    CHECKPOINT_DIR,
    SAMPLE_RATE
)

from src.augmentations import (
    ComposeAudio,
    AddGaussianNoise,
    RandomGain,
    RandomTimeShift
)

from src.data_loader import create_dataloaders
from src.models import AudioCNN
from src.transforms import LogMelSpectrogramTransform


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for inputs, labels in progress_bar:

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix(
            loss=running_loss / total,
            accuracy=100.0 * correct / total
        )

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

        for inputs, labels in progress_bar:

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(
                loss=running_loss / total,
                accuracy=100.0 * correct / total
            )

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy



def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    train_augmentation = ComposeAudio(
        [
            RandomGain(
                gain_range=(0.8, 1.2),
                probability=0.5,
            ),
            AddGaussianNoise(
                noise_factor=0.005,
                probability=0.5
            ),
            RandomTimeShift(
                max_shift_seconds=0.5,
                sample_rate=SAMPLE_RATE,
                probability=0.5
            )
        ]
    )

    spectrogram_transform = LogMelSpectrogramTransform(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
    )

    train_loader, validation_loader, test_loader = create_dataloaders(
        train_waveform_transform=train_augmentation,
        spectrogram_transform=spectrogram_transform
    )

    model = AudioCNN(
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_accuracy = 0.0

    checkpoint_path = CHECKPOINT_DIR / "best_audio_cnn.pt"

    for epoch in range(NUM_EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
        )

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f}"
        )

        print(
            f"Validation Loss: {validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": validation_accuracy,
                },
                checkpoint_path,
            )

            print(
                f"Saved best model to {checkpoint_path}"
            )

    print("\nTraining completed.")


if __name__ == "__main__":
    main()