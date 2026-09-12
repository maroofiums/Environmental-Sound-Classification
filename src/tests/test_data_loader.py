from src.data_loader import create_dataloaders


def main():
    train_loader, validation_loader, test_loader = create_dataloaders()

    print("Number of training batches:", len(train_loader))
    print("Number of validation batches:", len(validation_loader))
    print("Number of test batches:", len(test_loader))

    waveforms, labels = next(iter(train_loader))

    print("\nBatch information:")
    print("Waveforms shape:", waveforms.shape)
    print("Labels shape:", labels.shape)
    print("Waveforms dtype:", waveforms.dtype)
    print("Labels dtype:", labels.dtype)


if __name__ == "__main__":
    main()