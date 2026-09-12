import torch

from src.models import AudioCNN


model = AudioCNN(num_classes=50)

x = torch.randn(
    16,
    1,
    128,
    431,
)

output = model(x)

print("\nModel: ", model)
print("\nInput shape :", x.shape)
print("Output shape:", output.shape)