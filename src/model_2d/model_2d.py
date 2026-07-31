import torch
import torch.nn as nn


class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=5, input_channels=1, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


if __name__ == "__main__":
    model = SpectrogramCNN(num_classes=5)
    dummy_input = torch.randn(4, 1, 128, 128)
    output = model(dummy_input)
    print("Input shape: ", tuple(dummy_input.shape))
    print("Output shape:", tuple(output.shape))
    print("Params:     ", sum(p.numel() for p in model.parameters()))
