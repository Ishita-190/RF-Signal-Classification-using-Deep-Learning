import torch
import torch.nn as nn


class SignalCNN(nn.Module):
    def __init__(self, num_classes=5, input_channels=2, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(
                f"Expected input with 3 dimensions, got shape {tuple(x.shape)}"
            )

        if x.shape[1] != 2 and x.shape[-1] == 2:
            x = x.permute(0, 2, 1)
        elif x.shape[1] != 2:
            raise ValueError(
                "Expected input shape [batch_size, 2, sequence_length] or [batch_size, sequence_length, 2]"
            )

        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


if __name__ == "__main__":
    model = SignalCNN(num_classes=5)
    dummy_input = torch.randn(4, 512000, 2)
    output = model(dummy_input)
    print("Input shape:", tuple(dummy_input.shape))
    print("Output shape:", tuple(output.shape))
