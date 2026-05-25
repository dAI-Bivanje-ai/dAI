import torch
import torch.nn as nn


class Branch(nn.Module):

    # v vhod pride tak (batch, 1, 129, 62)
    # 62 je cas, 129 je freq_bins, 1 ker je mono, batch -> stevilo slik
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            # zaradi velikega dinamičnega razpona je treba normalizirat
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),  # (512), ker je en
        )

    def forward(self, x):
        return self.network(x)


class CNNModel(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.mic_branch = Branch()

        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, mic):
        mic_features = self.mic_branch(mic)

        return self.classifier(mic_features)
