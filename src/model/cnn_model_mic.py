"""
CNN model za klasifikacijo zvoka iz mikrofonskih spektrogramov.

Model ima eno konvolucijsko vejo, ki iz STFT spektrograma izlušči
značilke, nato pa jih klasifikator razvrsti v enega od razredov
(npr. glasba ali pogovor).
"""

import torch
import torch.nn as nn


class Branch(nn.Module):
    """
    Konvolucijska veja za mikrofonski spektrogram.

    Iz enokanalnega spektrograma izlušči značilke in jih splošči v
    1D vektor.
    """

    # v vhod pride tak (batch, 1, 129, 62)
    # 62 je cas, 129 je freq_bins, 1 ker je mono, batch -> stevilo slik
    def __init__(self):
        """
        Sestavi konvolucijsko mrežo veje.
        """
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
        """
        Forward pass skozi vejo.

        Parameters:
            x (Tensor):
                Mikrofonski spektrogram oblike (batch, 1, freq_bins, time).

        Returns:
            Tensor:
                1D feature vektor.
        """
        return self.network(x)


class CNNModel(nn.Module):
    """
    Glavni CNN model za klasifikacijo mikrofonskega zvoka.

    Spektrogram pošlje skozi konvolucijsko vejo, nato pa značilke
    klasifikator razvrsti v enega od razredov.
    """

    def __init__(self, num_classes):
        """
        Inicializacija modela.

        Parameters:
            num_classes (int):
                Število razredov zvoka.
        """
        super().__init__()

        self.mic_branch = Branch()

        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, mic):
        """
        Forward pass skozi celoten model.

        Parameters:
            mic (Tensor):
                Batch mikrofonskih spektrogramov.

        Returns:
            Tensor:
                Score za posamezne razrede.
        """
        mic_features = self.mic_branch(mic)

        return self.classifier(mic_features)
