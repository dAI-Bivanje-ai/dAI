import torch.nn as nn


class CNNModel(nn.Module):
    """
    Osnovni CNN model za klasifikacijo IMU spektrogramov.
    """

    def __init__(self, num_classes):
        """
        Inicializacija arhitekture nevronske mreže.

        Parameters:
            num_classes (int): Število izhodnih razredov/aktivnosti.
        """

        super().__init__()

        # Sequential omogoča izvajanje slojev po vrstnem redu.
        self.network = nn.Sequential(
            # Convolution layer išče lokalne vzorce v spektrogramu.
            # Input channel = 1 (en spektrogram)
            # Output channel = 16 feature map.
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            # ReLU uvede nelinearnost v mrežo.
            nn.ReLU(),
            # Pooling zmanjša dimenzije podatkov
            # in zmanjša količino računanja.
            nn.MaxPool2d(2),
            # Pretvorba 2D feature map v 1D vektor.
            nn.Flatten(),
            # Fully connected layer.
            # LazyLinear sam določi vhodno dimenzijo.
            nn.LazyLinear(64),
            # Dodatna nelinearnost.
            nn.ReLU(),
            # Output layer vrne score za vsak razred.
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Forward pass skozi mrežo.

        Parameters:
            x (Tensor): Vhodni spektrogram.

        Returns:
            Tensor: Izhodni score za posamezne razrede.
        """

        return self.network(x)
