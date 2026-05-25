import torch
import torch.nn as nn


class SensorBranch(nn.Module):
    """
    Ena CNN veja za en senzor.

    Vsaka veja iz spektrograma izlušči značilke (features),
    ki opuisujejo gibanje uporabnika.
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            # prva konvolucija prejme 3 kanale x,y,z
            # ustvari 16 feature map - odziv filtra na vhod
            # filter velikosti 3x3
            # padding doda rob okoli slike
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            # odstrani negativne vrednosti in doda nelinearnost
            nn.ReLU(),
            # zmanjša dimenzije spektograma za prb. 2x
            nn.MaxPool2d(2),
            # druga konvolucija iz 16 feature map naredi 32 kompleksnejših FM.
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # ne glede velikost vhoda mreža na koncu vrne (32, 4, 4)
            nn.AdaptiveAvgPool2d((4, 4)),
            # pretvori feature map z naučenimi značilkami senzorja v 1D vektor
            nn.Flatten(),
        )

    def forward(self, x):
        """
        Forward pass skozi eno CNN vejo.

        Parameters:
            x (Tensor):
                Oblika:
                (batch, channels, freq_bins, time)

        Returns:
            Tensor:
                1D feature vektor za posamezen senzor.
        """
        return self.network(x)


class CNNModel(nn.Module):
    """
    Glavni CNN model z dvema vhodoma.

    Model uporablja:
    - ACC vejo
    - GYRO vejo

    Obe veji ločeno analizirata spektrograme,
    nato pa njune značilke združi.
    """

    def __init__(self, num_classes):
        """
        Inicializacija celotnega modela.

        Parameters:
            num_classes (int):
                Število aktivnosti / razredov.
        """
        super().__init__()

        self.acc_branch = SensorBranch()
        self.gyro_branch = SensorBranch()

        # iz značilk napove aktivnost
        self.classifier = nn.Sequential(
            # Fully connected layer
            # Input: 1024 značilk
            # Output: 64 značilk
            nn.Linear(32 * 4 * 4 * 2, 64),
            nn.ReLU(),
            # 30% nevronov se naključno ignorira
            nn.Dropout(0.3),
            # Output layer
            # Vrne rezultat za vsak razred aktivnosti
            nn.Linear(64, num_classes),
        )

    def forward(self, acc, gyro):
        """
        Forward pass skozi celoten model.

        Parameters:
            acc (Tensor):
                ACC spektrogram batch.

            gyro (Tensor):
                GYRO spektrogram batch.

        Returns:
            Tensor:
                Score za posamezne razrede.
        """
        acc_features = self.acc_branch(acc)
        gyro_features = self.gyro_branch(gyro)

        # torch.cat zlepi tensorje skupaj, dim = 1 po feature dimenziji
        combined = torch.cat(
            [acc_features, gyro_features],
            dim=1,
        )
        # iz združenih značilk classifier napove aktivnost
        return self.classifier(combined)
