"""
PyTorch dataset za mikrofonske spektrograme.

Modul naloži predobdelane spektrograme in oznake iz .npz datoteke ter jih
pretvori v PyTorch tensorje v obliki, primerni za CNN model.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class MicDataset(Dataset):
    """
    PyTorch dataset za nalaganje mikrofonskih spektrogramov iz .npz datoteke.

    Dataset vrne:
        (mic_spectrogram, label)
    """

    def __init__(self, dataset_path):
        """
        Naloži spektrograme in labele iz dataset datoteke.

        Parameters:
            dataset_path (str): Pot do .npz dataset datoteke.
        """
        data = np.load(dataset_path)

        x = data["X"]
        y = data["y"]

        x = x.astype(np.float32)
        y = y.astype(np.int64)

        # treba dodat 1, da bo prave oblike
        x = np.expand_dims(x, axis=1)

        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self):
        """
        Vrne število učnih primerov v datasetu.
        """
        return len(self.y)

    def __getitem__(self, index):
        """
        Vrne en učni primer.

        Returns:
            (Tensor, Tensor) — (mikrofonski spektrogram, label)
        """
        return (self.x[index], self.y[index])
