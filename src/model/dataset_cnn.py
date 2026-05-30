import numpy as np
import torch
from torch.utils.data import Dataset


class IMUDataset(Dataset):
    """
    PyTorch dataset za nalaganje ACC in GYRO spektrogramov iz .npz datoteke.

    Dataset vrne:
        ((acc_spectrogram, gyro_spectrogram), label)

    ACC in GYRO predstavljata dva ločena vhoda v CNN model.
    """

    def __init__(self, dataset_path):
        """
        Naloži spektrograme in labele iz dataset datoteke.

        Parameters:
            dataset_path (str): Pot do .npz dataset datoteke.
        """

        data = np.load(dataset_path)

        # (N, freq_bins, segment_lenght, 3 channels)
        x_acc = data["X_acc"]
        x_gyro = data["X_gyro"]
        y = data["y"]

        # pretvorba tipov za PyTorch
        # float32 za vhodne podatke, int64 za klasifikacijske oznake
        x_acc = x_acc.astype(np.float32)
        x_gyro = x_gyro.astype(np.float32)
        y = y.astype(np.int64)

        # CNN oblika (N, channels = 3, freq_bins = height, segment_lenght = width).
        # spremeni dimenzije | zamenjava 1. dimenzije z 3.jo
        x_acc = np.transpose(x_acc, (0, 3, 1, 2))
        x_gyro = np.transpose(x_gyro, (0, 3, 1, 2))

        # pretvorba array - ev v PyTorch tensorje
        self.x_acc = torch.tensor(x_acc)
        self.x_gyro = torch.tensor(x_gyro)
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
            ((Tensor, Tensor), Tensor)

            (
                (ACC_spektrogram, GYRO_spektrogram),
                label
            )

        ACC in GYRO sta ločena vhoda v model.
        """
        return (self.x_acc[index], self.x_gyro[index]), self.y[index]
