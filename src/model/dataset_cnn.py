import numpy as np
import torch
from torch.utils.data import Dataset


class IMUDataset(Dataset):
    """
    PyTorch dataset za nalaganje IMU spektrogramov iz .npz datoteke.
    """

    def __init__(self, dataset_path, sensor="acc"):
        data = np.load(dataset_path)

        if sensor == "acc":
            x = data["X_acc"]
        elif sensor == "gyro":
            x = data["X_gyro"]
        else:
            raise ValueError("sensor must be 'acc' or 'gyro'")

        y = data["y"]

        # Spektrogrami so shranjeni kot 0–255, za model jih damo v območje 0–1.
        x = x.astype(np.float32) / 255.0
        y = y.astype(np.int64)

        # CNN pričakuje obliko (N, C, H, W).
        # dataset ima (N, freq_bins, 3), zato dodamo kanal C = 1.
        x = np.expand_dims(x, axis=1)

        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
