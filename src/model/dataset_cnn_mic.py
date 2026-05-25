import numpy as np
import torch
from torch.utils.data import Dataset


class IMUDataset(Dataset):

    def __init__(self, dataset_path):

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
        return len(self.y)

    def __getitem__(self, index):

        return (self.x[index], self.y[index])
