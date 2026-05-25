from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader

from src.model.cnn_model_mic import CNNModel
from src.model.dataset_cnn_mic import MicDataset
from src.preprocessing.dataset_builder_mic import build_dataset_mic


# glasba, pogovor
NUM_CLASSES = 2

BATCH_SIZE = 16

EPOCHS = 50

LEARNING_RATE = 0.001

TRAIN_FILES = [
    (str(ROOT_DIR / "podatki/mic_podatki/glasba_01.bin"), 0),
    (str(ROOT_DIR / "podatki/mic_podatki/glasba_02.bin"), 0),
    (str(ROOT_DIR / "podatki/mic_podatki/glasba_03.bin"), 0),
    (str(ROOT_DIR / "podatki/mic_podatki/glasba_04.bin"), 0),
    (str(ROOT_DIR / "podatki/mic_podatki/glasba_05.bin"), 0),
    (str(ROOT_DIR / "podatki/mic_podatki/pogovor_01.bin"), 1),
    (str(ROOT_DIR / "podatki/mic_podatki/pogovor_02.bin"), 1),
    (str(ROOT_DIR / "podatki/mic_podatki/pogovor_03.bin"), 1),
    (str(ROOT_DIR / "podatki/mic_podatki/pogovor_04.bin"), 1),
    (str(ROOT_DIR / "podatki/mic_podatki/pogovor_05.bin"), 1),
]

VAL_FILES = [
    (str(ROOT_DIR / "podatki/delo_podatki/delo_06.bin"), 0),
    (str(ROOT_DIR / "podatki/delo_podatki/delo_07.bin"), 0),
    (str(ROOT_DIR / "podatki/mic_podatki/pogovor_06.bin"), 1),
    (str(ROOT_DIR / "podatki/mic_podatki/pogovor_07.bin"), 1),
]

TRAIN_NPZ = str(ROOT_DIR / "train_dataset_mic.npz")
VAL_NPZ = str(ROOT_DIR / "val_dataset_mic.npz")


def train():

    X_mic_train, y_train = build_dataset_mic(TRAIN_FILES)
    X_mic_val, y_val = build_dataset_mic(VAL_FILES)

    np.savez(TRAIN_NPZ, X_mic=X_mic_train, y_train=y_train)
    np.savez(VAL_NPZ, X_mic=X_mic_val, y=y_val)
