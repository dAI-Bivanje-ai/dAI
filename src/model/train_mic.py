"""
Učenje mikrofonskega CNN modela za klasifikacijo zvoka (glasba / pogovor).

Modul iz seznama sej zgradi učni in validacijski dataset, izvede učno zanko
čez več epoh, sproti beleži izgubo in točnost ter na koncu shrani naučene
uteži (skupaj z normalizacijskimi parametri) in zgodovino učenja.
"""

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

# glasba, pogovorsssss
NUM_CLASSES = 2

BATCH_SIZE = 16

EPOCHS = 50

LEARNING_RATE = 0.0001

TRAIN_FILES = [
    (str(ROOT_DIR / "podatki/mic_podatki/glasba_01.bin"), 0),
    (str(ROOT_DIR / "podatki/mic_podatki/glasba_02.bin"), 0),
    (str(ROOT_DIR / "podatki/mic_podatki/glasba_05.bin"), 0),
    (str(ROOT_DIR / "podatki/mic_podatki/glasba_06.bin"), 0),
    (str(ROOT_DIR / "podatki/mic_podatki/glasba_07.bin"), 0),
    (str(ROOT_DIR / "podatki/mic_podatki/glasba_08.bin"), 0),
    (str(ROOT_DIR / "podatki/mic_podatki/glasba_11.bin"), 0),
    (str(ROOT_DIR / "podatki/mic_podatki/pogovor_02.bin"), 1),
    (str(ROOT_DIR / "podatki/mic_podatki/pogovor_03.bin"), 1),
    (str(ROOT_DIR / "podatki/mic_podatki/pogovor_05.bin"), 1),
    (str(ROOT_DIR / "podatki/mic_podatki/pogovor_06.bin"), 1),
    (str(ROOT_DIR / "podatki/mic_podatki/pogovor_07.bin"), 1),
    (str(ROOT_DIR / "podatki/mic_podatki/pogovor_09.bin"), 1),
    (str(ROOT_DIR / "podatki/mic_podatki/pogovor_10.bin"), 1),
    (str(ROOT_DIR / "podatki/mic_podatki/pogovor_11.bin"), 1),
    (str(ROOT_DIR / "podatki/mic_podatki/pogovor_12.bin"), 1),
    (str(ROOT_DIR / "podatki/mic_podatki/pogovor_13.bin"), 1),
]

VAL_FILES = [
    (str(ROOT_DIR / "podatki/mic_podatki/glasba_03.bin"), 0),
    (str(ROOT_DIR / "podatki/mic_podatki/glasba_04.bin"), 0),
    (str(ROOT_DIR / "podatki/mic_podatki/glasba_09.bin"), 0),
    (str(ROOT_DIR / "podatki/mic_podatki/glasba_10.bin"), 0),
    (str(ROOT_DIR / "podatki/mic_podatki/pogovor_04.bin"), 1),
    (str(ROOT_DIR / "podatki/mic_podatki/pogovor_14.bin"), 1),
]

TRAIN_NPZ = str(ROOT_DIR / "train_dataset_mic.npz")
VAL_NPZ = str(ROOT_DIR / "val_dataset_mic.npz")


def train():
    """
    Nauči mikrofonski CNN model in shrani uteži ter zgodovino učenja.

    Zgradi dataset iz TRAIN_FILES in VAL_FILES, izvede učno zanko čez EPOCHS
    epoh, sproti izpisuje izgubo in točnost ter na koncu shrani model in
    normalizacijska parametra (log_min, log_max) v models/mic_cnn.pt,
    zgodovino pa v models/history_mic.json.
    """
    X_mic_train, y_train, log_min, log_max = build_dataset_mic(TRAIN_FILES)
    X_mic_val, y_val, _, _ = build_dataset_mic(VAL_FILES)

    np.savez(TRAIN_NPZ, X=X_mic_train, y=y_train)
    np.savez(VAL_NPZ, X=X_mic_val, y=y_val)

    train_dataset_mic = MicDataset(TRAIN_NPZ)
    val_dataset_mic = MicDataset(VAL_NPZ)

    train_loader_mic = DataLoader(
        train_dataset_mic, batch_size=BATCH_SIZE, shuffle=True
    )

    val_loader_mic = DataLoader(val_dataset_mic, batch_size=BATCH_SIZE, shuffle=False)

    model = CNNModel(num_classes=NUM_CLASSES)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for sample, y in train_loader_mic:
            optimizer.zero_grad()

            outputs = model(sample)

            loss = loss_fn(outputs, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == y).sum().item()

            total += y.size(0)

        train_acc = correct / total
        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for sample, y in val_loader_mic:

                outputs = model(sample)

                predictions = outputs.argmax(dim=1)

                val_correct += (predictions == y).sum().item()

                val_total += y.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0

        history["train_loss"].append(total_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"loss={total_loss:.4f} | "
            f"train_acc={train_acc:.3f} | "
            f"val_acc={val_acc:.3f}"
        )

    (ROOT_DIR / "models").mkdir(exist_ok=True)

    with open(ROOT_DIR / "models" / "history_mic.json", "w") as f:
        json.dump(history, f)

    torch.save(
        {
            "model": model.state_dict(),
            "log_min": log_min,
            "log_max": log_max,
        },
        str(ROOT_DIR / "models" / "mic_cnn.pt"),
    )

    print(f"Model shranjen v {ROOT_DIR / 'models' / 'mic_cnn.pt'}")


if __name__ == "__main__":
    train()
