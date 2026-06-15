from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader

from src.model.cnn_model import CNNModel
from src.model.dataset_cnn import IMUDataset
from src.preprocessing.dataset_builder import build_dataset

# večinoma sej je lokalno na računalniku, ne bomo pushali
TRAIN_FILES = [
    (str(ROOT_DIR / "podatki/delo_podatki/delo_01.bin"), 0),
    (str(ROOT_DIR / "podatki/delo_podatki/delo_02.bin"), 0),
    (str(ROOT_DIR / "podatki/delo_podatki/delo_04.bin"), 0),
    (str(ROOT_DIR / "podatki/delo_podatki/delo_05.bin"), 0),
    (str(ROOT_DIR / "podatki/telefon_podatki/telefon_01.bin"), 1),
    (str(ROOT_DIR / "podatki/telefon_podatki/telefon_02.bin"), 1),
    (str(ROOT_DIR / "podatki/telefon_podatki/telefon_04.bin"), 1),
    (str(ROOT_DIR / "podatki/telefon_podatki/telefon_05.bin"), 1),
]

VAL_FILES = [
    (str(ROOT_DIR / "podatki/delo_podatki/delo_03.bin"), 0),
    (str(ROOT_DIR / "podatki/telefon_podatki/telefon_03.bin"), 1),
]


# trenutno 2 razreda delo/telefon
NUM_CLASSES = 2
# uteži se posodobijo po 16 primerih
BATCH_SIZE = 16
# prehodi čez training podatke
EPOCHS = 25
# velikost koraka pri popravljanju uteži
# 0.001 je običajna začetna vrednost za Adam opt.
LEARNING_RATE = 0.001

TRAIN_NPZ = str(ROOT_DIR / "train_dataset.npz")
VAL_NPZ = str(ROOT_DIR / "val_dataset.npz")


def train():
    # uporabimo NVIDIA CUDA GPU, drugače CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # shranjevanje podatkov za kasnejšo vizualizacijo
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    # nalozi dataset.npz v PyTorch dataset
    X_acc_train, X_gyro_train, y_train = build_dataset(TRAIN_FILES)
    X_acc_val, X_gyro_val, y_val = build_dataset(VAL_FILES)

    np.savez(TRAIN_NPZ, X_acc=X_acc_train, X_gyro=X_gyro_train, y=y_train)
    np.savez(VAL_NPZ, X_acc=X_acc_val, X_gyro=X_gyro_val, y=y_val)

    train_dataset = IMUDataset(TRAIN_NPZ)
    val_dataset = IMUDataset(VAL_NPZ)

    # DataLoader iz dataseta dela batch-e
    # shuffle = True , učni primeri se vsako epoho premešajo
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # pri validaciji ni potrebnega mešanja, ker model samo preverjamo
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # ustvari cnn model
    # NUM_CLASSES = 2 model bo imel 2 outputa - score za delo/telfon
    model = CNNModel(num_classes=NUM_CLASSES).to(device)

    # standardna loss funkcija za klasifikacijo
    # primerja output modela z dejanskim label-om
    loss_fn = nn.CrossEntropyLoss()

    # Adam opt. popravlja uteži
    # model.parameters() - vse uteži in biasi v cnn
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # glavna učna zanka
    for epoch in range(EPOCHS):
        # model nstavimo na train mode zaradi Dropout sloja, ki je aktiven samo med učenjem
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        # gre čez vse batche učnih podatkov
        for (acc, gyro), y in train_loader:

            # Podatke prestavimo na isto napravo kot model.
            acc = acc.to(device)
            gyro = gyro.to(device)
            y = y.to(device)

            # pobriše stare gradiente
            optimizer.zero_grad()

            # Forward pass
            # izrčun score-a za vsak razred
            outputs = model(acc, gyro)

            # izračun napake med napovedjo in pravilnim rezultatom
            loss = loss_fn(outputs, y)

            # Backward pass
            # koliko mora spremeniti vsako utež, da se izguba zmanjša
            loss.backward()

            # posodobi uteži
            optimizer.step()

            # hranjenje izgube za izpis
            total_loss += loss.item()

            # izbere razred z najvišjim rezultatom
            # (batch_size, num_classes)
            predictions = outputs.argmax(dim=1)

            # prešteje pravilne napovedi
            correct += (predictions == y).sum().item()

            # prešteje vse primere
            total += y.size(0)

        # natančnost na učnih podatkih
        train_acc = correct / total

        # model nastavimo na eval mode
        # Dropout se izklopi
        model.eval()

        val_correct = 0
        val_total = 0

        # pri validaciji se gradient ne računa
        with torch.no_grad():
            # prehod čez vse validation batch-e
            for (acc, gyro), y in val_loader:
                # Tudi pri validaciji morajo biti podatki na isti napravi kot model.
                acc = acc.to(device)
                gyro = gyro.to(device)
                y = y.to(device)

                # model naredi napoved
                outputs = model(acc, gyro)

                # izbira razreda z najvišjim rezultatom
                predictions = outputs.argmax(dim=1)

                # prešteje pravilne napovedi
                val_correct += (predictions == y).sum().item()

                # prešteje vse val. primere
                val_total += y.size(0)

        # natančnost na validation podatkih
        val_acc = val_correct / val_total if val_total > 0 else 0

        history["train_loss"].append(total_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        # izpis rezultata za vsako epoho
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"loss={total_loss:.4f} | "
            f"train_acc={train_acc:.3f} | "
            f"val_acc={val_acc:.3f}"
        )
    # ustvari mapo models
    (ROOT_DIR / "models").mkdir(exist_ok=True)

    with open(ROOT_DIR / "models" / "history.json", "w") as f:
        json.dump(history, f)

    # shrani naučene uteži modela -> state_dict - slovar vseh naučenih parametrov
    torch.save(
        model.state_dict(),
        str(ROOT_DIR / "models" / "imu_cnn.pt"),
    )

    print(f"Model shranjen v {ROOT_DIR / 'models' / 'imu_cnn.pt'}")


if __name__ == "__main__":
    train()
