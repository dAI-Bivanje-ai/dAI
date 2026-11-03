"""
Napovedovanje aktivnosti iz IMU seje z naučenim CNN modelom.

Modul predobdela eno .bin sejo v spektrograme, naloži naučen IMU model in
za vsak segment izpiše napovedano aktivnost skupaj s časovno oznako.
"""

import numpy as np
import torch

from src.config import GYRO_FFT_WINDOW, IMU_CLASSES, IMU_MODEL_PATH, ROOT_DIR
from src.model.cnn_model import CNNModel, load_imu_checkpoint
from src.model.dataset_cnn import IMUDataset
from src.preprocessing.dataset_builder import build_dataset, image_seconds, load_session

TEMP_DATASET_PATH = ROOT_DIR / "temp_predict_dataset.npz"


def predict_sesh(bin_file):
    """
    Napove aktivnosti za eno .bin sejo.

    Funkcija:
    1. uporabi obstoječi dataset_builder preprocessing,
    2. naloži naučen CNN model,
    3. gre čez vse segmente seje,
    4. za vsak segment izpiše napoved aktivnosti.
    """

    files = [
        (bin_file, 0),
    ]

    model, log_ranges = load_imu_checkpoint(IMU_MODEL_PATH)
    # y se ignorira ker pri predictingu ne rabimo labelov
    X_acc, X_gyro, y, _ = build_dataset(files, log_ranges)

    # prava fvz giroskopa, da vemo, kdaj se začne vsaka slika (~2.01 s narazen)
    _, _, fvz_gyro, _ = load_session(bin_file)
    gyro_image_seconds = image_seconds(GYRO_FFT_WINDOW, fvz_gyro)

    # začasno hrani X_acc, X_gyro in y v temp. dataset
    np.savez(
        TEMP_DATASET_PATH,
        X_acc=X_acc,
        X_gyro=X_gyro,
        y=y,
    )

    # pretvori začasni dataset -> tensorje
    dataset = IMUDataset(TEMP_DATASET_PATH)

    pred_text = []

    with torch.no_grad():

        # prehod čez vse segmente v seji
        for i in range(len(dataset)):

            # uzamemo 1 segment
            # dobimo (3, freq, čas)
            (acc, gyro), _ = dataset[i]

            # model pričakuje (batch, channels=3, height=frekv, width=čas)
            # dodamo batch dimenzijo
            acc = acc.unsqueeze(0)
            gyro = gyro.unsqueeze(0)

            # izračun rezultata za vsak razred
            output = model(acc, gyro)

            # izbere razred z najvišjim rezultatom
            pred = output.argmax(dim=1).item()

            # številčni razred v besedo
            total_sec = int(i * gyro_image_seconds)  # dobimo natančen čas
            minutes = total_sec // 60
            seconds = total_sec % 60

            print(f"{minutes:02d}:{seconds:02d}  {IMU_CLASSES[pred]}")

        # izpis napovedi
        print(" ".join(pred_text))


if __name__ == "__main__":
    predict_sesh(str(ROOT_DIR / "seja.bin"))
