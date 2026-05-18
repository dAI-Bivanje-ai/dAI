import numpy as np
import torch

from src.model.cnn_model import CNNModel
from src.model.dataset_cnn import IMUDataset
from src.preprocessing.dataset_builder import build_dataset
from src.preprocessing.dataset_builder import build_dataset, SEGMENT_LENGTH

# pot do shranjenega dict_map -a
MODEL_PATH = "models/imu_cnn.pt"

# začasno shranimo predobdelano podatke v .npz
TEMP_DATASET_PATH = "temp_predict_dataset.npz"

# pretvorba številčnega razreda v ime aktivnosti
CLASS_NAMES = {
    0: "DELO",
    1: "TELEFON",
}


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

    # y se ignorira ker pri predictingu ne rabimo labelov
    X_acc, X_gyro, y = build_dataset(files)

    # začasno hrani X_acc, X_gyro in y v temp. dataset
    np.savez(
        TEMP_DATASET_PATH,
        X_acc=X_acc,
        X_gyro=X_gyro,
        y=y,
    )

    # pretvori začasni dataset -> tensorje
    dataset = IMUDataset(TEMP_DATASET_PATH)

    # ustvari isto arhitekturo kot pri treningu
    model = CNNModel(num_classes=2)

    # naloži shranjene uteži
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    # model preklopi v eval mode
    model.eval()

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
            total_sec = i * SEGMENT_LENGTH
            minutes = total_sec // 60
            seconds = total_sec % 60

            print(f"{minutes:02d}:{seconds:02d}  {CLASS_NAMES[pred]}")

        # izpis napovedi
        print(" ".join(pred_text))


if __name__ == "__main__":
    # nastavitev poti do mešane seje
    predict_sesh("src/data_logger/seja.bin")
