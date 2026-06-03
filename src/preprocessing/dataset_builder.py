from pathlib import Path
import numpy as np
from src.visualization.data_visualizer import pripravi_pakete, sestavi_podatke
from src.data_logger.data_logger import DataLogger
from src.preprocessing.windower import window_signal_seconds
from src.preprocessing.stft import compute_spectrograms, group_spectrograms

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_session(bin_file):
    """
    Prebere eno .bin datoteko.
    Vrne (Fvz_acc, sig_acc, Fvz_gyro, sig_gyro).
    sig_acc.shape  = (N, 3)load_
    sig_gyro.shape = (N, 3)
    """

    logger = DataLogger()
    raw_packets = logger.parse_file(bin_file)

    packets = pripravi_pakete(raw_packets)

    fvz_gyro, gyro_raw = sestavi_podatke(packets, 1)
    fvz_acc, acc_raw = sestavi_podatke(packets, 2)

    sig_gyro = gyro_raw * 8.75e-3
    sig_acc = acc_raw * 1e-3
    # print(sig_gyro.shape)
    return (fvz_acc, sig_acc, fvz_gyro, sig_gyro)


SEGMENT_LENGTH = 2


def build_dataset(files):
    """
    Gre čez vse datoteke, za vsako pokliče load_session,
    windower in stft, doda labele.
    Vrne (X_acc, X_gyro, y).
    """
    all_acc = []
    all_gyro = []
    all_y = []

    for bin_file, label in files:
        fvz_acc, sig_acc, fvz_gyro, sig_gyro = load_session(bin_file)

        ACC_W = 49
        GYRO_W = 211

        windows_acc = window_signal_seconds(sig_acc, fvz_acc, force_W=ACC_W)
        windows_gyro = window_signal_seconds(sig_gyro, fvz_gyro, force_W=GYRO_W)

        spectrograms_acc = compute_spectrograms(windows_acc)
        spectrograms_gyro = compute_spectrograms(windows_gyro)

        M = min(spectrograms_acc.shape[0], spectrograms_gyro.shape[0])
        spectrograms_acc = spectrograms_acc[:M]
        spectrograms_gyro = spectrograms_gyro[:M]

        spectrograms_acc_2d = group_spectrograms(spectrograms_acc, SEGMENT_LENGTH)
        spectrograms_gyro_2d = group_spectrograms(spectrograms_gyro, SEGMENT_LENGTH)

        N = min(spectrograms_acc_2d.shape[0], spectrograms_gyro_2d.shape[0])
        spectrograms_acc_2d = spectrograms_acc_2d[:N]
        spectrograms_gyro_2d = spectrograms_gyro_2d[:N]

        labels = np.full(N, label)

        all_acc.append(spectrograms_acc_2d)
        all_gyro.append(spectrograms_gyro_2d)
        all_y.append(labels)

    X_acc = np.concatenate(all_acc, axis=0)
    X_gyro = np.concatenate(all_gyro, axis=0)
    y = np.concatenate(all_y, axis=0)

    # log normalizacija
    X_acc = np.log10(X_acc + 1e-10)
    X_gyro = np.log10(X_gyro + 1e-10)

    # skaliranje na [0 - 1]
    X_acc = (X_acc - X_acc.min()) / (X_acc.max() - X_acc.min())
    X_gyro = (X_gyro - X_gyro.min()) / (X_gyro.max() - X_gyro.min())

    return X_acc, X_gyro, y


def save_dataset(X_acc, X_gyro, y, filename="dataset.npz"):
    """
    Shrani dataset na disk.
    """
    np.savez(filename, X_acc=X_acc, X_gyro=X_gyro, y=y)


if __name__ == "__main__":
    # oznake za .bin glede na aktivnost 0 - telefon 1 - delo
    files = [
        (str(ROOT_DIR / "podatki/telefon_podatki/telefon_01.bin"), 0),
        (str(ROOT_DIR / "podatki/telefon_podatki/telefon_02.bin"), 0),
        (str(ROOT_DIR / "podatki/delo_podatki/delo_03.bin"), 1),
    ]

    X_acc, X_gyro, y = build_dataset(files)

    save_dataset(
        X_acc,
        X_gyro,
        y,
        str(ROOT_DIR / "dataset.npz"),
    )

    print("Dataset ustvarjen.")
    print("X_acc:", X_acc.shape)
    print("X_gyro:", X_gyro.shape)
    print("y:", y.shape)
