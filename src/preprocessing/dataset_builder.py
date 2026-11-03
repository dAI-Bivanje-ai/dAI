import numpy as np
from src.config import (
    ACC_FFT_WINDOW,
    ACC_RESOLUTION,
    CHUNK_ID_ACC,
    CHUNK_ID_GYRO,
    GYRO_FFT_WINDOW,
    GYRO_RESOLUTION,
    IMU_FFT_OVERLAP,
    IMU_FRAMES_PER_IMAGE,
    ROOT_DIR,
)
from src.visualization.data_visualizer import pripravi_pakete, sestavi_podatke
from src.data_logger.data_logger import DataLogger
from src.preprocessing.windower import window_signal_seconds
from src.preprocessing.stft import (
    compute_spectrograms,
    group_spectrograms,
    compute_log_range,
    scale_log_spectrograms,
)


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

    fvz_gyro, gyro_raw = sestavi_podatke(packets, CHUNK_ID_GYRO)
    fvz_acc, acc_raw = sestavi_podatke(packets, CHUNK_ID_ACC)

    sig_gyro = gyro_raw * GYRO_RESOLUTION
    sig_acc = acc_raw * ACC_RESOLUTION
    # print(sig_gyro.shape)
    return (fvz_acc, sig_acc, fvz_gyro, sig_gyro)


# ACC ima 25 fvz
# razrezemo na okna po 49 vzorcev (2s pribl)
# vsako okno -> FFT -> 1 frame
# prvi frame gre od 0s do 2s
# pri IMU_FRAMES_PER_IMAGE = 2 damo 2 frama v spektrogram, drugi frame gre od 1s do 3s
# skupaj od 0 do 3s


def image_seconds(window_size, fvz):
    """
    Koliko sekund je med začetkoma dveh zaporednih IMU slik.

    Ena slika = IMU_FRAMES_PER_IMAGE frame-ov, frame-i so zamaknjeni za pol okna.
    ACC:  2 * 24 / 24.7 Hz  = 1.94 s
    GYRO: 2 * 106 / 105.6 Hz = 2.01 s
    """
    # isti korak kot v window_signal_seconds
    step = round(window_size * (1 - IMU_FFT_OVERLAP))
    return IMU_FRAMES_PER_IMAGE * step / fvz


def pair_by_time(segs_acc, segs_gyro, acc_image_seconds, gyro_image_seconds):
    """
    Vsaki GYRO sliki poišče ACC sliko, ki se začne ob (skoraj) istem času.

    ACC slike si sledijo hitreje kot GYRO (1.94 s proti 2.01 s), zato nekaj
    ACC slik izpustimo. Če bi jih združili po indeksu, bi bila ACC in GYRO
    polovica para po 10 minutah že ~20 s narazen.

    Returns:
        (segs_acc, segs_gyro) - enako dolga, poravnana po času
    """
    paired_acc = []
    paired_gyro = []

    for gyro_idx in range(len(segs_gyro)):
        start = gyro_idx * gyro_image_seconds
        acc_idx = round(start / acc_image_seconds)

        # ACC posnetka je zmanjkalo
        if acc_idx >= len(segs_acc):
            break

        paired_acc.append(segs_acc[acc_idx])
        paired_gyro.append(segs_gyro[gyro_idx])

    return np.array(paired_acc), np.array(paired_gyro)


def build_dataset(files, log_ranges=None):
    """
    Gre čez vse datoteke, za vsako pokliče load_session,
    windower in stft, doda labele.
    Vrne (X_acc, X_gyro, y, log_ranges).
    """
    all_acc = []
    all_gyro = []
    all_y = []

    for bin_file, label in files:
        fvz_acc, sig_acc, fvz_gyro, sig_gyro = load_session(bin_file)

        windows_acc = window_signal_seconds(
            sig_acc, fvz_acc, prekrivanje=IMU_FFT_OVERLAP, force_W=ACC_FFT_WINDOW
        )
        windows_gyro = window_signal_seconds(
            sig_gyro, fvz_gyro, prekrivanje=IMU_FFT_OVERLAP, force_W=GYRO_FFT_WINDOW
        )

        spectrograms_acc = compute_spectrograms(windows_acc)
        spectrograms_gyro = compute_spectrograms(windows_gyro)

        spectrograms_acc_2d = group_spectrograms(spectrograms_acc, IMU_FRAMES_PER_IMAGE)
        spectrograms_gyro_2d = group_spectrograms(
            spectrograms_gyro, IMU_FRAMES_PER_IMAGE
        )

        # ACC in GYRO sliko združimo po času, ne po indeksu
        spectrograms_acc_2d, spectrograms_gyro_2d = pair_by_time(
            spectrograms_acc_2d,
            spectrograms_gyro_2d,
            image_seconds(ACC_FFT_WINDOW, fvz_acc),
            image_seconds(GYRO_FFT_WINDOW, fvz_gyro),
        )
        N = spectrograms_gyro_2d.shape[0]

        labels = np.full(N, label)

        all_acc.append(spectrograms_acc_2d)
        all_gyro.append(spectrograms_gyro_2d)
        all_y.append(labels)

    X_acc = np.concatenate(all_acc, axis=0)
    X_gyro = np.concatenate(all_gyro, axis=0)
    y = np.concatenate(all_y, axis=0)

    # ce to velja pomeni, da smo v treningu
    if log_ranges is None:
        log_min_acc, log_max_acc = compute_log_range(X_acc)
        log_min_gyro, log_max_gyro = compute_log_range(X_gyro)
        log_ranges = {
            "log_min_acc": log_min_acc,
            "log_max_acc": log_max_acc,
            "log_min_gyro": log_min_gyro,
            "log_max_gyro": log_max_gyro,
        }

    X_acc = scale_log_spectrograms(
        X_acc, log_ranges["log_min_acc"], log_ranges["log_max_acc"]
    )
    X_gyro = scale_log_spectrograms(
        X_gyro, log_ranges["log_min_gyro"], log_ranges["log_max_gyro"]
    )

    return X_acc, X_gyro, y, log_ranges


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

    X_acc, X_gyro, y, log_ranges = build_dataset(files)

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
