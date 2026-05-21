import matplotlib.pyplot as plt

from src.data_logger.data_logger import DataLogger
from src.visualization.data_visualizer import pripravi_pakete, sestavi_podatke
from src.preprocessing.stft import compute_spectrograms
from src.preprocessing.windower import window_signal_seconds
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
BIN_FILE = ROOT_DIR / "podatki" / "delo_podatki" / "delo_01.bin"


def main():
    """
    Primer pretvorbe IMU signalov v spektrograme.

    Program:
    1. prebere .bin datoteko,
    2. iz nje sestavi ACC in GYRO signal,
    3. signal razdeli na časovna okna,
    4. za vsako okno izračuna spektrogram,
    5. prikaže RGB spektrograme.
    """
    logger = DataLogger()
    # parsanje bin datoteke
    raw_packets = logger.parse_file(str(BIN_FILE))

    # priprava paketov za nadaljno uporabo
    packets = pripravi_pakete(raw_packets)

    # 1 = gyro, 2 = acc
    Fvz_gyro, sig_gyro = sestavi_podatke(packets, 1)
    Fvz_acc, sig_acc = sestavi_podatke(packets, 2)

    # razdelitev signalov na časovna okna
    acc_windows = window_signal_seconds(sig_acc, Fvz_acc, 2.0, 0.5)
    gyro_windows = window_signal_seconds(sig_gyro, Fvz_gyro, 2.0, 0.5)

    # izračun spektrogramov za vsa okna
    spec_acc = compute_spectrograms(acc_windows)
    spec_gyro = compute_spectrograms(gyro_windows)

    print("ACC windows:", acc_windows.shape)
    print("GYRO windows:", gyro_windows.shape)
    print("ACC spectrograms:", spec_acc.shape)
    print("GYRO spectrograms:", spec_gyro.shape)

    plt.figure(figsize=(10, 5))
    plt.imshow(spec_acc[0], aspect="auto", origin="lower")
    plt.title("RGB spektrogram pospeškometra")
    plt.xlabel("Časovna okna")
    plt.ylabel("Frekvenčni bin")
    plt.tight_layout()

    plt.figure(figsize=(10, 5))
    plt.imshow(spec_gyro[0], aspect="auto", origin="lower")
    plt.title("RGB spektrogram žiroskopa")
    plt.xlabel("Časovna okna")
    plt.ylabel("Frekvenčni bin")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
