import matplotlib.pyplot as plt
import numpy as np

from src.data_logger.data_logger import DataLogger


class Paket:

    def __init__(self, id: int, ts: float, data: np.ndarray) -> None:
        self.id = id
        self.ts = ts
        self.data = data


def pripravi_pakete(packets: list[dict]) -> list[Paket]:

    handled_packets: list[Paket] = []

    for packet in packets:
        timestamp = packet["timestamp"] / 1000  # ms -> s
        for paket_id, samples in packet["chunks"].items():
            handled_packets.append(Paket(paket_id, timestamp, np.array(samples)))

    return handled_packets


def sestavi_podatke(
    seznam_paketov: list[Paket], paket_id: int
) -> tuple[float, np.ndarray]:

    packets = [p for p in seznam_paketov if p.id == paket_id]

    deltas = [packets[i + 1].ts - packets[i].ts for i in range(len(packets) - 1)]
    avg_ts = float(np.mean(deltas))

    nvz_list = [p.data.shape[0] for p in packets]  # stevilo vzorcev na pakets
    avg_nvz = float(np.mean(nvz_list))

    fvz = avg_nvz / avg_ts
    result = np.vstack([p.data for p in packets])

    return fvz, result


def sestavi_podatke_mic(seznam_paketov):

    packets = [p for p in seznam_paketov if p.id == 4]

    result = np.concatenate([p.data for p in packets])

    return 8000, result


def prikazi_signal(
    signal: np.ndarray,
    naslov: str = "",
    startInd: int = 0,
    endInd: int | None = None,
    fvz: float | None = None,
    enota: str = "",
) -> None:
    endInd = endInd or len(signal)
    segment = signal[startInd:endInd]

    if fvz is not None:
        x_os = np.arange(startInd, endInd) / fvz
        x_label = "čas [s]"
        naslov = f"{naslov}  (Fvz = {fvz:.2f} Hz)"
    else:
        x_os = np.arange(startInd, endInd)
        x_label = "indeks vzorca"

    plt.figure(figsize=(12, 6))
    plt.plot(x_os, segment)
    plt.title(naslov)
    plt.xlabel(x_label)
    plt.ylabel(f"[{enota}]" if enota else "vrednost")
    plt.legend(["X", "Y", "Z"])
    plt.tight_layout()
    plt.show()


def prikazi_vse_signale(
    signali: list[tuple[str, np.ndarray, float, str]],
    naslov: str = "",
    start_s: float | None = None,
    end_s: float | None = None,
) -> None:
    fig, axes = plt.subplots(len(signali), 1, figsize=(12, 3 * len(signali)))

    for ax, (ime, signal, fvz, enota) in zip(axes, signali):
        start_i = int((start_s or 0) * fvz)
        end_i = int(end_s * fvz) if end_s else len(signal)
        segment = signal[start_i:end_i]
        t = np.arange(start_i, start_i + len(segment)) / fvz

        ax.plot(t, segment)
        ax.set_title(f"{ime}  (Fvz = {fvz:.2f} Hz)")
        ax.set_xlabel("čas [s]")
        ax.set_ylabel(f"[{enota}]")
        ax.legend(["X", "Y", "Z"])

    if naslov:
        fig.suptitle(naslov)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    logger = DataLogger()
    raw_packets = logger.parse_file("raw_data.bin")
    paketi = pripravi_pakete(raw_packets)

    fvz_gyro, gyro_raw = sestavi_podatke(paketi, 1)
    fvz_accel, accel_raw = sestavi_podatke(paketi, 2)
    fvz_mag, mag_raw = sestavi_podatke(paketi, 3)

    gyro = gyro_raw * 8.75e-3
    accel = accel_raw * 1e-3
    mag = mag_raw * 1.5e-3

    # print(f"gyro  Fvz={fvz_gyro:7.2f} Hz  oblika={gyro.shape}")
    # print(f"accel Fvz={fvz_accel:7.2f} Hz  oblika={accel.shape}")
    # print(f"mag   Fvz={fvz_mag:7.2f} Hz  oblika={mag.shape}")

    signali = [
        ("Žiroskop", gyro, fvz_gyro, "°/s"),
        ("Pospeškometer", accel, fvz_accel, "g"),
        ("Magnetometer", mag, fvz_mag, "Gauss"),
    ]

    prikazi_vse_signale(signali, naslov="Vsi signali - celota")

    dolzina_signala_s = gyro.shape[0] / fvz_gyro

    """
    raw_data.bin -> 30s tipkanja, 30s mirovanja, 30s tipkanje, 30s hoje, pri 20s marku voda
    raw_data_2.bin -> 30s tipkanje, 30s mirovanje, 30s hoje, 30s tipkanje
    raw_data_3.bin -> 60s tipkanje, 30s telefon
    """

    start_s = (
        dolzina_signala_s / 6
    )  # / 6 = raw_data.bin , / 2 = raw_data_2.bin, 1.8 = raw_data_3.bin
    end_s = start_s + 4
    prikazi_vse_signale(
        signali,
        naslov="Vsi signali - izbrani dogodek",
        start_s=start_s,
        end_s=end_s,
    )
