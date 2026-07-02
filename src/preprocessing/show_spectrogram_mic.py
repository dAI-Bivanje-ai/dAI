"""
Skripta za prikaz STFT spektrograma iz mikrofonskega signala.

Prebere en .bin posnetek, iz njega sestavi mikrofonski signal, ga A-law
dekodira, razdeli na časovna okna in za vsako okno izračuna STFT. Rezultat
prikaže kot spektrogram v decibelski skali (frekvenca proti času).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.data_logger.data_logger import DataLogger

# Korenski direktorij projekta (dve mapi nad to datoteko).
ROOT_DIR = Path(__file__).resolve().parents[2]
from src.visualization.data_visualizer import pripravi_pakete, sestavi_podatke_mic
from src.preprocessing.windower import window_signal_seconds
from src.preprocessing.stft import compute_spectrograms_1d
from src.preprocessing.alaw import alaw_decode_all

# parametri za STFT
MIC_FVZ = 8000.0  # vzorčevalna frekvenca mikrofona [Hz]
FFT_W = 256  # dolžina FFT okna [vzorci] = 32 ms pri 8 kHz
FFT_OVERLAP = 0.5  # 50% prekrivanje med okni
FREQ_RES = MIC_FVZ / FFT_W  # frekvenčna resolucija [Hz/bin] = 31.25 Hz
TIME_STEP = (FFT_W * (1 - FFT_OVERLAP)) / MIC_FVZ  # časovni korak [s] = 16 ms

if __name__ == "__main__":
    # branje in parsanje binarne datoteke ter priprava paketov
    logger = DataLogger()
    packets = logger.parse_file(str(ROOT_DIR / "pogovor_04.bin"))
    paketi = pripravi_pakete(packets)

    # sestavimo signal iz mic chunkov (chunk id 4)
    fvz_mic, y_mic = sestavi_podatke_mic(paketi)
    print(f"Mic signal: {y_mic.shape}, Fvz={fvz_mic} Hz")

    # A-law dekodiranje
    y_mic = alaw_decode_all(y_mic)

    # razrežemo signal na 32ms okna s 50% prekrivanjem
    windows = window_signal_seconds(
        y_mic, fvz_mic, T_window=0.032, prekrivanje=FFT_OVERLAP
    )
    print(f"Okna: {windows.shape}")

    # STFT: za vsako okno FFT
    S = compute_spectrograms_1d(windows)
    print(f"Spektrogram: {S.shape}")

    # log-power v dB za vizualizacijo, transponiramo za imshow (freq × čas)
    S_db = 10 * np.log10(S.T + 1e-10)

    # prikaz spektrograma: čas na x osi, frekvenca na y osi
    plt.figure(figsize=(14, 6))
    plt.imshow(
        S_db,
        origin="lower",
        aspect="auto",
        extent=(0, S.shape[0] * TIME_STEP, 0, MIC_FVZ / 2),
    )
    plt.colorbar(label="Moč [dB]")
    plt.xlabel("Čas [s]")
    plt.ylabel("Frekvenca [Hz]")
    plt.title("Mikrofon STFT spektrogram")
    plt.tight_layout()
    plt.show()
