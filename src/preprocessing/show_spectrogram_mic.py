import numpy as np
import matplotlib.pyplot as plt

from src.data_logger.data_logger import DataLogger
from src.visualization.data_visualizer import pripravi_pakete, sestavi_podatke_mic
from src.preprocessing.windower import window_signal_seconds
from src.preprocessing.stft import compute_spectrograms_1d
from src.preprocessing.alaw import alaw_decode_all

MIC_FVZ = 8000.0
FFT_W = 256
FFT_OVERLAP = 0.5
FREQ_RES = MIC_FVZ / FFT_W
TIME_STEP = (FFT_W * (1 - FFT_OVERLAP)) / MIC_FVZ

if __name__ == "__main__":
    logger = DataLogger()
    packets = logger.parse_file("seja.bin")
    paketi = pripravi_pakete(packets)

    fvz_mic, y_mic = sestavi_podatke_mic(paketi)
    print(f"Mic signal: {y_mic.shape}, Fvz={fvz_mic} Hz")

    y_mic = alaw_decode_all(y_mic)
    windows = window_signal_seconds(y_mic, fvz_mic, T_window=0.032, prekrivanje=FFT_OVERLAP)
    print(f"Okna: {windows.shape}")

    S = compute_spectrograms_1d(windows)
    print(f"Spektrogram: {S.shape}")

    S_db = 10 * np.log10(S.T + 1e-10)

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
    plt.title("Mikrofon — STFT spektrogram")
    plt.tight_layout()
    plt.show()
