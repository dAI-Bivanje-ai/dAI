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
TIME_STEP = (FFT_W * (1 - FFT_OVERLAP)) / MIC_FVZ  # 0.016 s/frame
N_FRAMES = 500  # 8 s


def load_session_mic(path):
    logger = DataLogger()
    raw = logger.parse_file(path)
    paketi = pripravi_pakete(raw)
    fvz, mic_raw = sestavi_podatke_mic(paketi)
    signal = alaw_decode_all(mic_raw)
    return fvz, signal


def compute_stft(signal, fvz):
    windows = window_signal_seconds(
        signal, fvz, T_window=FFT_W / fvz, prekrivanje=FFT_OVERLAP
    )
    return compute_spectrograms_1d(windows)  # (M, 129)


def normalize_db(S):
    S_db = 10 * np.log10(S + 1e-10)
    return (S_db - S_db.min()) / (S_db.max() - S_db.min())


def get_middle(S, n):
    M = S.shape[0]
    mid = M // 2
    start = max(0, mid - n // 2)
    return S[start : start + n]


def plot_grid(axes_row, S, title, use_all=False):
    seg = S if use_all else get_middle(S, N_FRAMES)
    img = normalize_db(seg.T)  # (129, M)
    n = seg.shape[0]
    total_s = n * TIME_STEP

    axes_row.imshow(
        img,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=(0, total_s, 0, MIC_FVZ / 2),
    )
    axes_row.set_title(title, fontsize=11, fontweight="bold")
    axes_row.set_ylabel("Frekvenca [Hz]")
    axes_row.set_xlabel("Čas [s]")


def main():
    fvz_g, sig_g = load_session_mic("podatki/mic_podatki/glasba_01.bin")
    fvz_p, sig_p = load_session_mic("podatki/mic_podatki/pogovor_02.bin")

    S_glasba = compute_stft(sig_g, fvz_g)
    S_pogovor = compute_stft(sig_p, fvz_p)
    print(
        f"  glasba:  {S_glasba.shape[0]} frame-ov ({S_glasba.shape[0] * TIME_STEP:.1f}s)"
    )
    print(
        f"  pogovor: {S_pogovor.shape[0]} frame-ov ({S_pogovor.shape[0] * TIME_STEP:.1f}s)"
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.suptitle(
        f"Mikrofon — odsek ~{N_FRAMES * TIME_STEP:.0f}s: GLASBA vs. POGOVOR",
        fontsize=13,
        fontweight="bold",
    )
    plot_grid(axes[0], S_glasba, "GLASBA")
    plot_grid(axes[1], S_pogovor, "POGOVOR")
    plt.tight_layout()
    plt.savefig("models/compare_spectrograms_mic.png", dpi=150, bbox_inches="tight")
    print("Slika shranjena: models/compare_spectrograms_mic.png")
    plt.close()

    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 7))
    fig2.suptitle(
        "Mikrofon — cel signal: GLASBA vs. POGOVOR", fontsize=13, fontweight="bold"
    )
    plot_grid(axes2[0], S_glasba, "GLASBA", use_all=True)
    plot_grid(axes2[1], S_pogovor, "POGOVOR", use_all=True)
    plt.tight_layout()
    plt.savefig("models/compare_spectrograms_mic_all.png", dpi=150, bbox_inches="tight")
    print("Slika shranjena: models/compare_spectrograms_mic_all.png")
    plt.close()


if __name__ == "__main__":
    main()
