import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.data_logger.data_logger import DataLogger
from src.visualization.data_visualizer import pripravi_pakete, sestavi_podatke
from src.preprocessing.windower import window_signal_seconds
from src.preprocessing.stft import compute_spectrograms

ACC_W = 49
GYRO_W = 211
N_WINDOWS = 40  # število oken za prikaz


def load_session(path):
    logger = DataLogger()
    raw = logger.parse_file(path)
    paketi = pripravi_pakete(raw)

    fvz_gyro, gyro_raw = sestavi_podatke(paketi, 1)
    fvz_acc, acc_raw = sestavi_podatke(paketi, 2)

    sig_gyro = gyro_raw * 8.75e-3
    sig_acc = acc_raw * 1e-3

    return fvz_acc, sig_acc, fvz_gyro, sig_gyro


def get_middle_segment(spectrograms, n):
    """Vzame n oken iz sredine seje."""
    M = spectrograms.shape[0]
    mid = M // 2
    start = max(0, mid - n // 2)
    end = min(M, start + n)
    return spectrograms[start:end]  # (n, freq_bins, 3)


def normalize_for_display(spec_2d):
    """
    spec_2d: (freq_bins, n_windows) — skupna magnituda
    Vrne: (freq_bins, n_windows) float v [0, 1]
    """
    s = np.log10(spec_2d + 1e-10)
    s = (s - s.min()) / (s.max() - s.min())
    return s


def spectrograms_to_image(spectrograms, n=N_WINDOWS):
    """
    spectrograms: (M, freq_bins, 3)
    Vrne: (freq_bins, n) normalizirano za imshow z viridis
    """
    seg = get_middle_segment(spectrograms, n)  # (n, freq_bins, 3)
    magnitude = np.sqrt(np.sum(seg**2, axis=2))  # (n, freq_bins)
    img = magnitude.T  # (freq_bins, n)
    return normalize_for_display(img)


def spectrograms_to_image_all(spectrograms):
    """
    spectrograms: (M, freq_bins, 3) — vsa okna
    Vrne: (freq_bins, M) normalizirano za imshow z viridis
    """
    magnitude = np.sqrt(np.sum(spectrograms**2, axis=2))  # (M, freq_bins)
    img = magnitude.T  # (freq_bins, M)
    return normalize_for_display(img)


def add_axis_labels(ax, fvz, window_size, n_windows, title):
    freq_bins = window_size // 2 + 1
    nyquist = fvz / 2

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Frekvenca [Hz]")
    ax.set_xlabel("Čas [s]")

    # y os hz
    y_ticks = np.linspace(0, freq_bins - 1, 5)
    y_labels = [f"{v:.0f}" for v in np.linspace(0, nyquist, 5)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # x os: sekunde
    step_samples = window_size // 2
    step_s = step_samples / fvz
    total_s = n_windows * step_s
    x_ticks = np.linspace(0, n_windows - 1, 5)
    x_labels = [f"{v:.2f}" for v in np.linspace(0, total_s, 5)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)


def main():
    fvz_acc_d, sig_acc_d, fvz_gyro_d, sig_gyro_d = load_session(
        "podatki/delo_podatki/delo_01.bin"
    )
    fvz_acc_t, sig_acc_t, fvz_gyro_t, sig_gyro_t = load_session(
        "podatki/telefon_podatki/telefon_01.bin"
    )

    spec_acc_d = compute_spectrograms(
        window_signal_seconds(sig_acc_d, fvz_acc_d, force_W=ACC_W)
    )
    spec_gyro_d = compute_spectrograms(
        window_signal_seconds(sig_gyro_d, fvz_gyro_d, force_W=GYRO_W)
    )
    spec_acc_t = compute_spectrograms(
        window_signal_seconds(sig_acc_t, fvz_acc_t, force_W=ACC_W)
    )
    spec_gyro_t = compute_spectrograms(
        window_signal_seconds(sig_gyro_t, fvz_gyro_t, force_W=GYRO_W)
    )

    img_acc_d = spectrograms_to_image(spec_acc_d)
    img_gyro_d = spectrograms_to_image(spec_gyro_d)
    img_acc_t = spectrograms_to_image(spec_acc_t)
    img_gyro_t = spectrograms_to_image(spec_gyro_t)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        "Primerjava spektrogramov: DELO vs. TELEFON", fontsize=14, fontweight="bold"
    )

    n = N_WINDOWS
    add_axis_labels(axes[0, 0], fvz_acc_d, ACC_W, n, "DELO — ACC (pospeškometer)")
    add_axis_labels(axes[0, 1], fvz_gyro_d, GYRO_W, n, "DELO — GYRO (žiroskop)")
    add_axis_labels(axes[1, 0], fvz_acc_t, ACC_W, n, "TELEFON — ACC (pospeškometer)")
    add_axis_labels(axes[1, 1], fvz_gyro_t, GYRO_W, n, "TELEFON — GYRO (žiroskop)")

    for ax, img in zip(axes.flat, [img_acc_d, img_gyro_d, img_acc_t, img_gyro_t]):
        ax.imshow(img, aspect="auto", origin="lower", cmap="viridis")

    plt.tight_layout()

    out_path = "models/compare_spectrograms.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Slika shranjena: {out_path}")
    plt.close()

    img_acc_d_all = spectrograms_to_image_all(spec_acc_d)
    img_gyro_d_all = spectrograms_to_image_all(spec_gyro_d)
    img_acc_t_all = spectrograms_to_image_all(spec_acc_t)
    img_gyro_t_all = spectrograms_to_image_all(spec_gyro_t)

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
    fig2.suptitle(
        "Primerjava spektrogramov (cel signal): DELO vs. TELEFON",
        fontsize=14,
        fontweight="bold",
    )

    add_axis_labels(
        axes2[0, 0], fvz_acc_d, ACC_W, spec_acc_d.shape[0], "DELO — ACC (pospeškometer)"
    )
    add_axis_labels(
        axes2[0, 1], fvz_gyro_d, GYRO_W, spec_gyro_d.shape[0], "DELO — GYRO (žiroskop)"
    )
    add_axis_labels(
        axes2[1, 0],
        fvz_acc_t,
        ACC_W,
        spec_acc_t.shape[0],
        "TELEFON — ACC (pospeškometer)",
    )
    add_axis_labels(
        axes2[1, 1],
        fvz_gyro_t,
        GYRO_W,
        spec_gyro_t.shape[0],
        "TELEFON — GYRO (žiroskop)",
    )

    for ax, img in zip(
        axes2.flat, [img_acc_d_all, img_gyro_d_all, img_acc_t_all, img_gyro_t_all]
    ):
        ax.imshow(img, aspect="auto", origin="lower", cmap="viridis")

    plt.tight_layout()

    out_path_all = "models/compare_spectrograms_all.png"
    plt.savefig(out_path_all, dpi=150, bbox_inches="tight")
    print(f"Slika shranjena: {out_path_all}")
    plt.close()


if __name__ == "__main__":
    main()
