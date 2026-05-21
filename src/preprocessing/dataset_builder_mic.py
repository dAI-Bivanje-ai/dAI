import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from visualization.data_visualizer import pripravi_pakete, sestavi_podatke_mic
from data_logger.data_logger import DataLogger
from windower import window_signal_seconds
from stft import compute_spectrograms_1d
from alaw import alaw_decode_all


def load_session_mic(bin_file):
    """
    Prebere eno .bin datoteko in vrne dekodirani mikrofon signal.

    Args:
        bin_file (str): Pot do .bin datoteke.

    Returns:
        tuple: (fvz_mic, mic_signal)
            fvz_mic: float — vzorčevalna frekvenca (8000.0 Hz)
            mic_signal: numpy array (N,) — linearni PCM signal
    """
    logger = DataLogger()
    raw_packets = logger.parse_file(bin_file)

    packets = pripravi_pakete(raw_packets)

    fvz_mic, mic_raw = sestavi_podatke_mic(packets)

    # A-law dekodiranje: int8 → linearni PCM
    mic_signal = alaw_decode_all(mic_raw)

    return fvz_mic, mic_signal


def build_dataset_mic(files):
    """
    Zgradi dataset za mikrofon CNN iz seznama .bin datotek.

    Za vsako datoteko:
      1. Naloži in dekodira signal
      2. Razreže na 32ms STFT okna s 50% prekrivanjem → (M, 256)
      3. Izračuna STFT frame-e → (M, 129)
      4. Razreže STFT na 2-sekundne segmente po 62 frame-ov → (N, 129, 62)
      5. Doda labelo za vsak segment

    Na koncu: log10 normalizacija + skaliranje na [0, 1] globalno.

    Args:
        files: list of (bin_file, label) parov

    Returns:
        tuple: (X, y)
            X: numpy array (N, 129, 62) — normalizirani spektrogrami
            y: numpy array (N,) — labele (0=glasba, 1=pogovor)
    """
    result = []
    SEG_FRAMES = 62  # floor((2s * 8000 - 256) / 128) + 1 = število STFT frame-ov v 2s

    for bin_file, label in files:

        fvz_mic, sig_mic = load_session_mic(bin_file)

        # razrežemo signal na 32ms okna s 50% prekrivanjem → (M, 256)
        windows = window_signal_seconds(
            sig_mic, fvz_mic, T_window=0.032, prekrivanje=0.5
        )

        # STFT: za vsako okno FFT → cel spektrogram seje (M, 129)
        spectograms = compute_spectrograms_1d(windows)

        # razrežemo spektrogram na 2-sekundne segmente za CNN
        for i in range(spectograms.shape[0] // SEG_FRAMES):
            segment = spectograms[i * SEG_FRAMES : (i + 1) * SEG_FRAMES]
            segment = segment.T  # (62, 129) → (129, 62)
            result.append((segment, label))

    X = []
    y = []

    for segment, label in result:
        X.append(segment)
        y.append(label)

    X = np.stack(X)   # (N, 129, 62)
    y = np.array(y)   # (N,)

    # globalna log10 normalizacija + skaliranje na [0, 1]
    X = np.log10(X + 1e-10)
    X = (X - X.min()) / (X.max() - X.min())

    return X, y


def save_dataset(X, y, filename="dataset_mic.npz"):
    """
    Shrani dataset na disk.

    Args:
        X: numpy array (N, 129, 62)
        y: numpy array (N,)
        filename: ime izhodne .npz datoteke
    """
    np.savez(filename, X=X, y=y)


if __name__ == "__main__":
    files = [
        ("podatki/mic_podatki/glasba_01.bin", 0),
        ("podatki/mic_podatki/pogovor_02.bin", 1),
    ]

    X, y = build_dataset_mic(files)
    save_dataset(X, y, "dataset_mic.npz")

    print("Dataset ustvarjen")
    print("X:", X.shape)
    print("y:", y.shape)
