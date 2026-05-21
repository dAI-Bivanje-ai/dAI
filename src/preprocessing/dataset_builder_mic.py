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

    logger = DataLogger()
    raw_packets = logger.parse_file(bin_file)

    packets = pripravi_pakete(raw_packets)

    fvz_mic, mic_raw = sestavi_podatke_mic(packets)

    mic_signal = alaw_decode_all(mic_raw)

    return fvz_mic, mic_signal


def build_dataset_mic(files):

    result = []
    SEG_FRAMES = 62  # floor((16000 - 256) / 128) + 1
    for bin_file, label in files:

        fvz_mic, sig_mic = load_session_mic(bin_file)
        # iz signala nardimo okna
        windows = window_signal_seconds(
            sig_mic, fvz_mic, T_window=0.032, prekrivanje=0.5
        )

        # za vsa okna nardimo en velik spektrogram, ki ga je treba še razrezati
        spectograms = compute_spectrograms_1d(windows)
        # gremo in jemljemo segmente celotnega spektrograma,
        # rezultat je več manjših spektrogramov
        for i in range(spectograms.shape[0] // SEG_FRAMES):
            segment = spectograms[i * SEG_FRAMES : (i + 1) * SEG_FRAMES]
            segment = segment.T  # transponiramo
            result.append((segment, label))

    X = []  # segmenti
    y = []  # labeli

    for segment, label in result:
        X.append(segment)
        y.append(label)

    X = np.stack(X)
    y = np.array(y)

    X = np.log10(X + 1e-10)
    X = (X - X.min()) / (X.max() - X.min())

    return X, y


def save_dataset(X, y, filename="dataset_mic.npz"):
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
