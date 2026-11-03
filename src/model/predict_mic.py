"""
Napovedovanje zvočnega razreda iz mikrofonske seje z naučenim CNN modelom.

Modul sejo razdeli na segmente, tihe segmente prepozna po nizki glasnosti
(rms), ostale pa predobdela v spektrograme in zanje napove razred (glasba
ali pogovor).
"""

import numpy as np
import torch

from src.config import (
    LOG_EPSILON,
    MIC_CLASSES,
    MIC_FRAMES_PER_IMAGE,
    MIC_IMAGE_SECONDS,
    MIC_MODEL_PATH,
    MIC_RMS_SILENCE_THRESHOLD,
    MIC_STFT_OVERLAP,
    MIC_STFT_WINDOW_SECONDS,
    NUM_MIC_CLASSES,
    ROOT_DIR,
    SILENCE_LABEL,
)
from src.model.cnn_model_mic import CNNModel
from src.preprocessing.dataset_builder_mic import load_session_mic
from src.preprocessing.windower import window_signal_seconds
from src.preprocessing.stft import compute_spectrograms_1d


def predict_mic(bin_file):
    """
    Napove zvočni razred za vsak segment mikrofonske seje.

    Sejo razdeli na ~3-sekundne segmente, tihe segmente označi kot TIŠINA,
    ostale pa predobdela v spektrograme in skozi naučen model napove razred.

    Args:
        bin_file: str - pot do .bin seje
    """
    fvz_mic, sig_mic = load_session_mic(bin_file)

    samples_per_seg = int(MIC_IMAGE_SECONDS * fvz_mic)
    n_segments = len(sig_mic) // samples_per_seg

    checkpoint = torch.load(str(MIC_MODEL_PATH), map_location="cpu")
    log_min = checkpoint["log_min"]
    log_max = checkpoint["log_max"]

    model = CNNModel(num_classes=NUM_MIC_CLASSES)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    for i in range(n_segments):

        start = i * samples_per_seg
        end = start + samples_per_seg
        segment_raw = sig_mic[start:end]

        rms = np.sqrt(np.mean(segment_raw**2))

        t_sec = int(i * MIC_IMAGE_SECONDS)  # int, ker {:02d} ne sprejme float
        minutes = t_sec // 60
        seconds = t_sec % 60
        time_str = f"{minutes:02d}:{seconds:02d}"

        if rms < MIC_RMS_SILENCE_THRESHOLD:
            print(f"{time_str}  {SILENCE_LABEL}")
            continue

        windows = window_signal_seconds(
            segment_raw,
            fvz_mic,
            T_window=MIC_STFT_WINDOW_SECONDS,
            prekrivanje=MIC_STFT_OVERLAP,
        )

        spectrograms = compute_spectrograms_1d(windows)

        if spectrograms.shape[0] < MIC_FRAMES_PER_IMAGE:
            print(f"{time_str}  {SILENCE_LABEL}")
            continue

        spec = spectrograms[:MIC_FRAMES_PER_IMAGE].T

        spec = np.log10(spec + LOG_EPSILON)
        spec = (spec - log_min) / (log_max - log_min)
        spec = np.clip(spec, 0.0, 1.0)

        x = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(x)
            pred = output.argmax(dim=1).item()

        print(f"{time_str}  {MIC_CLASSES[pred]}  (rms={rms:.4f})")


if __name__ == "__main__":
    predict_mic(str(ROOT_DIR / "seja.bin"))
