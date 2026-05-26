from pathlib import Path
import numpy as np
import torch

from src.model.cnn_model_mic import CNNModel
from src.preprocessing.dataset_builder_mic import load_session_mic
from src.preprocessing.windower import window_signal_seconds
from src.preprocessing.stft import compute_spectrograms_1d

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT_DIR / "models" / "mic_cnn.pt"


RMS_THRESHOLD = 0.01
SEG_FRAMES = 311

CLASS_NAMES = {
    0: "GLASBA",
    1: "POGOVOR",
}


def predict_mic(bin_file):

    fvz_mic, sig_mic = load_session_mic(bin_file)

    samples_per_seg = int(5.0 * fvz_mic)
    n_segments = len(sig_mic) // samples_per_seg

    checkpoint = torch.load(str(MODEL_PATH), map_location="cpu")
    log_min = checkpoint["log_min"]
    log_max = checkpoint["log_max"]

    model = CNNModel(num_classes=2)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    for i in range(n_segments):

        start = i * samples_per_seg
        end = start + samples_per_seg
        segment_raw = sig_mic[start:end]

        rms = np.sqrt(np.mean(segment_raw**2))

        minutes = (i * 5) // 60
        seconds = (i * 5) % 60
        time_str = f"{minutes:02d}:{seconds:02d}"

        if rms < RMS_THRESHOLD:
            print(f"{time_str}  TIŠINA")
            continue

        windows = window_signal_seconds(
            segment_raw, fvz_mic, T_window=0.032, prekrivanje=0.5
        )

        spectrograms = compute_spectrograms_1d(windows)

        if spectrograms.shape[0] < SEG_FRAMES:
            print(f"{time_str}  TIŠINA")
            continue

        spec = spectrograms[:SEG_FRAMES].T

        spec = np.log10(spec + 1e-10)
        spec = (spec - log_min) / (log_max - log_min)
        spec = np.clip(spec, 0.0, 1.0)

        x = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(x)
            pred = output.argmax(dim=1).item()

        print(f"{time_str}  {CLASS_NAMES[pred]}  (rms={rms:.4f})")


if __name__ == "__main__":
    predict_mic(str(ROOT_DIR / "seja.bin"))
