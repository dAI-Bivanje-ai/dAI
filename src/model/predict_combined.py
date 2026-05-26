from pathlib import Path
import numpy as np
import torch


from src.model.cnn_model import CNNModel as IMUModel
from src.model.cnn_model_mic import CNNModel as MicModel
from src.preprocessing.dataset_builder import load_session, SEGMENT_LENGTH
from src.preprocessing.dataset_builder_mic import load_session_mic
from src.preprocessing.windower import window_signal_seconds
from src.preprocessing.stft import (
    compute_spectrograms,
    group_spectrograms,
    compute_spectrograms_1d,
)


ROOT_DIR = Path(__file__).resolve().parents[2]


IMU_MODEL_PATH = ROOT_DIR / "models" / "imu_cnn.pt"
MIC_MODEL_PATH = ROOT_DIR / "models" / "mic_cnn.pt"

IMU_CLASSES = {0: "DELO", 1: "TELEFON"}
MIC_CLASSES = {0: "GLASBA", 1: "POGOVOR"}

RMS_THRESHOLD = 0.01
SEG_FRAMES = 311
MIC_SEG_SEC = 5


NOTIFICATIONS = {
    ("DELO", "GLASBA"): "Delaš, ampak glasba je preglasna",
    ("DELO", "POGOVOR"): "Delaš, ampak nekdo govori v ozadju",
    ("TELEFON", "GLASBA"): "Ne delaš ...",
    ("TELEFON", "POGOVOR"): "Ne delaš ...",
}


def _imu_predictions(bin_file):

    fvz_acc, sig_acc, fvz_gyro, sig_gyro = load_session(bin_file)

    windows_acc = window_signal_seconds(sig_acc, fvz_acc, force_W=49)
    windows_gyro = window_signal_seconds(sig_gyro, fvz_gyro, force_W=211)

    specs_acc = compute_spectrograms(windows_acc)
    specs_gyro = compute_spectrograms(windows_gyro)

    M = min(specs_acc.shape[0], specs_gyro.shape[0])

    specs_acc = specs_acc[:M]
    specs_gyro = specs_gyro[:M]

    segs_acc = group_spectrograms(specs_acc, SEGMENT_LENGTH)
    segs_gyro = group_spectrograms(specs_gyro, SEGMENT_LENGTH)

    N = min(segs_acc.shape[0], segs_gyro.shape[0])
    segs_acc = segs_acc[:N].astype(np.float32)
    segs_gyro = segs_gyro[:N].astype(np.float32)

    segs_acc = np.log10(segs_acc + 1e-10)
    segs_gyro = np.log10(segs_gyro + 1e-10)
    segs_acc = (segs_acc - segs_acc.min()) / (segs_acc.max() - segs_acc.min())
    segs_gyro = (segs_gyro - segs_gyro.min()) / (segs_gyro.max() - segs_gyro.min())

    segs_acc = np.transpose(segs_acc, (0, 3, 1, 2))
    segs_gyro = np.transpose(segs_gyro, (0, 3, 1, 2))

    model = IMUModel(num_classes=2)
    model.load_state_dict(torch.load(str(IMU_MODEL_PATH), map_location="cpu"))
    model.eval()

    predictions = []
    with torch.no_grad():
        for i in range(N):
            acc = torch.tensor(segs_acc[i], dtype=torch.float32).unsqueeze(0)
            gyro = torch.tensor(segs_gyro[i], dtype=torch.float32).unsqueeze(0)
            pred = model(acc, gyro).argmax(dim=1).item()
            predictions.append(pred)

    return predictions


def _mic_predictions(bin_file):
    """
    Vrne seznam (rms, class_or_None) — ena na 2-sekundni segment.
    """
    fvz_mic, sig_mic = load_session_mic(bin_file)
    samples_per_seg = int(MIC_SEG_SEC * fvz_mic)
    n_segments = len(sig_mic) // samples_per_seg

    checkpoint = torch.load(str(MIC_MODEL_PATH), map_location="cpu")
    log_min = checkpoint["log_min"]
    log_max = checkpoint["log_max"]

    model = MicModel(num_classes=2)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    results = []
    with torch.no_grad():
        for i in range(n_segments):
            start = i * samples_per_seg
            segment_raw = sig_mic[start : start + samples_per_seg]
            rms = np.sqrt(np.mean(segment_raw**2))

            if rms < RMS_THRESHOLD:
                results.append((rms, None))
                continue

            windows = window_signal_seconds(
                segment_raw, fvz_mic, T_window=0.032, prekrivanje=0.5
            )
            spectrograms = compute_spectrograms_1d(windows)

            if spectrograms.shape[0] < SEG_FRAMES:
                results.append((rms, None))
                continue

            spec = spectrograms[:SEG_FRAMES].T
            spec = np.log10(spec + 1e-10)
            spec = (spec - log_min) / (log_max - log_min)
            spec = np.clip(spec, 0.0, 1.0)

            x = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            pred = model(x).argmax(dim=1).item()
            results.append((rms, pred))

    return results


def predict_combined(bin_file):
    print(f"Nalagam: {bin_file}\n")

    imu_preds = _imu_predictions(bin_file)
    mic_results = _mic_predictions(bin_file)

    for mic_idx, (rms, mic_class) in enumerate(mic_results):
        t_sec = mic_idx * MIC_SEG_SEC
        minutes = t_sec // 60
        seconds = t_sec % 60
        time_str = f"{minutes:02d}:{seconds:02d}"

        imu_idx = t_sec // SEGMENT_LENGTH
        imu_label = IMU_CLASSES[imu_preds[imu_idx]] if imu_idx < len(imu_preds) else "?"

        if mic_class is None:
            print(f"{time_str}  IMU={imu_label}  MIC=TIŠINA  (rms={rms:.4f})")
            continue

        mic_label = MIC_CLASSES[mic_class]
        notif = NOTIFICATIONS.get((imu_label, mic_label), "")
        notif_str = f"  → {notif}" if notif else ""
        print(
            f"{time_str}  IMU={imu_label}  MIC={mic_label}  (rms={rms:.4f}){notif_str}"
        )


if __name__ == "__main__":
    predict_combined(str(ROOT_DIR / "seja.bin"))
