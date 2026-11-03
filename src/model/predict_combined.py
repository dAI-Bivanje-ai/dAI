"""
Združeno napovedovanje aktivnosti iz IMU in mikrofonske seje.

Modul za vsako sejo posebej napove IMU aktivnost (delo / telefon) in
mikrofonski razred (glasba / pogovor / tišina), oba rezultata poravna po
času in po potrebi izpiše ustrezno opozorilo.
"""

import numpy as np
import torch

from src.config import (
    ACC_FFT_WINDOW,
    GYRO_FFT_WINDOW,
    IMU_CLASSES,
    IMU_FFT_OVERLAP,
    IMU_FRAMES_PER_IMAGE,
    IMU_MODEL_PATH,
    LOG_EPSILON,
    MIC_CLASSES,
    MIC_FRAMES_PER_IMAGE,
    MIC_IMAGE_SECONDS,
    MIC_MODEL_PATH,
    MIC_RMS_SILENCE_THRESHOLD,
    MIC_STFT_OVERLAP,
    MIC_STFT_WINDOW_SECONDS,
    NOTIFICATIONS,
    NUM_MIC_CLASSES,
    ROOT_DIR,
    SILENCE_LABEL,
)
from src.model.cnn_model import load_imu_checkpoint
from src.model.cnn_model_mic import CNNModel as MicModel
from src.preprocessing.dataset_builder import (
    load_session,
    image_seconds,
    pair_by_time,
)
from src.preprocessing.dataset_builder_mic import load_session_mic
from src.preprocessing.windower import window_signal_seconds
from src.preprocessing.stft import (
    compute_spectrograms,
    group_spectrograms,
    compute_spectrograms_1d,
    scale_log_spectrograms,
)


def _imu_predictions(bin_file):
    """
    Vrne seznam IMU napovedi - ena na segment seje.

    Sejo predobdela v ACC in GYRO spektrograme, jih normalizira in skozi
    naučen IMU model napove razred za vsak segment.

    Args:
        bin_file: str - pot do .bin seje

    Returns:
        tuple(list[int], float) - napovedani razredi po segmentih in
        čas med začetkoma dveh zaporednih napovedi v sekundah
    """
    fvz_acc, sig_acc, fvz_gyro, sig_gyro = load_session(bin_file)

    windows_acc = window_signal_seconds(
        sig_acc, fvz_acc, prekrivanje=IMU_FFT_OVERLAP, force_W=ACC_FFT_WINDOW
    )
    windows_gyro = window_signal_seconds(
        sig_gyro, fvz_gyro, prekrivanje=IMU_FFT_OVERLAP, force_W=GYRO_FFT_WINDOW
    )

    specs_acc = compute_spectrograms(windows_acc)
    specs_gyro = compute_spectrograms(windows_gyro)

    segs_acc = group_spectrograms(specs_acc, IMU_FRAMES_PER_IMAGE)
    segs_gyro = group_spectrograms(specs_gyro, IMU_FRAMES_PER_IMAGE)

    # ACC in GYRO sliko združimo po času, ne po indeksu (kot pri treningu)
    gyro_image_seconds = image_seconds(GYRO_FFT_WINDOW, fvz_gyro)
    segs_acc, segs_gyro = pair_by_time(
        segs_acc, segs_gyro, image_seconds(ACC_FFT_WINDOW, fvz_acc), gyro_image_seconds
    )

    N = segs_gyro.shape[0]
    segs_acc = segs_acc.astype(np.float32)
    segs_gyro = segs_gyro.astype(np.float32)

    model, log_ranges = load_imu_checkpoint(IMU_MODEL_PATH)

    segs_acc = scale_log_spectrograms(
        segs_acc, log_ranges["log_min_acc"], log_ranges["log_max_acc"]
    )
    segs_gyro = scale_log_spectrograms(
        segs_gyro, log_ranges["log_min_gyro"], log_ranges["log_max_gyro"]
    )

    segs_acc = np.transpose(segs_acc, (0, 3, 1, 2))
    segs_gyro = np.transpose(segs_gyro, (0, 3, 1, 2))

    predictions = []
    with torch.no_grad():
        for i in range(N):
            acc = torch.tensor(segs_acc[i], dtype=torch.float32).unsqueeze(0)
            gyro = torch.tensor(segs_gyro[i], dtype=torch.float32).unsqueeze(0)
            pred = model(acc, gyro).argmax(dim=1).item()
            predictions.append(pred)

    # Vsaka napoved je ena GYRO slika, te se začnejo vsakih ~2.01 s
    return predictions, gyro_image_seconds


def _mic_predictions(bin_file):
    """
    Vrne seznam (rms, class_or_None) - ena na ~3-sekundni segment.
    """
    fvz_mic, sig_mic = load_session_mic(bin_file)
    samples_per_seg = int(MIC_IMAGE_SECONDS * fvz_mic)
    n_segments = len(sig_mic) // samples_per_seg

    checkpoint = torch.load(str(MIC_MODEL_PATH), map_location="cpu")
    log_min = checkpoint["log_min"]
    log_max = checkpoint["log_max"]

    model = MicModel(num_classes=NUM_MIC_CLASSES)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    results = []
    with torch.no_grad():
        for i in range(n_segments):
            start = i * samples_per_seg
            segment_raw = sig_mic[start : start + samples_per_seg]
            rms = np.sqrt(np.mean(segment_raw**2))

            if rms < MIC_RMS_SILENCE_THRESHOLD:
                results.append((rms, None))
                continue

            windows = window_signal_seconds(
                segment_raw,
                fvz_mic,
                T_window=MIC_STFT_WINDOW_SECONDS,
                prekrivanje=MIC_STFT_OVERLAP,
            )
            spectrograms = compute_spectrograms_1d(windows)

            if spectrograms.shape[0] < MIC_FRAMES_PER_IMAGE:
                results.append((rms, None))
                continue

            spec = spectrograms[:MIC_FRAMES_PER_IMAGE].T
            spec = np.log10(spec + LOG_EPSILON)
            spec = (spec - log_min) / (log_max - log_min)
            spec = np.clip(spec, 0.0, 1.0)

            x = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            pred = model(x).argmax(dim=1).item()
            results.append((rms, pred))

    return results


def predict_combined(bin_file):
    """
    Združi IMU in mikrofonske napovedi za eno sejo in jih izpiše.

    Za vsak mikrofonski segment poišče ujemajočo IMU napoved po času in
    izpiše obe oznaki, glasnost (rms) ter morebitno opozorilo.

    Args:
        bin_file: str - pot do .bin seje
    """
    print(f"Nalagam: {bin_file}\n")

    imu_preds, imu_image_seconds = _imu_predictions(bin_file)
    mic_results = _mic_predictions(bin_file)

    for mic_idx, (rms, mic_class) in enumerate(mic_results):
        mic_start = mic_idx * MIC_IMAGE_SECONDS  # začetek mic segmenta v sekundah
        t_sec = int(mic_start)  # int, ker {:02d} ne sprejme float
        minutes = t_sec // 60
        seconds = t_sec % 60
        time_str = f"{minutes:02d}:{seconds:02d}"

        # IMU napoved, v kateri se začne ta mic segment
        imu_idx = int(mic_start / imu_image_seconds)
        imu_label = IMU_CLASSES[imu_preds[imu_idx]] if imu_idx < len(imu_preds) else "?"

        if mic_class is None:
            print(f"{time_str}  IMU={imu_label}  MIC={SILENCE_LABEL}  (rms={rms:.4f})")
            continue

        mic_label = MIC_CLASSES[mic_class]
        notif = NOTIFICATIONS.get((imu_label, mic_label), "")
        notif_str = f"  → {notif}" if notif else ""
        print(
            f"{time_str}  IMU={imu_label}  MIC={mic_label}  (rms={rms:.4f}){notif_str}"
        )


if __name__ == "__main__":
    predict_combined(str(ROOT_DIR / "seja.bin"))
