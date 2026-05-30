import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.model.cnn_model import CNNModel as IMUModel
from src.model.cnn_model_mic import CNNModel as MicModel
from src.realtime.packet_parser import LivePacketParser, ID_ACC, ID_GYRO, ID_MIC
from src.realtime.preproc_rt import MicRealtimePreprocessor, RealtimePreprocessor
from src.realtime.serial_reader import LiveSerialReader


ROOT_DIR = Path(__file__).resolve().parents[2]

IMU_MODEL_PATH = ROOT_DIR / "models" / "imu_cnn.pt"
MIC_MODEL_PATH = ROOT_DIR / "models" / "mic_cnn.pt"


IMU_CLASSES = {0: "DELO", 1: "TELEFON"}
MIC_CLASSES = {0: "GLASBA", 1: "POGOVOR"}


NOTIFICATIONS = {
    ("DELO", "GLASBA"): "Delaš, ampak glasba je preglasna",
    ("DELO", "POGOVOR"): "Delaš, ampak nekdo govori v ozadju",
    ("TELEFON", "GLASBA"): "Ne delaš ...",
    ("TELEFON", "POGOVOR"): "Ne delaš ...",
}


ACC_SAMPLE_RATE = 25.0
GYRO_SAMPLE_RATE = 105.0
MIC_SAMPLE_RATE = 8000.0


# tile parametri se bodo spreminjali se, zaenkrat dela okej
WINDOW_SECONDS = 8.0
ACC_MAXLEN = int(WINDOW_SECONDS * ACC_SAMPLE_RATE)  # 200
GYRO_MAXLEN = int(WINDOW_SECONDS * GYRO_SAMPLE_RATE)  # 840

MIC_MAXLEN = int(8.0 * MIC_SAMPLE_RATE)  # 64000

ACC_RESOLUTION = 1e-3
GYRO_RESOLUTION = 8.75e-3

IMU_PREDICT_EVERY_N = 12
MIC_PREDICT_EVERY_N = 100


def load_imu_model() -> IMUModel:
    model = IMUModel(num_classes=2)
    model.load_state_dict(torch.load(str(IMU_MODEL_PATH), map_location="cpu"))
    model.eval()
    return model


def load_mic_model() -> tuple[MicModel, float, float]:
    checkpoint = torch.load(str(MIC_MODEL_PATH), map_location="cpu")
    model = MicModel(num_classes=2)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    log_min = float(checkpoint["log_min"])
    log_max = float(checkpoint["log_max"])
    return model, log_min, log_max


def run_imu_inference(
    model: IMUModel,
    preprocessor: RealtimePreprocessor,
    acc_buf: deque,
    gyro_buf: deque,
) -> str | None:
    if not acc_buf or not gyro_buf:
        return None

    acc_window = np.array(acc_buf, dtype=np.float32)
    gyro_window = np.array(gyro_buf, dtype=np.float32)

    result = preprocessor.process(acc_window, gyro_window)
    if result is None:
        return None

    acc_spec, gyro_spec = result
    acc_t = torch.from_numpy(acc_spec).float().permute(2, 0, 1).unsqueeze(0)
    gyro_t = torch.from_numpy(gyro_spec).float().permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        output = model(acc_t, gyro_t)
        pred = output.argmax(dim=1).item()

    return IMU_CLASSES[pred]


def run_mic_inference(
    model: MicModel,
    preprocessor: MicRealtimePreprocessor,
    mic_buf: deque,
) -> tuple[str | None, float]:
    if not mic_buf:
        return None, 0.0

    samples = np.array(mic_buf, dtype=np.int8)
    tensor, rms = preprocessor.process(samples)

    if tensor is None:
        return None, rms

    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()

    return MIC_CLASSES[pred], rms
