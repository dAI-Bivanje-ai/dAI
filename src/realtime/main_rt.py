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


def print_status(imu_label: str | None, mic_label: str | None, rms: float) -> None:
    imu_str = imu_label or "?"
    if mic_label is None:
        mic_str = "TIŠINA"
        notif_str = ""
    else:
        mic_str = mic_label
        notif = NOTIFICATIONS.get((imu_str, mic_label), "")
        notif_str = f"  →  {notif}" if notif else ""

    print(f"IMU={imu_str}  MIC={mic_str}  (rms={rms:.4f}){notif_str}")


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    imu_model = load_imu_model()
    logging.info("IMU model naložen: %s", IMU_MODEL_PATH)

    mic_model, log_min, log_max = load_mic_model()
    logging.info("Mic model naložen: %s", MIC_MODEL_PATH)

    imu_preprocessor = RealtimePreprocessor()
    mic_preprocessor = MicRealtimePreprocessor(log_min=log_min, log_max=log_max)

    parser = LivePacketParser()
    reader = LiveSerialReader()

    acc_buf: deque = deque(maxlen=ACC_MAXLEN)
    gyro_buf: deque = deque(maxlen=GYRO_MAXLEN)
    mic_buf: deque = deque(maxlen=MIC_MAXLEN)

    packet_count = 0
    last_imu_label: str | None = None
    last_mic_label: str | None = None
    last_rms = 0.0

    try:
        for chunk in reader.read_stream():
            packets = parser.feed(chunk)

            for packet in packets:
                chunks = packet.get("chunks", {})

                if ID_ACC in chunks:
                    for x, y, z in chunks[ID_ACC]:
                        acc_buf.append(
                            (
                                x * ACC_RESOLUTION,
                                y * ACC_RESOLUTION,
                                z * ACC_RESOLUTION,
                            )
                        )

                if ID_GYRO in chunks:
                    for x, y, z in chunks[ID_GYRO]:
                        gyro_buf.append(
                            (
                                x * GYRO_RESOLUTION,
                                y * GYRO_RESOLUTION,
                                z * GYRO_RESOLUTION,
                            )
                        )

                if ID_MIC in chunks:
                    mic_buf.extend(chunks[ID_MIC])

                packet_count += 1

                if packet_count % IMU_PREDICT_EVERY_N == 0:
                    label = run_imu_inference(
                        imu_model, imu_preprocessor, acc_buf, gyro_buf
                    )
                    if label:
                        last_imu_label = label
                        print_status(last_imu_label, last_mic_label, last_rms)

                if packet_count % MIC_PREDICT_EVERY_N == 0:
                    label, rms = run_mic_inference(mic_model, mic_preprocessor, mic_buf)
                    last_rms = rms
                    last_mic_label = label
                    print_status(last_imu_label, last_mic_label, last_rms)

    except KeyboardInterrupt:
        print("\nUstavljanje...")
    finally:
        stats = parser.stats()
        print(
            f"\nPaketi: {stats['valid_packets']} veljavnih / {stats['total_packets']} skupaj"
        )
        reader.stop()


if __name__ == "__main__":
    run()
