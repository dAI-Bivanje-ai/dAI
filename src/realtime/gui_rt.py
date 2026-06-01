import queue
import threading
import tkinter as tk
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

WINDOW_SECONDS = 8.0
ACC_MAXLEN = int(WINDOW_SECONDS * ACC_SAMPLE_RATE)
GYRO_MAXLEN = int(WINDOW_SECONDS * GYRO_SAMPLE_RATE)
MIC_MAXLEN = int(8.0 * MIC_SAMPLE_RATE)

ACC_RESOLUTION = 1e-3
GYRO_RESOLUTION = 8.75e-3

IMU_PREDICT_EVERY_N = 12
MIC_PREDICT_EVERY_N = 100
CLASS_COLORS = {
    "DELO": "#2ecc71",
    "TELEFON": "#e74c3c",
    "GLASBA": "#3498db",
    "POGOVOR": "#9b59b6",
}
IDLE_COLOR = "#95a5a6"
BG = "#2c3e50"
CARD_BG = "#34495e"


def load_imu_model():
    model = IMUModel(num_classes=2)
    model.load_state_dict(torch.load(str(IMU_MODEL_PATH), map_location="cpu"))
    model.eval()
    return model


def load_mic_model():
    checkpoint = torch.load(str(MIC_MODEL_PATH), map_location="cpu")
    model = MicModel(num_classes=2)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, float(checkpoint["log_min"]), float(checkpoint["log_max"])


class GUI:

    REFRESH_RATE_MS = 200

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("dAI")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

    def build(self):
        self.root.geometry("480x400")

        tk.Label(
            self.root,
            text="dAI - Realtime klasifikacija aktivnosti",
            font=("Menlo", 14, "bold"),
            fg="white",
            bg=BG,
            pady=14,
        ).pack()
