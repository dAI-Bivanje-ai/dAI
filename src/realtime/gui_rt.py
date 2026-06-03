import queue
import threading
import tkinter as tk
from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.model.cnn_model import CNNModel as IMUModel
from src.model.cnn_model_mic import CNNModel as MicModel
from src.realtime.packet_parser import LivePacketParser, ID_MIC
from src.realtime.preproc_rt import MicRealtimePreprocessor, RealtimePreprocessor
from src.realtime.serial_reader import LiveSerialReader
from src.realtime.signal_buffer import SignalBuffer

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
MIC_SEGMENT_SECONDS = 5.0
MIC_STFT_WINDOW_SECONDS = 0.032
MIC_STFT_OVERLAP = 0.5
MIC_RMS_THRESHOLD = 0.01
ACC_RESOLUTION = 1e-3
GYRO_RESOLUTION = 8.75e-3
MIC_MAXLEN = int(MIC_SEGMENT_SECONDS * MIC_SAMPLE_RATE)

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

        self.queue: queue.Queue = queue.Queue()
        self.last_imu = None
        self.last_mic = None

        self.build()

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

        cards = tk.Frame(self.root, bg=BG)
        cards.pack(fill=tk.X, padx=24)

        self.imu_frame = self.make_card(cards, "GIBANJE (IMU)")
        self.imu_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.mic_frame = self.make_card(cards, "ZVOK (MIC)")
        self.mic_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))

        self.imu_label_var = tk.StringVar(value=" ")
        self.mic_label_var = tk.StringVar(value=" ")

        self.imu_lbl = tk.Label(
            self.imu_frame,
            textvariable=self.imu_label_var,
            font=("Menlo", 32, "bold"),
            fg=IDLE_COLOR,
            bg=CARD_BG,
        )
        self.imu_lbl.pack(pady=(4, 16))

        self.mic_lbl = tk.Label(
            self.mic_frame,
            textvariable=self.mic_label_var,
            font=("Menlo", 32, "bold"),
            fg=IDLE_COLOR,
            bg=CARD_BG,
        )
        self.mic_lbl.pack(pady=(4, 16))

        notif_frame = tk.Frame(self.root, bg=CARD_BG, pady=12)
        notif_frame.pack(fill=tk.X, padx=24, pady=20)

        tk.Label(
            notif_frame,
            text="STATUS",
            font=("Menlo", 9, "bold"),
            fg=IDLE_COLOR,
            bg=CARD_BG,
        ).pack()

        self.notif_var = tk.StringVar(value="Čakam na podatke ...")
        tk.Label(
            notif_frame,
            textvariable=self.notif_var,
            font=("Menlo", 13),
            fg="white",
            bg=CARD_BG,
            wraplength=400,
            justify=tk.CENTER,
        ).pack(pady=(4, 0))

        self.conn_var = tk.StringVar(value="Ni povezave")
        tk.Label(
            self.root,
            textvariable=self.conn_var,
            font=("Menlo", 9),
            fg=IDLE_COLOR,
            bg=BG,
            pady=8,
        ).pack(side=tk.BOTTOM)

    def make_card(self, parent, title):
        frame = tk.Frame(parent, bg=CARD_BG, pady=10, padx=10)
        tk.Label(
            frame, text=title, font=("Menlo", 9, "bold"), fg=IDLE_COLOR, bg=CARD_BG
        ).pack()
        return frame

    def set_imu(self, text, color):
        self.imu_label_var.set(text)
        self.imu_lbl.configure(fg=color)

    def set_mic(self, text, color):
        self.mic_label_var.set(text)
        self.mic_lbl.configure(fg=color)


def run():
    root = tk.Tk()
    GUI(root)
    root.mainloop()


if __name__ == "__main__":
    run()
