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
# kako velik je buffer lahko
ACC_MAXLEN = int(WINDOW_SECONDS * ACC_SAMPLE_RATE)
GYRO_MAXLEN = int(WINDOW_SECONDS * GYRO_SAMPLE_RATE)
MIC_MAXLEN = int(8.0 * MIC_SAMPLE_RATE)

ACC_RESOLUTION = 1e-3
GYRO_RESOLUTION = 8.75e-3

# tu mamo malo magic numbers, prvo gui, pole uskladimo
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

        self.queue = queue.Queue = queue.Queue()
        self.last_imu = None
        self.last_mic = None

        self.build()
        self.schedule_refresh()

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

        # rabimo dve kartici z napovedmi

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

    def schedule_refresh(self):
        self.root.after(self.REFRESH_RATE_MS, self.refresh)

    def refresh(self):
        while True:
            try:
                state = self.queue.get_nowait()
            except queue.Empty:
                break
            self.apply(state)
        self.schedule_refresh

    def set_imu(self, text, color):
        self.imu_label_var.set(text)
        self.imu_lbl.configure(fg=color)

    def set_mic(self, text, color):
        self.mic_label_var.set(text)
        self.mic_lbl.configure(fg=color)

    def start_thread(self):
        threading.Thread(target=self.prediction_loop, daemon=True).start()

    def prediction_loop(self):

        # nalozimo oba modela
        try:
            imu_model = load_imu_model()
            mic_model, log_min, log_max = load_mic_model()
        except Exception as e:
            self.notif_var.set(f"Napaka pri nalaganju modelov: {e}")
            return

        imu_prep = RealtimePreprocessor()
        mic_prer = MicRealtimePreprocessor(log_min=log_min, log_max=log_max)
        parser = LivePacketParser()
        reader = LiveSerialReader()

        acc_buf: deque = deque(maxlen=ACC_MAXLEN)
        gyro_buf: deque = deque(maxlen=GYRO_MAXLEN)
        mic_buf: deque = deque(maxlen=MIC_MAXLEN)
        n = 0

        import time

        t0 = time.time()
        n_acc = 0
        n_gyro = 0
        n_mic = 0

        try:
            for chunk in reader.read_stream():
                if reader.port:
                    self.queue.put({"type": "connected", "port": reader.port})

                for packet in parser.feed(chunk):
                    chunks = packet.get("chunks", {})
                    if ID_ACC in chunks:
                        n_acc += len(chunks[ID_ACC])

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
                        n_mic += len(chunks[ID_MIC])
                        mic_buf.extend(chunks[ID_MIC])

                    n += 1

        except Exception as e:
            pass

    def apply(self, state):
        pass


def run():
    root = tk.Tk()
    GUI(root)
    root.mainloop()


if __name__ == "__main__":
    run()
