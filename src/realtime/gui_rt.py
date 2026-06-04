import queue
import threading
import tkinter as tk
from collections import deque
from pathlib import Path
import time
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
MIC_SEGMENT_SECONDS = 1
MIC_STFT_WINDOW_SECONDS = 0.032
MIC_STFT_OVERLAP = 0.5
MIC_RMS_THRESHOLD = 0.015
ACC_RESOLUTION = 1e-3
GYRO_RESOLUTION = 8.75e-3
MIC_MAXLEN = int(MIC_SEGMENT_SECONDS * MIC_SAMPLE_RATE)


IMU_PREDICT_INTERVAL_S = 3.0
MIC_PREDICT_INTERVAL_S = 1.0

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

    def schedule_refresh(self):
        self.root.after(self.REFRESH_RATE_MS, self.refresh)

    def refresh(self):
        while True:
            try:
                state = self.queue.get_nowait()
            except queue.Empty:
                break
            self.apply(state)
        self.schedule_refresh()

    def apply(self, state):
        t = state.get("type")

        if t == "connected":
            self.conn_var.set(f"Povezano: {state['port']}")

        elif t == "disconnected":
            self.conn_var.set("Ni povezave")

        elif t == "imu":
            label = state["label"]
            self.last_imu = label
            self.set_imu(label, CLASS_COLORS.get(label, IDLE_COLOR))
            self.update_notif()

        elif t == "mic":
            label = state.get("label")
            self.last_mic = label
            if label is None:
                self.set_mic("TIŠINA", IDLE_COLOR)
            else:
                self.set_mic(label, CLASS_COLORS.get(label, IDLE_COLOR))
            self.update_notif()

        elif t == "error":
            self.notif_var.set(f"Napaka: {state['msg']}")

    def update_notif(self):
        if self.last_imu is None:
            self.notif_var.set("Čakam na podatke ...")
            return
        if self.last_mic is None:
            self.notif_var.set(f"{self.last_imu} — tišina v ozadju")
            return
        notif = NOTIFICATIONS.get((self.last_imu, self.last_mic))
        if notif:
            self.notif_var.set(notif)
        else:
            self.notif_var.set(f"{self.last_imu} + {self.last_mic}")

    def start_thread(self):
        threading.Thread(target=self.prediction_loop, daemon=True).start()

    def prediction_loop(self):
        try:
            imu_model = load_imu_model()
            mic_model, log_min, log_max = load_mic_model()
        except Exception as e:
            self.queue.put({"type": "error", "msg": str(e)})
            return

        imu_prep = RealtimePreprocessor()
        mic_prep = MicRealtimePreprocessor(
            log_min=log_min,
            log_max=log_max,
            sample_rate=MIC_SAMPLE_RATE,
            segment_seconds=MIC_SEGMENT_SECONDS,
            stft_window_seconds=MIC_STFT_WINDOW_SECONDS,
            stft_overlap=MIC_STFT_OVERLAP,
            rms_threshold=MIC_RMS_THRESHOLD,
        )

        parser = LivePacketParser()
        reader = LiveSerialReader()

        signal_buffer = SignalBuffer(
            window_seconds=WINDOW_SECONDS,
            acc_sample_rate=ACC_SAMPLE_RATE,
            gyro_sample_rate=GYRO_SAMPLE_RATE,
            acc_resolution=ACC_RESOLUTION,
            gyro_resolution=GYRO_RESOLUTION,
        )
        mic_buf: deque = deque(maxlen=MIC_MAXLEN)
        last_imu_time = 0.0
        last_mic_time = 0.0

        try:
            for chunk in reader.read_stream():
                if reader.port:
                    self.queue.put({"type": "connected", "port": reader.port})

                for packet in parser.feed(chunk):
                    chunks = packet.get("chunks", {})

                    signal_buffer.add_packet(packet)

                    if ID_MIC in chunks:
                        mic_buf.extend(chunks[ID_MIC])

                    now = time.monotonic()

                    if now - last_imu_time > IMU_PREDICT_INTERVAL_S:
                        last_imu_time = now
                        acc_window, gyro_window = signal_buffer.get_window()
                        if acc_window is not None and gyro_window is not None:
                            result = imu_prep.process(acc_window, gyro_window)
                            if result is not None:
                                acc_t, gyro_t = result
                                with torch.no_grad():
                                    pred = imu_model(acc_t, gyro_t).argmax(dim=1).item()
                                self.queue.put(
                                    {"type": "imu", "label": IMU_CLASSES[pred]}
                                )

                    if now - last_mic_time > MIC_PREDICT_INTERVAL_S:
                        last_mic_time = now
                        samples = np.array(mic_buf, dtype=np.int8)
                        tensor, rms = mic_prep.process(samples)
                        if tensor is not None:
                            with torch.no_grad():
                                pred = mic_model(tensor).argmax(dim=1).item()
                            self.queue.put(
                                {"type": "mic", "label": MIC_CLASSES[pred], "rms": rms}
                            )
                        else:
                            self.queue.put({"type": "mic", "label": None, "rms": rms})

        except Exception:
            self.queue.put({"type": "disconnected"})


def run():

    root = tk.Tk()
    gui = GUI(root)
    gui.start_thread()
    root.mainloop()


if __name__ == "__main__":
    run()
