"""
Realtime grafični vmesnik za klasifikacijo aktivnosti.

Modul prikaže preprosto tkinter nadzorno ploščo z zadnjima napovedma za
gibanje (IMU) in zvok (MIC), statusom in seznamom časov aktivnosti.
Inferenca teče v ločeni niti, rezultati pa se v GUI prenašajo prek vrste
(queue).
"""

import queue
import threading
import tkinter as tk
from collections import deque
from pathlib import Path
import time
import numpy as np
import torch

import customtkinter as ctk
import random

from src.model.cnn_model import CNNModel as IMUModel
from src.model.cnn_model_mic import CNNModel as MicModel
from src.realtime.packet_parser import LivePacketParser, ID_MIC
from src.realtime.preproc_rt import MicRealtimePreprocessor, RealtimePreprocessor
from src.realtime.serial_reader import LiveSerialReader
from src.realtime.signal_buffer import SignalBuffer
from src.realtime.prediction_stabilizer import PredictionStabilizer

from src.realtime.activity_timer import ActivityTimer
from src.realtime.activity_viewer import ActivityViewer
from src.realtime.notifier import notify

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
MIC_RMS_THRESHOLD = 0.009
ACC_RESOLUTION = 1e-3
GYRO_RESOLUTION = 8.75e-3
MIC_MAXLEN = int(MIC_SEGMENT_SECONDS * MIC_SAMPLE_RATE)


# Kako pogosto najmanj izvedemo posamezno napoved (v sekundah)
IMU_PREDICT_INTERVAL_S = 3.0
MIC_PREDICT_INTERVAL_S = 1.0

# Barve za posamezne razrede in osnovna paleta vmesnika
CLASS_COLORS = {
    "DELO": "#3ee6b0",      # neon zelena / cyan
    "TELEFON": "#ff4d5e",   # rdeča
    "GLASBA": "#4d9fff",    # modra
    "POGOVOR": "#a855f7",   # vijolična
}
IDLE_COLOR = "#8b8ba0"

# Barve za donut / legendo / časovne vrstice 
DONUT_COLORS = {**CLASS_COLORS, "TIŠINA": "#5a5a72"}

BG = "#07080d"          # skoraj črno ozadje 
CARD_BG = "#12131c"     # temno modro-črne kartice
TEXT = "#f5f5fa"        # primarni tekst
TEXT_SEC = "#8b8ba0"    # sekundarni tekst
ACCENT = "#a855f7"      # neon vijolični accent (naslov)
STAR_COLORS = ["#ffffff", "#ffffff", "#d7d7ff", "#c9b8ff", "#a855f7"]

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

def load_imu_model():
    """
    Naloži naučen IMU model in ga preklopi v eval način.
    """
    model = IMUModel(num_classes=2)
    model.load_state_dict(torch.load(str(IMU_MODEL_PATH), map_location="cpu"))
    model.eval()
    return model


def load_mic_model():
    """
    Naloži mikrofonski model in normalizacijska parametra (log_min, log_max).
    """
    checkpoint = torch.load(str(MIC_MODEL_PATH), map_location="cpu")
    model = MicModel(num_classes=2)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, float(checkpoint["log_min"]), float(checkpoint["log_max"])


class GUI:
    """
    Nadzorna plošča realtime sistema (tkinter).

    Sestavi vmesnik, redno prebira vrsto z rezultati inference iz delovne
    niti in posodablja prikaz: napovedi, status in čase aktivnosti.
    """

    # Kako pogosto osvežimo vmesnik (v milisekundah)
    REFRESH_RATE_MS = 200
    WIN_W = 660
    WIN_H = 820

    def __init__(self, root: tk.Tk):
        """
        Pripravi okno, stanje in časovnike ter sestavi vmesnik.

        Args:
            root: tk.Tk — glavno okno aplikacije
        """
        self.root = root
        self.root.title("dAI")
        self.root.configure(fg_color=BG)
        self.root.resizable(False, False)

        # Vrsta, prek katere delovna nit pošilja rezultate v GUI nit
        self.queue: queue.Queue = queue.Queue()
        # Zadnji znani napovedi in zadnje prikazano obvestilo
        self.last_imu = None
        self.last_mic = None
        self.last_notif_text = None

        # Časovnika, ki merita trajanje posameznih aktivnosti
        self.imu_timer = ActivityTimer()
        self.mic_timer = ActivityTimer()

        # Spremljevalnik aktivne aplikacije (produktivno / neproduktivno)
        self.last_activity = None
        self.activity_viewer = ActivityViewer()

        self.build()
        self.schedule_refresh()
    
    def _make_fonts(self):
        # geometrijski sans (Avenir Next) za besedilo
        # Menlo (mono) samo za poravnane številčne stolpce.
        sans = "Avenir Next"
        self.font_card_title = ctk.CTkFont(family=sans, size=12, weight="bold")
        self.font_value = ctk.CTkFont(family=sans, size=34, weight="bold")
        self.font_status = ctk.CTkFont(family=sans, size=14)
        self.font_label = ctk.CTkFont(family=sans, size=13)
        self.font_small = ctk.CTkFont(family=sans, size=12)
        self.font_mono = ctk.CTkFont(family="Menlo", size=12)


    def build(self):
        W, H = self.WIN_W, self.WIN_H
        self.root.geometry(f"{W}x{H}")
        self._make_fonts()

        cx = W // 2
        margin = 24
        content_w = W - 2 * margin
        gap = 14

        # zvezdno ozadje
        self.bg_canvas = tk.Canvas(
            self.root, bg=BG, highlightthickness=0, bd=0
        )
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        self._draw_starfield(W, H)

        y = 24

        # naslov
        self.bg_canvas.create_text(
            cx, y, anchor="n", text="dAI", fill=TEXT,
            font=("Avenir Next", 30, "bold"),
        )
        self.bg_canvas.create_text(
            cx, y + 42, anchor="n",
            text="Realtime klasifikacija aktivnosti",
            fill=TEXT_SEC, font=("Avenir Next", 12),
        )
        # neon accent črta pod naslovom
        self.bg_canvas.create_line(
            cx - 60, y + 64, cx + 60, y + 64, fill=ACCENT, width=2
        )
        y += 84

        # kartici GIBANJE / ZVOK 
        card_h = 150
        card_w = (content_w - gap) // 2
        left = cx - content_w // 2
        right = cx + content_w // 2

        self.imu_frame, self.imu_value_lbl, self.imu_underline = self.make_card(
            "GIBANJE (IMU)"
        )
        self.bg_canvas.create_window(
            left, y, anchor="nw", window=self.imu_frame,
            width=card_w, height=card_h,
        )
        self.mic_frame, self.mic_value_lbl, self.mic_underline = self.make_card(
            "ZVOK (MIC)"
        )
        self.bg_canvas.create_window(
            right, y, anchor="ne", window=self.mic_frame,
            width=card_w, height=card_h,
        )
        y += card_h + gap

        # STATUS panel 
        self.notif_var = tk.StringVar(value="Čakam na podatke ...")
        status = ctk.CTkFrame(self.root, corner_radius=16, fg_color=CARD_BG)
        ctk.CTkLabel(
            status, text="STATUS", font=self.font_card_title,
            text_color=TEXT_SEC,
        ).pack(pady=(14, 0))
        ctk.CTkLabel(
            status, textvariable=self.notif_var, font=self.font_status,
            text_color=TEXT, wraplength=content_w - 60, justify="center",
        ).pack(expand=True, pady=(2, 4))
        # Trenutna aktivna aplikacija in njena kategorija
        self.activity_var = tk.StringVar(value="Aktivnost: -")
        ctk.CTkLabel(
            status, textvariable=self.activity_var, font=self.font_mono,
            text_color=TEXT_SEC,
        ).pack(pady=(0, 12))
        status_h = 112
        self.bg_canvas.create_window(
            cx, y, anchor="n", window=status,
            width=content_w, height=status_h,
        )
        y += status_h + gap

        # tortni diagram za prikaz razporeditve časa 
        donut_card = ctk.CTkFrame(self.root, corner_radius=16, fg_color=CARD_BG)
        ctk.CTkLabel(
            donut_card, text="RAZPOREDITEV ČASA", font=self.font_card_title,
            text_color=TEXT_SEC,
        ).pack(pady=(12, 2))
        donut_body = ctk.CTkFrame(donut_card, fg_color="transparent")
        donut_body.pack(fill="both", expand=True, pady=(0, 10))
        self.donut_canvas = tk.Canvas(
            donut_body, width=160, height=160, bg=CARD_BG,
            highlightthickness=0, bd=0,
        )
        self.donut_canvas.pack(side="left", padx=(26, 12))
        self.legend_frame = ctk.CTkFrame(donut_body, fg_color="transparent")
        self.legend_frame.pack(
            side="left", fill="both", expand=True, padx=(12, 24)
        )
        donut_h = 222
        self.bg_canvas.create_window(
            cx, y, anchor="n", window=donut_card,
            width=content_w, height=donut_h,
        )
        y += donut_h + gap

        # kartica časov aktivnosti
        times_card = ctk.CTkFrame(self.root, corner_radius=16, fg_color=CARD_BG)
        ctk.CTkLabel(
            times_card, text="ČASI AKTIVNOSTI", font=self.font_card_title,
            text_color=TEXT_SEC,
        ).pack(pady=(12, 2))
        self.times_body = ctk.CTkFrame(times_card, fg_color="transparent")
        self.times_body.pack(
            fill="both", expand=True, padx=24, pady=(2, 12)
        )
        times_h = 172
        self.bg_canvas.create_window(
            cx, y, anchor="n", window=times_card,
            width=content_w, height=times_h,
        )

        # conn_var (apply() ga nastavlja ob connect/disconnect)
        self.conn_var = tk.StringVar(value="Ni povezave")

        # prvi izris donuta in tabele časov
        self.update_time_display()

        

        

    def make_card(self, title):
        """ 
        Ustvari dashboard kartico;
        vrne (frame, value_label, underline).
        """
        frame = ctk.CTkFrame(self.root, corner_radius=16, fg_color=CARD_BG)
        ctk.CTkLabel(
            frame, text=title, font=self.font_card_title, text_color=TEXT_SEC
        ).pack(pady=(16, 0))
        value = ctk.CTkLabel(
            frame, text="—", font=self.font_value, text_color=IDLE_COLOR
        )
        value.pack(expand=True)
        underline = ctk.CTkFrame(
            frame, height=4, corner_radius=2, fg_color=IDLE_COLOR
        )
        underline.pack(fill="x", padx=34, pady=(0, 18))
        return frame, value, underline
    
    def _draw_starfield(self, W, H):
        self._stars = []
        for _ in range(random.randint(40, 60)):
            x = random.randint(0, W)
            y = random.randint(0, H)
            r = random.choice([1, 1, 1, 2, 2, 3])
            color = random.choice(STAR_COLORS)
            sid = self.bg_canvas.create_oval(
                x - r, y - r, x + r, y + r, fill=color, outline=""
            )
            self._stars.append((sid, color))
        self._twinkle()

    def _twinkle(self):
        for sid, color in self._stars:
            if random.random() < 0.12:
                dim = random.random() < 0.5
                self.bg_canvas.itemconfigure(
                    sid, fill="#3a3a55" if dim else color
                )
        self.root.after(700, self._twinkle)

    def set_imu(self, text, color):
        """
        Posodobi besedilo in barvo IMU kartice.
        """
        self.imu_label_var.set(text)
        self.imu_lbl.configure(fg=color)

    def set_imu(self, text, color):
        self.imu_value_lbl.configure(text=text, text_color=color)
        self.imu_underline.configure(fg_color=color)

    def set_mic(self, text, color):
        self.mic_value_lbl.configure(text=text, text_color=color)
        self.mic_underline.configure(fg_color=color)

    def schedule_refresh(self):
        """
        Načrtuje naslednji periodični klic refresh().
        """
        self.root.after(self.REFRESH_RATE_MS, self.refresh)

    def refresh(self):
        """
        Obdela vse čakajoče dogodke iz vrste in znova načrtuje osvežitev.
        """
        # Izpraznimo vrsto, dokler so v njej novi dogodki
        while True:
            try:
                state = self.queue.get_nowait()
            except queue.Empty:
                break
            self.apply(state)
        self.schedule_refresh()

    def apply(self, state):
        """
        Uveljavi en dogodek iz vrste na prikazu.

        Glede na tip dogodka (connected, disconnected, imu, mic, error)
        posodobi ustrezne kartice, časovnike, status in obvestila.

        Args:
            state: dict — dogodek iz delovne niti
        """
        t = state.get("type")

        if t == "connected":
            self.conn_var.set(f"Povezano: {state['port']}")

        elif t == "disconnected":
            self.conn_var.set("Ni povezave")

        elif t == "imu":
            label = state["label"]
            self.last_imu = label

            self.imu_timer.update(label)
            self.update_time_display()

            self.set_imu(label, CLASS_COLORS.get(label, IDLE_COLOR))
            self.update_notif()

        elif t == "mic":
            label = state.get("label")
            self.last_mic = label

            # Brez razreda obravnavamo kot tišino
            if label is None:
                display_label = "TIŠINA"
                self.set_mic(display_label, IDLE_COLOR)
            else:
                display_label = label
                self.set_mic(label, CLASS_COLORS.get(label, IDLE_COLOR))

            self.mic_timer.update(display_label)
            self.update_time_display()

            self.update_notif()

        elif t == "activity":
            self.last_activity = state
            app = state.get("app") or "?"
            self.activity_var.set(f"Aktivnost: {app} · {state['label']}")

        elif t == "error":
            self.notif_var.set(f"Napaka: {state['msg']}")

    def update_notif(self):
        """
        Sestavi besedilo statusa iz zadnjih IMU in MIC oznak.

        Posodobi prikazani status, ob dejanski spremembi pa sproži tudi
        sistemsko notifikacijo.
        """
        if self.last_imu is None:
            text = "Čakam na podatke ..."
        elif self.last_mic is None:
            text = f"{self.last_imu} — tišina v ozadju"
        else:
            notif = NOTIFICATIONS.get((self.last_imu, self.last_mic))
            text = notif if notif else f"{self.last_imu} + {self.last_mic}"

        self.notif_var.set(text)

        # le ob dejanski spremembi stanja
        if text != self.last_notif_text and text != "Čakam na podatke ...":
            notify("dAI", text)

        self.last_notif_text = text

    def format_seconds(self, seconds: float) -> str:
        """
        Pretvori sekunde v zapis MM:SS.
        """
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def update_time_display(self):
        """
        Sestavi in osveži besedilni prikaz časov IMU in MIC aktivnosti.
        """
        imu_times = self.imu_timer.get_durations()
        mic_times = self.mic_timer.get_durations()

        combined = {}
        for durations in (imu_times, mic_times):
            for label, seconds in durations.items():
                combined[label] = combined.get(label, 0.0) + seconds

        self._draw_donut(combined)
        self._render_times(imu_times, mic_times)
    
    def _draw_donut(self, combined):
        c = self.donut_canvas
        c.delete("all")
        bbox = (8, 8, 152, 152)
        total = sum(combined.values())

        if total <= 0:
            c.create_oval(*bbox, outline=IDLE_COLOR, width=16)
        else:
            start = 90.0
            for label, seconds in combined.items():
                if seconds <= 0:
                    continue
                extent = -360.0 * (seconds / total)
                if abs(extent) >= 360.0:
                    extent = -359.99
                c.create_arc(
                    *bbox, start=start, extent=extent,
                    fill=DONUT_COLORS.get(label, IDLE_COLOR),
                    outline=CARD_BG, width=1, style=tk.PIESLICE,
                )
                start += extent

        # luknja v sredini -> donut videz
        c.create_oval(44, 44, 116, 116, fill=CARD_BG, outline=CARD_BG)
        c.create_text(
            80, 80, text=self.format_seconds(total), fill=TEXT,
            font=("Menlo", 15, "bold"),
        )

        # legenda
        for child in self.legend_frame.winfo_children():
            child.destroy()
        items = combined.items() if total > 0 else []
        for row, (label, seconds) in enumerate(items):
            pct = 100.0 * seconds / total
            ctk.CTkLabel(
                self.legend_frame, text="●", font=self.font_small,
                text_color=DONUT_COLORS.get(label, IDLE_COLOR),
            ).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=1)
            ctk.CTkLabel(
                self.legend_frame, text=label, font=self.font_label,
                text_color=TEXT,
            ).grid(row=row, column=1, sticky="w", pady=1)
            ctk.CTkLabel(
                self.legend_frame, text=f"{pct:.0f}%", font=self.font_mono,
                text_color=TEXT_SEC,
            ).grid(row=row, column=2, sticky="e", padx=(16, 0), pady=1)
        self.legend_frame.grid_columnconfigure(1, weight=1)

    def _render_times(self, imu_times, mic_times):
        body = self.times_body
        for child in body.winfo_children():
            child.destroy()

        body.grid_columnconfigure(0, weight=1, uniform="cols")
        body.grid_columnconfigure(1, weight=1, uniform="cols")
        body.grid_rowconfigure(0, weight=1)

        self._time_column(body, 0, "GIBANJE (IMU)", imu_times, (0, 16))
        self._time_column(body, 1, "ZVOK (MIC)", mic_times, (16, 0))

    def _time_column(self, parent, col, title, durations, padx):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=0, column=col, sticky="new", padx=padx)
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text=title, font=self.font_small, text_color=TEXT_SEC
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))

        if not durations:
            ctk.CTkLabel(
                frame, text="—", font=self.font_label, text_color=TEXT_SEC
            ).grid(row=1, column=1, sticky="w")
            return

        for i, (label, seconds) in enumerate(durations.items(), start=1):
            ctk.CTkLabel(
                frame, text="●", font=self.font_small,
                text_color=DONUT_COLORS.get(label, IDLE_COLOR),
            ).grid(row=i, column=0, sticky="w", padx=(0, 8), pady=1)
            ctk.CTkLabel(
                frame, text=label, font=self.font_label, text_color=TEXT
            ).grid(row=i, column=1, sticky="w", pady=1)
            ctk.CTkLabel(
                frame, text=self.format_seconds(seconds),
                font=self.font_mono, text_color=TEXT_SEC,
            ).grid(row=i, column=2, sticky="e", pady=1)

    def start_thread(self):
        """
        Zažene inferenco in spremljanje aktivnosti v ločenih daemon nitih.
        """
        threading.Thread(target=self.prediction_loop, daemon=True).start()
        threading.Thread(target=self.activity_loop, daemon=True).start()

    def activity_loop(self):
        """
        Delovna nit: bere dogodke o spremembi aktivne aplikacije in jih
        pošilja v GUI nit prek vrste (queue).
        """
        for event in self.activity_viewer.run():
            self.queue.put({"type": "activity", **event})

    def prediction_loop(self):
        """
        Delovna nit: bere tok, izvaja inferenco in pošilja rezultate v vrsto.

        Naloži modele in preprocessorje, periodično izvaja IMU in MIC
        napovedi ter stanje sporoča GUI niti prek vrste (queue).
        """
        # Naložimo oba modela; ob napaki sporočimo GUI in končamo nit
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
        # Mikrofon hranimo v ločenem bufferju, ker je 1D
        mic_buf: deque = deque(maxlen=MIC_MAXLEN)

        # Stabilizator zgladi utripanje mikrofonskih napovedi
        mic_stabilizer = PredictionStabilizer(window_size=5, min_ratio=0.60)

        # Čas zadnje izvedene napovedi za vsak senzor
        last_imu_time = 0.0
        last_mic_time = 0.0

        try:
            # Glavna zanka: beremo tok bajtov iz naprave
            for chunk in reader.read_stream():
                if reader.port:
                    self.queue.put({"type": "connected", "port": reader.port})

                # Iz bajtov sestavimo pakete in jih obdelamo
                for packet in parser.feed(chunk):
                    chunks = packet.get("chunks", {})

                    signal_buffer.add_packet(packet)

                    if ID_MIC in chunks:
                        mic_buf.extend(chunks[ID_MIC])

                    now = time.monotonic()

                    # IMU napoved izvedemo periodično, ne na vsak paket
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

                    # MIC napoved izvajamo redkeje in le, ko imamo vzorce
                    if now - last_mic_time > MIC_PREDICT_INTERVAL_S and mic_buf:
                        last_mic_time = now
                        samples = np.array(mic_buf, dtype=np.int8)
                        tensor, rms = mic_prep.process(samples)
                        if tensor is not None:
                            with torch.no_grad():
                                pred = mic_model(tensor).argmax(dim=1).item()
                            # V GUI pošljemo samo potrjeno (stabilno) napoved
                            stable = mic_stabilizer.update(MIC_CLASSES[pred])
                            if stable is not None:
                                self.queue.put(
                                    {
                                        "type": "mic",
                                        "label": stable,
                                        "rms": rms,
                                    }
                                )
                        else:
                            self.queue.put({"type": "mic", "label": None, "rms": rms})

        except Exception:
            # Ob napaki ali odklopu naprave sporočimo GUI
            self.queue.put({"type": "disconnected"})


def run():

    root = ctk.CTk()
    gui = GUI(root)
    gui.start_thread()
    root.mainloop()


if __name__ == "__main__":
    run()
