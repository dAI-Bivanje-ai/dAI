"""
Glavna realtime zanka za klasifikacijo aktivnosti iz STM32 toka (konzola).

Modul poveže celotno verigo: branje serijskega toka, razčlenjevanje paketov,
polnjenje bufferjev, periodično inferenco IMU in mikrofonskega modela,
stabilizacijo napovedi in izpis stanja s pripadajočimi opozorili.
"""

import logging
import time
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
import torch

from src.config import (
    ACC_RESOLUTION,
    ACC_SAMPLE_RATE,
    BUFFER_READY_RATIO,
    CHUNK_ID_MIC,
    GYRO_RESOLUTION,
    GYRO_SAMPLE_RATE,
    IMU_BUFFER_SECONDS,
    IMU_CLASSES,
    IMU_MODEL_PATH,
    IMU_PREDICT_EVERY_SECONDS,
    MIC_CLASSES,
    MIC_IMAGE_SECONDS,
    MIC_MODEL_PATH,
    MIC_PREDICT_EVERY_SECONDS,
    MIC_RMS_SILENCE_THRESHOLD,
    MIC_SAMPLE_RATE,
    MIC_STFT_OVERLAP,
    MIC_STFT_WINDOW_SECONDS,
    NOTIFICATIONS,
    NUM_MIC_CLASSES,
    SILENCE_LABEL,
    STABLE_PREDICTION_RATIO,
    STABLE_PREDICTION_WINDOW,
)
from src.model.cnn_model import CNNModel as IMUModel, load_imu_checkpoint
from src.model.cnn_model_mic import CNNModel as MicModel
from src.realtime.packet_parser import LivePacketParser
from src.realtime.preproc_rt import MicRealtimePreprocessor, RealtimePreprocessor
from src.realtime.serial_reader import LiveSerialReader
from src.realtime.signal_buffer import SignalBuffer


@dataclass(frozen=True)
class RealtimeConfig:
    """
    Nastavitve za realtime zanko.

    Vrednosti pridejo iz src/config.py, tu so samo zbrane skupaj.
    """

    # frekvence senzorjev, ki jih pričakujemo
    acc_sample_rate: float = ACC_SAMPLE_RATE
    gyro_sample_rate: float = GYRO_SAMPLE_RATE
    mic_sample_rate: float = MIC_SAMPLE_RATE

    # koliko sekund ACC in GYRO vzorcev drži SignalBuffer
    imu_window_seconds: float = IMU_BUFFER_SECONDS

    # dolžina enega mic segmenta, mora bit enaka kot pri treningu (186 frame-ov)
    mic_segment_seconds: float = MIC_IMAGE_SECONDS

    # STFT za mikrofon: 0.032 s pri 8000 Hz je 256 vzorcev
    mic_stft_window_seconds: float = MIC_STFT_WINDOW_SECONDS
    # okna se prekrivajo za polovico
    mic_stft_overlap: float = MIC_STFT_OVERLAP

    # če je rms pod tem pragom, je tišina in mic modela ne kličemo
    mic_rms_threshold: float = MIC_RMS_SILENCE_THRESHOLD

    # s tem pomnožimo surove int16 vrednosti, da dobimo fizikalne enote
    acc_resolution: float = ACC_RESOLUTION
    gyro_resolution: float = GYRO_RESOLUTION

    # buffer je pripravljen, ko ima vsaj 95 % vzorcev, ki jih pričakujemo
    # (prava frekvenca je lahko malo nižja, npr. 24.7 Hz namesto 25 Hz)
    buffer_ready_ratio: float = BUFFER_READY_RATIO

    # model kličemo samo na vsak N-ti paket, ker je predobdelava počasna
    imu_predict_every_seconds: float = IMU_PREDICT_EVERY_SECONDS
    mic_predict_every_seconds: float = MIC_PREDICT_EVERY_SECONDS

    # novo stanje sprejmemo šele, ko je dovolj zadnjih napovedi enakih
    stable_prediction_window: int = STABLE_PREDICTION_WINDOW
    stable_prediction_ratio: float = STABLE_PREDICTION_RATIO


CONFIG = RealtimeConfig()


class PredictionStabilizer:
    """
    Stabilizira realtime napovedi modela.

    Hrani zadnjih N napovedi in potrdi razred šele,
    ko dovolj velik delež napovedi kaže na isti razred.
    """

    def __init__(self, window_size: int, min_ratio: float) -> None:
        """
        Inicializira stabilizator.

        Args:
            window_size: int - velikost drsečega okna zadnjih napovedi
            min_ratio: float - najmanjši delež enakih napovedi za potrditev
        """
        # Sliding window zadnjih N predictionov.
        self.window: deque[str] = deque(maxlen=window_size)

        # Minimalni delež enakih predictionov 90 %.
        self.min_ratio = min_ratio

        # Zadnji že potrjeni razred.
        # To prepreči, da bi isti stabilen razred izpisovali znova in znova.
        self.last_confirmed: str | None = None

    def update(self, label: str | None) -> str | None:
        """
        Doda novo napoved in vrne stabilen razred, če je dovolj zanesljiv.

        Vrne:
        - None, če še ni dovolj napovedi ali stanje ni stabilno,
        - ime razreda, če je potrjen nov stabilen razred.
        """

        if label is None:
            return None

        self.window.append(label)

        # Dokler okno ni polno, še ne odločamo.
        if len(self.window) < self.window.maxlen:
            return None

        # Preštejemo, kateri razred je najpogostejši v zadnjih N napovedih.
        label_counts = Counter(self.window)
        most_common_label, count = label_counts.most_common(1)[0]

        ratio = count / len(self.window)

        # Če najpogostejši razred ne doseže praga, stanje še ni stabilno.
        if ratio < self.min_ratio:
            return None

        # Če je to isti razred kot zadnjič, ga ne izpišemo ponovno.
        if most_common_label == self.last_confirmed:
            return None

        self.last_confirmed = most_common_label
        return most_common_label


# Mikrofon hrani zadnjih ~3 sekunde zvoka (en segment).
# 2.992 s * 8000 Hz = 23936 vzorcev.
MIC_MAXLEN = int(CONFIG.mic_segment_seconds * CONFIG.mic_sample_rate)


def load_mic_model() -> tuple[MicModel, float, float]:
    """
    Naloži za mikrofon naučen model.

    Poleg uteži modela naložimo še log_min in log_max.
    Ti vrednosti sta potrebni, da realtime spektrogram normaliziramo enako kot pri treningu.
    """

    checkpoint = torch.load(str(MIC_MODEL_PATH), map_location="cpu")
    model = MicModel(num_classes=NUM_MIC_CLASSES)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    log_min = float(checkpoint["log_min"])
    log_max = float(checkpoint["log_max"])
    return model, log_min, log_max


def run_imu_inference(
    model: IMUModel,
    preprocessor: RealtimePreprocessor,
    acc_window: np.ndarray | None,
    gyro_window: np.ndarray | None,
) -> str | None:
    """
    Izvede klasifikacijo aktivnosti iz ACC in GYRO okna.

    Vhod:
    - acc_window: zadnje ACC okno oblike (N, 3)
    - gyro_window: zadnje GYRO okno oblike (N, 3)

    Če buffer še nima dovolj podatkov, dobimo None in inference preskočimo.
    """
    if acc_window is None or gyro_window is None:
        return None

    # RealtimePreprocessor vrne pripravljene vhode za model - tensorje
    result = preprocessor.process(acc_window, gyro_window)
    if result is None:
        return None
    acc_t, gyro_t = result

    with torch.no_grad():
        output = model(acc_t, gyro_t)
        pred = output.argmax(dim=1).item()

    return IMU_CLASSES[pred]


def run_mic_inference(
    model: MicModel,
    preprocessor: MicRealtimePreprocessor,
    mic_buf: deque,
) -> tuple[str | None, float]:
    """
    Izvede klasifikacijo zvoka iz mikrofonskega bufferja.

    Mikrofon ima ločen buffer, ker njegovi podatki niso trojice x,y,z,
    ampak 1D audio vzorci.
    """

    # Če ni vzorcev inference preskočimo
    if len(mic_buf) < MIC_MAXLEN:
        return None, 0.0

    # Deque pretvorimo v NumPy array.
    # Mikrofon je A-law encoded, zato je dtype int8.
    samples = np.array(mic_buf, dtype=np.int8)
    tensor, rms = preprocessor.process(samples)

    # Če je signal tišina ali še ni dovolj dolg, preprocessor vrne None.
    if tensor is None:
        return SILENCE_LABEL, rms

    # Model uporabimo samo za napoved, zato gradienti niso potrebni.
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()

    # Številčni razred pretvorimo v tekstovno oznako.
    return MIC_CLASSES[pred], rms


def print_status(imu_label: str | None, mic_label: str | None, rms: float) -> None:
    """
    Izpiše trenutno stanje sistema.

    Združi zadnjo IMU napoved in zadnjo MIC napoved.
    Če obstaja opozorilo za to kombinacijo, ga doda na konec izpisa.
    """
    # Če IMU še ni dal napovedi, izpišemo vprašaj.
    imu_str = imu_label or "?"

    # Če mikrofon nima razreda, to obravnavamo kot tišino.
    if mic_label is None:
        mic_str = SILENCE_LABEL
        notif_str = ""
    else:
        mic_str = mic_label
        # Poiščemo obvestilo za kombinacijo IMU + MIC.
        notif = NOTIFICATIONS.get((imu_str, mic_label), "")
        notif_str = f"  →  {notif}" if notif else ""

    print(f"IMU={imu_str}  MIC={mic_str}  (rms={rms:.4f}){notif_str}")


def run_pipeline(on_update, reader):
    """
    Glavna realtime zanka.

    Tukaj:
    - naložimo modele,
    - pripravimo preprocessors,
    - odpremo realtime reader,
    - parsamo pakete,
    - polnimo bufferje,
    - periodično izvajamo inference.
    """

    # Nastavimo osnovni logging, da vidimo informacije o nalaganju in povezavi.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Naložimo IMU model.
    imu_model, log_ranges = load_imu_checkpoint(IMU_MODEL_PATH)
    logging.info("IMU model naložen: %s", IMU_MODEL_PATH)

    # Naložimo mikrofonski model in normalizacijske vrednosti.
    mic_model, log_min, log_max = load_mic_model()
    logging.info("Mic model naložen: %s", MIC_MODEL_PATH)

    # Preprocessor za ACC/GYRO pripravi spektrograme za IMU model.
    imu_preprocessor = RealtimePreprocessor(log_ranges)
    # Preprocessor za mikrofon, ki uporablja iste nastavitve kot training.
    mic_preprocessor = MicRealtimePreprocessor(
        log_min=log_min,
        log_max=log_max,
        sample_rate=CONFIG.mic_sample_rate,
        segment_seconds=CONFIG.mic_segment_seconds,
        stft_window_seconds=CONFIG.mic_stft_window_seconds,
        stft_overlap=CONFIG.mic_stft_overlap,
        rms_threshold=CONFIG.mic_rms_threshold,
    )

    # Parser iz surovih bajtov naredi strukturirane pakete.
    parser = LivePacketParser()
    # Reader skrbi za serial povezavo s STM32 in vrača bajte
    # SignalBuffer skrbi za ACC in GYRO.
    # Sam pobere pravilne chunke, pretvori vrednosti v fizikalne enote in hrani ločeni okni za ACC in GYRO
    signal_buffer = SignalBuffer(
        window_seconds=CONFIG.imu_window_seconds,
        acc_sample_rate=CONFIG.acc_sample_rate,
        gyro_sample_rate=CONFIG.gyro_sample_rate,
        acc_resolution=CONFIG.acc_resolution,
        gyro_resolution=CONFIG.gyro_resolution,
        ready_ratio=CONFIG.buffer_ready_ratio,
    )
    # Mikrofon ostane v ločenem bufferju ker je 1D.
    mic_buf: deque = deque(maxlen=MIC_MAXLEN)

    # Ustvarimo stabilizatorja
    imu_stabilizer = PredictionStabilizer(
        window_size=CONFIG.stable_prediction_window,
        min_ratio=CONFIG.stable_prediction_ratio,
    )

    mic_stabilizer = PredictionStabilizer(
        window_size=CONFIG.stable_prediction_window,
        min_ratio=CONFIG.stable_prediction_ratio,
    )

    # kdaj smo nazadnje klicali vsak model (monotonic ne skoči, ko se ura sinhronizira)
    last_imu_time = time.monotonic()
    last_mic_time = time.monotonic()
    # Hranimo zadnje oznake za izpis
    last_imu_label: str | None = None
    last_mic_label: str | None = None
    last_rms = 0.0

    try:
        # prehod po blokih bajtov iz serial porta
        for chunk in reader.read_stream():
            # Parser lahko iz enega bloka dobi 0, 1 ali več kompletnih paketov.
            packets = parser.feed(chunk)

            for packet in packets:
                # chunks je slovar senzorskih podatkov znotraj paketa.
                # Ključi so CHUNK_ID_ACC, CHUNK_ID_GYRO, CHUNK_ID_MIC.
                chunks = packet.get("chunks", {})

                # ACC in GYRO vzorce damo v SignalBuffer.
                signal_buffer.add_packet(packet)

                # Mikrofon ostane posebej, ker 1D
                if CHUNK_ID_MIC in chunks:
                    mic_buf.extend(chunks[CHUNK_ID_MIC])

                now = time.monotonic()

                # IMU napoved na vsakih imu_predict_every_seconds sekund
                if now - last_imu_time >= CONFIG.imu_predict_every_seconds:
                    last_imu_time = now

                    acc_window, gyro_window = signal_buffer.get_window()

                    imu_label = run_imu_inference(
                        imu_model,
                        imu_preprocessor,
                        acc_window,
                        gyro_window,
                    )
                    stable_imu_label = imu_stabilizer.update(imu_label)
                    # Če model vrne oznako, jo shranimo kot zadnje znano IMU stanje.
                    if stable_imu_label is not None:
                        last_imu_label = stable_imu_label
                        on_update(last_imu_label, last_mic_label, last_rms)
                # MIC napoved redkeje, ker je mikrofonski segment daljši
                if now - last_mic_time >= CONFIG.mic_predict_every_seconds:
                    last_mic_time = now

                    mic_label, rms = run_mic_inference(
                        mic_model, mic_preprocessor, mic_buf
                    )
                    last_rms = rms
                    stable_mic_label = mic_stabilizer.update(mic_label)

                    if stable_mic_label is not None:
                        last_mic_label = stable_mic_label
                        on_update(last_imu_label, last_mic_label, last_rms)

    except KeyboardInterrupt:
        # CTRL + C
        print("\nUstavljanje...")
    finally:
        # izpis statisktike parserja
        stats = parser.stats()
        print(
            f"\nPaketi: {stats['valid_packets']} veljavnih / {stats['total_packets']} skupaj"
        )
        reader.stop()


if __name__ == "__main__":
    reader = LiveSerialReader()
    # print("run_pipeline zagnan")
    run_pipeline(print_status, reader)
