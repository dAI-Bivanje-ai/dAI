"""
Skupne nastavitve za dAI.

Tu so vse številke, ki jih rabijo zajem, predobdelava, trening in realtime.
Če so na enem mestu, se trening in realtime ne moreta razlikovat.

Če spremeniš kaj pri predobdelavi (okna, frame-i, normalizacija), moraš oba
modela na novo naučit, drugače model dobi drugačen vhod kot pri treningu.
"""

from pathlib import Path

# poti do mape projekta in modelov
ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
IMU_MODEL_PATH = MODELS_DIR / "imu_cnn.pt"
MIC_MODEL_PATH = MODELS_DIR / "mic_cnn.pt"


# STM32 povezava

# USB id od STM32 virtual COM porta (vidiš ga z lsusb)
STM32_VID = "0483"
STM32_PID = "5740"

SERIAL_BAUDRATE = 115200
# koliko bajtov preberemo naenkrat
SERIAL_READ_SIZE = 2048
# koliko časa read() čaka na podatke
SERIAL_TIMEOUT_SECONDS = 1

# pavze med ukazi, ko zaganjamo stream
# pri 0.1 s STM32 izgubi CONFIG ukaz (mic ostane izklopljen), 0.2 s še dela
SERIAL_COMMAND_DELAY_SECONDS = 0.5  # po vsakem CONFIG ukazu
SERIAL_STREAM_DELAY_SECONDS = 0.5  # po OFF in po STREAM
# pavza preden se spet poskusimo povezat
SERIAL_RECONNECT_DELAY_SECONDS = 1


# paketi, ki jih pošilja STM32

# vsak paket se začne z 0xFF 0xFF
PACKET_SYNC_MARKER = b"\xff\xff"

# id-ji chunkov v paketu
CHUNK_ID_GYRO = 1
CHUNK_ID_ACC = 2
CHUNK_ID_MAG = 3
CHUNK_ID_MIC = 4

# če parser v toliko bajtih ne najde sync markerja, buffer izprazne
PARSER_MAX_BUFFER_BYTES = 200000


# senzorji

# frekvence, ki jih pričakujemo (izmerjeno je ACC okoli 24.7 Hz, GYRO okoli 105.6 Hz)
ACC_SAMPLE_RATE = 25.0
GYRO_SAMPLE_RATE = 105.0
MIC_SAMPLE_RATE = 8000.0

# s tem pomnožimo surove int16 vrednosti, da dobimo fizikalne enote
ACC_RESOLUTION = 1e-3  # g
GYRO_RESOLUTION = 8.75e-3  # stopinje na sekundo

# ACC in GYRO imata 3 osi (x, y, z)
IMU_AXES = 3


# IMU predobdelava, mora bit enaka pri treningu in v realtime

# eno FFT okno je približno 2 s
ACC_FFT_WINDOW = 49
GYRO_FFT_WINDOW = 211

# okna se prekrivajo za polovico, korak je pol okna: ACC 24, GYRO 106
IMU_FFT_OVERLAP = 0.5

# ena IMU slika sta 2 zaporedna FFT frame-a
# ACC: 49 + 24 = 73 vzorcev, približno 2.9 s
# GYRO: 211 + 106 = 317 vzorcev, približno 3.0 s
IMU_FRAMES_PER_IMAGE = 2


# MIC predobdelava, mora bit enaka pri treningu in v realtime

# band-pass: pod 80 Hz so vibracije, nad 3500 Hz je šum senzorja
MIC_BANDPASS_LOW_HZ = 80.0
MIC_BANDPASS_HIGH_HZ = 3500.0
MIC_BANDPASS_ORDER = 4

# STFT okno je 32 ms (256 vzorcev), okna se prekrivajo za polovico (korak 128)
MIC_STFT_WINDOW_SECONDS = 0.032
MIC_STFT_OVERLAP = 0.5
MIC_STFT_WINDOW_SAMPLES = round(MIC_STFT_WINDOW_SECONDS * MIC_SAMPLE_RATE)  # 256
MIC_STFT_HOP_SAMPLES = round(MIC_STFT_WINDOW_SAMPLES * (1 - MIC_STFT_OVERLAP))  # 128

# ena MIC slika je 186 frame-ov, da traja približno toliko kot IMU slika (3 s)
MIC_FRAMES_PER_IMAGE = 186
# prvi frame rabi 256 vzorcev, vsak naslednji še 128 več: 256 + 185 * 128 = 23936
MIC_IMAGE_SAMPLES = (
    MIC_STFT_WINDOW_SAMPLES + (MIC_FRAMES_PER_IMAGE - 1) * MIC_STFT_HOP_SAMPLES
)
MIC_IMAGE_SECONDS = MIC_IMAGE_SAMPLES / MIC_SAMPLE_RATE  # 2.992 s

# če je rms filtriranega zvoka pod tem pragom, je tišina in modela ne kličemo
MIC_RMS_SILENCE_THRESHOLD = 0.001


# normalizacija spektrogramov

# to prištejemo pred log10, ker log10(0) da -inf
LOG_EPSILON = 1e-10


# razredi in obvestila

IMU_CLASSES = {0: "DELO", 1: "TELEFON"}
MIC_CLASSES = {0: "GLASBA", 1: "POGOVOR"}
NUM_IMU_CLASSES = len(IMU_CLASSES)
NUM_MIC_CLASSES = len(MIC_CLASSES)

# TIŠINA ni razred modela, določi jo samo rms prag zgoraj
SILENCE_LABEL = "TIŠINA"

# katero opozorilo izpišemo za IMU + MIC razred
NOTIFICATIONS = {
    ("DELO", "GLASBA"): "Delaš, ampak glasba je preglasna",
    ("DELO", "POGOVOR"): "Delaš, ampak nekdo govori v ozadju",
    ("TELEFON", "GLASBA"): "Ne delaš ...",
    ("TELEFON", "POGOVOR"): "Ne delaš ...",
}


# realtime

# koliko sekund ACC in GYRO vzorcev drži SignalBuffer
IMU_BUFFER_SECONDS = 8.0

# buffer je pripravljen, ko ima vsaj 95 % vzorcev, ki jih pričakujemo
# (prava frekvenca je lahko malo nižja, npr. 24.7 Hz namesto 25 Hz)
BUFFER_READY_RATIO = 0.95

# model kličemo na vsakih N sekund, ne na vsak N-ti paket,
# ker firmware lahko pošilja različno velike pakete (STREAM ~120/s, snemanje ~10/s)
# stabilizator rabi 9 od 10 enakih napovedi, zato menjava traja okoli 3 s + 9 * N
IMU_PREDICT_EVERY_SECONDS = 0.1
MIC_PREDICT_EVERY_SECONDS = 0.8

# novo stanje sprejmemo šele, ko je vsaj 9 od zadnjih 10 napovedi enakih
STABLE_PREDICTION_WINDOW = 10
STABLE_PREDICTION_RATIO = 0.90


# spletna stran

# samo 127.0.0.1, da stran vidiš le na tem računalniku
WEB_HOST = "127.0.0.1"
# port 5000 na macOS zaseda AirPlay (ControlCenter)
WEB_PORT = 8000
