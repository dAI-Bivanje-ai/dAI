import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from visualization.data_visualizer import pripravi_pakete, sestavi_podatke
from data_logger.data_logger import DataLogger
from windower import window_signal_seconds
from stft import compute_spectrograms_1d
from alaw import alaw_decode_all


def load_session_mic(bin_file):

    logger = DataLogger()
    raw_packets = logger.parse_file(bin_file)

    packets = pripravi_pakete(raw_packets)

    fvz_mic, mic_raw = sestavi_podatke(packets, 4)

    mic_signal = alaw_decode_all(mic_raw)

    return fvz_mic, mic_signal
