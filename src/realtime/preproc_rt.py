import numpy as np
import torch

from src.preprocessing.alaw import alaw_decode_all
from src.preprocessing.dataset_builder import SEGMENT_LENGTH
from src.preprocessing.filters import bandpass_mic
from src.preprocessing.stft import (
    compute_spectrograms,
    compute_spectrograms_1d,
    group_spectrograms,
    normalize_spectrogram,
)
from src.preprocessing.windower import window_signal, window_signal_seconds


class RealtimePreprocessor:
    """
    Realtime priprava ACC in GYRO signalov v spektrograme.

    Vhod:
        acc_window  -> zadnje okno pospeškometra, oblika (N, 3)
        gyro_window -> zadnje okno giroskopa, oblika (N, 3)

    Izhod:
        acc_spec  -> zadnji ACC spektrogramski segment
        gyro_spec -> zadnji GYRO spektrogramski segment
    """

    def __init__(self) -> None:
        # Enake velikosti FFT oken kot pri offline pripravi dataseta.
        self.acc_window_size = 49
        self.gyro_window_size = 211

        # Koliko zaporednih spektrogramov sestavlja en 2D segment.
        self.segment_length = SEGMENT_LENGTH

    def process(
        self,
        acc_window: np.ndarray,
        gyro_window: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Pripravi spektrograma za zadnje ACC in GYRO okno.

        Če še ni dovolj podatkov za cel segment, vrne None.
        """

        acc_spec = self.build_sensor_spectrogram(
            signal=acc_window,
            window_size=self.acc_window_size,
        )

        gyro_spec = self.build_sensor_spectrogram(
            signal=gyro_window,
            window_size=self.gyro_window_size,
        )

        if acc_spec is None or gyro_spec is None:
            return None

        # Uporabimo obstoječo normalizacijo iz stft.py
        acc_spec = normalize_spectrogram(acc_spec)
        gyro_spec = normalize_spectrogram(gyro_spec)

        return acc_spec, gyro_spec

    def build_sensor_spectrogram(
        self,
        signal: np.ndarray,
        window_size: int,
    ) -> np.ndarray | None:
        """
        Iz enega senzorskega signala naredi zadnji 2D spektrogramski segment.
        """

        signal = self.prepare_signal(signal)

        # Če signal še ni dovolj dolg za eno FFT okno, počakamo
        if len(signal) < window_size:
            return None

        # 50 % prekrivanje med okni
        step = round(window_size * 0.5)

        # Signal razrežemo na manjša okna oblike (window_size, 3)
        windows = window_signal(
            signal,
            window_size=window_size,
            step=step,
        )

        if windows.size == 0:
            return None

        # Za vsako okno izračunamo spektrogram
        spectrograms = compute_spectrograms(windows)

        if spectrograms.size == 0:
            return None

        # Zaporedne spektrograme združimo v 2D segmente
        grouped = group_spectrograms(
            spectrograms,
            segment_length=self.segment_length,
        )

        if grouped.size == 0:
            return None

        # V realtime prikazu uporabljamo samo zadnji segment
        return grouped[-1]

    def prepare_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Preveri obliko signala in ga pretvori v float32.
        """

        signal = np.asarray(signal, dtype=np.float32)

        if signal.ndim != 2:
            raise ValueError("Signal mora biti 2D tabela.")

        if signal.shape[1] != 3:
            raise ValueError("Signal mora imeti obliko (N, 3).")

        return signal


class MicRealtimePreprocessor:

    def __init__(self, log_min: float, log_max: float) -> None:
        self.log_min = log_min
        self.log_max = log_max
        self.FVZ = 8000.0
        self.SEG_FRAMES = 311
        self.RMS_THRESHOLD = 0.01

    def process(self, samples: np.ndarray) -> tuple[torch.Tensor | None, float]:

        raw = np.asarray(samples, dtype=np.int8)
        pcm = alaw_decode_all(raw)
        filtered = bandpass_mic(pcm, self.FVZ)

        rms = float(np.sqrt(np.mean(filtered**2)))
        if rms < self.RMS_THRESHOLD:
            return None, rms

        windows = window_signal_seconds(
            filtered, self.FVZ, T_window=0.032, prekrivanje=0.5
        )

        spectrograms = compute_spectrograms_1d(windows)
        if spectrograms.shape[0] < self.SEG_FRAMES:
            return None, rms

        spec = spectrograms[-self.SEG_FRAMES :].T
        spec = np.log10(spec + 1e-10)
        spec = (spec - self.log_min) / (self.log_max - self.log_min)
        spec = np.clip(spec, 0.0, 1.0)

        tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor, rms
