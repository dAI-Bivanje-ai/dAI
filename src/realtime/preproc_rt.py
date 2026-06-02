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
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
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

        # Model pričakuje PyTorch tensor oblike:
        # (batch, channels, freq, time)
        # permute - prestavimo osi
        acc_tensor = torch.from_numpy(acc_spec).float().permute(2, 0, 1).unsqueeze(0)
        gyro_tensor = torch.from_numpy(gyro_spec).float().permute(2, 0, 1).unsqueeze(0)

        return acc_tensor, gyro_tensor
    
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
    """
    Realtime preprocessing za mikrofon.

    Pomembno:
    realtime mora uporabljati iste nastavitve kot training.
    Če je bil model naučen na 5-sekundnih segmentih, potem mora tudi realtime
    uporabljati 5-sekundni segment.
    """

    def __init__(
        self,
        log_min: float,
        log_max: float,
        sample_rate: float,
        segment_seconds: float,
        stft_window_seconds: float,
        stft_overlap: float,
        rms_threshold: float,
    ) -> None:
        self.log_min = log_min
        self.log_max = log_max

        # Vzorčevalna frekvenca mikrofona, npr. 8000 Hz.
        self.sample_rate = sample_rate

        # Dolžina segmenta v sekundah, npr. 5 s.
        self.segment_seconds = segment_seconds

        # Dolžina enega STFT okna, npr. 0.032 s.
        self.stft_window_seconds = stft_window_seconds

        # Prekrivanje STFT oken, npr. 0.5 pomeni 50 %.
        self.stft_overlap = stft_overlap

        # Prag za tišino.
        self.rms_threshold = rms_threshold

        # Iz sekund izračunamo, koliko STFT frame-ov mora imeti en segment.
        self.segment_frames = self.calculate_segment_frames()

    def calculate_segment_frames(self) -> int:
        """
        Izračuna število STFT frame-ov za en mikrofonski segment.

        Primer:
        sample_rate = 8000 Hz
        segment_seconds = 5 s
        stft_window_seconds = 0.032 s

        window_samples = 0.032 * 8000 = 256
        hop_samples = 256 * (1 - 0.5) = 128
        segment_samples = 5 * 8000 = 40000

        frame_count = floor((40000 - 256) / 128) + 1 = 311
        """

        segment_samples = int(self.segment_seconds * self.sample_rate)
        window_samples = int(self.stft_window_seconds * self.sample_rate)
        hop_samples = int(window_samples * (1.0 - self.stft_overlap))

        if hop_samples <= 0:
            raise ValueError("STFT hop mora biti večji od 0.")

        if segment_samples < window_samples:
            raise ValueError("Mikrofonski segment je krajši od enega STFT okna.")

        return ((segment_samples - window_samples) // hop_samples) + 1

    def process(self, samples: np.ndarray) -> tuple[torch.Tensor | None, float]:

        raw = np.asarray(samples, dtype=np.int8)
        pcm = alaw_decode_all(raw)
        filtered = bandpass_mic(pcm, self.sample_rate)

        # Če je sig prenizek, predpostavimo, da gre za tišino in ne kličemo modela.
        rms = float(np.sqrt(np.mean(filtered**2)))
        if rms < self.rms_threshold:
            return None, rms

        # Signal razrežemo na kratka STFT okna.
        windows = window_signal_seconds(
            filtered,
            self.sample_rate,
            T_window=self.stft_window_seconds,
            prekrivanje=self.stft_overlap,
        )

        spectrograms = compute_spectrograms_1d(windows)

        # Če še nimamo dovolj frame-ov za cel segment, počakamo.
        if spectrograms.shape[0] < self.segment_frames:
            return None, rms

        # Modelu damo zadnji segment, ker nas v realtime zanima trenutno stanje.
        spec = spectrograms[-self.segment_frames :].T
        spec = np.log10(spec + 1e-10)
        spec = (spec - self.log_min) / (self.log_max - self.log_min)
        spec = np.clip(spec, 0.0, 1.0)

        tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor, rms
