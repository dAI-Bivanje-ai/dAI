import numpy as np
import torch

from src.preprocessing.windower import window_signal
from src.preprocessing.stft import compute_spectrograms, group_spectrograms
from src.preprocessing.dataset_builder import SEGMENT_LENGTH


class RealtimePreprocessor:
    """
    Realtime preprocessing za IMU podatke.

    Ta razred NE bere serial porta.
    Ta razred NE parsira paketov.
    Ta razred NE dela bufferja.

    Vhod dobi iz signal_buffer.py:
        acc_window  -> oblika (N, 3)
        gyro_window -> oblika (N, 3)

    Izhod pripravi za CNNModel:
        acc_tensor  -> oblika (1, 3, freq_bins, segment_length)
        gyro_tensor -> oblika (1, 3, freq_bins, segment_length)
    """

    def __init__(self) -> None:
        """
        Uporabimo iste nastavitve kot v dataset_builder.py,
        ker mora biti realtime preprocessing enak kot training preprocessing.
        """

        # Dolžina enega FFT okna za pospeškometer.
        # To je vzeto iz obstoječega dataset_builder.py.
        self.acc_window_size = 49

        # Dolžina enega FFT okna za giroskop.
        # To je vzeto iz obstoječega dataset_builder.py.
        self.gyro_window_size = 211

        # Koliko zaporednih FFT spektrogramov združimo v en 2D spektrogram.
        self.segment_length = SEGMENT_LENGTH

    def _build_sensor_spectrogram(
        self,
        signal_window: np.ndarray,
        fft_window_size: int,
    ) -> np.ndarray | None:
        """
        Iz enega realtime signala naredi en CNN spektrogram.

        Koraki so isti kot v offline dataset_builder.py:

        1. signal razrežemo na manjša okna
        2. za vsako okno izračunamo FFT spektrogram
        3. več zaporednih spektrogramov združimo v en 2D spektrogram
        4. vzamemo zadnji segment, ker v realtime gledamo zadnje stanje
        """

        # Korak med okni je 50 % okna.
        # Primer:
        #   window_size = 49
        #   step = 24
        step = round(fft_window_size * 0.5)

        # Iz signala oblike (N, 3) dobimo:
        #   windows -> (M, fft_window_size, 3)
        windows = window_signal(
            signal_window,
            window_size=fft_window_size,
            step=step,
        )

        # Če je signal prekratek, ni mogoče narediti spektrograma.
        if windows.size == 0:
            return None

        # Iz vsakega okna naredimo spektrogram.
        # Izhod:
        #   (M, freq_bins, 3)
        spectrograms = compute_spectrograms(windows)

        # Združimo več zaporednih spektrogramov v 2D obliko za CNN.
        # Izhod:
        #   (N, freq_bins, segment_length, 3)
        grouped = group_spectrograms(
            spectrograms,
            segment_length=self.segment_length,
        )

        # Če nimamo dovolj spektrogramov za en cel segment, še ne moremo napovedati.
        if grouped.size == 0:
            return None

        # V realtime nas zanima samo zadnji segment.
        # Oblika:
        #   (freq_bins, segment_length, 3)
        return grouped[-1]

    def _normalize(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Log normalizacija kot v dataset_builder.py.

        To naredimo zato, ker imajo FFT magnitude lahko zelo velike razlike.
        CNN lažje dela z vrednostmi približno med 0 in 1.
        """

        spectrogram = np.log10(spectrogram + 1e-10)

        min_value = spectrogram.min()
        max_value = spectrogram.max()

        if max_value == min_value:
            return np.zeros_like(spectrogram, dtype=np.float32)

        spectrogram = (spectrogram - min_value) / (max_value - min_value)

        return spectrogram.astype(np.float32)

    def _to_model_tensor(self, spectrogram: np.ndarray) -> torch.Tensor:
        """
        Pretvori spektrogram v obliko, ki jo pričakuje CNNModel.

        Pred tem:
            (freq_bins, segment_length, 3)

        Po transpose:
            (3, freq_bins, segment_length)

        Po unsqueeze:
            (1, 3, freq_bins, segment_length)

        1 pomeni batch size.
        3 pomeni osi X, Y, Z.
        """

        spectrogram = np.transpose(spectrogram, (2, 0, 1))

        tensor = torch.tensor(
            spectrogram,
            dtype=torch.float32,
        )

        return tensor.unsqueeze(0)

    def process(
        self,
        acc_window: np.ndarray,
        gyro_window: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Glavna funkcija.

        Vhod:
            acc_window  -> (N, 3)
            gyro_window -> (N, 3)

        Izhod:
            acc_tensor, gyro_tensor

        Če še ni dovolj podatkov, vrne None.
        """

        acc_spec = self._build_sensor_spectrogram(
            signal_window=acc_window,
            fft_window_size=self.acc_window_size,
        )

        gyro_spec = self._build_sensor_spectrogram(
            signal_window=gyro_window,
            fft_window_size=self.gyro_window_size,
        )

        if acc_spec is None or gyro_spec is None:
            return None

        acc_spec = self._normalize(acc_spec)
        gyro_spec = self._normalize(gyro_spec)

        acc_tensor = self._to_model_tensor(acc_spec)
        gyro_tensor = self._to_model_tensor(gyro_spec)

        return acc_tensor, gyro_tensor
