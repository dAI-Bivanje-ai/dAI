from collections import deque

import numpy as np


ID_GYRO = 1
ID_ACC = 2

# Pretvorba iz int16 vrednosti v fizikalne enote.
# To mora biti enako kot v offline vizualizaciji.
GYRO_RESOLUTION = 8.75e-3  # deg/s
ACC_RESOLUTION = 1e-3      # g


class SignalBuffer:
    """
    Buffer za realtime IMU podatke.

    - sprejema pakete iz LivePacketParser
    - iz njih pobere samo GYRO in ACC
    - vzorce pretvori v fizikalne enote
    - hrani zadnjih N vzorcev
    - vrne okno oblike (N, 3), ko je buffer poln
    """

    def __init__(self, window_seconds: float, sample_rate: float) -> None:
        """
        Args:
            window_seconds:
                Dolžina okna v sekundah.
            sample_rate:
                Vzorčevalna frekvenca senzorja v Hz.
        """
        self.window_seconds = window_seconds
        self.sample_rate = sample_rate

        # Koliko vzorcev potrebujemo za eno okno.
        # 2 s * 100 Hz = 200 vzorcev.
        self.max_samples = int(window_seconds * sample_rate)

        # deque z maxlen avtomatsko briše najstarejše vzorce,
        # ko dodamo nove in je buffer že poln.
        self.acc_buffer: deque[tuple[float, float, float]] = deque(
            maxlen=self.max_samples
        )
        self.gyro_buffer: deque[tuple[float, float, float]] = deque(
            maxlen=self.max_samples
        )

    def add_packet(self, packet: dict) -> None:
        """
        Doda en parsiran realtime paket v buffer.

        Args:
            packet:
                Paket, ki ga vrne LivePacketParser.
        """
        chunks = packet.get("chunks", {})

        # ID 2 = accelerometer.
        if ID_ACC in chunks:
            self.add_acc_samples(chunks[ID_ACC])

        # ID 1 = gyroscope.
        if ID_GYRO in chunks:
            self.add_gyro_samples(chunks[ID_GYRO])

    def add_acc_samples(self, samples: list[tuple[int, int, int]]) -> None:
        """
        Doda ACC vzorce v buffer.

        Vhod iz parserja je int16:
            (x, y, z)

        Pretvorimo ga v g:
            raw * 0.001
        """
        for x, y, z in samples:
            self.acc_buffer.append(
                (
                    x * ACC_RESOLUTION,
                    y * ACC_RESOLUTION,
                    z * ACC_RESOLUTION,
                )
            )

    def add_gyro_samples(self, samples: list[tuple[int, int, int]]) -> None:
        """
        Doda GYRO vzorce v buffer.

        Vhod iz parserja je int16:
            (x, y, z)

        Pretvorimo ga v deg/s:
            raw * 8.75e-3
        """
        for x, y, z in samples:
            self.gyro_buffer.append(
                (
                    x * GYRO_RESOLUTION,
                    y * GYRO_RESOLUTION,
                    z * GYRO_RESOLUTION,
                )
            )

    def is_ready(self) -> bool:
        """
        Vrne True, ko imata ACC in GYRO dovolj vzorcev za eno polno okno.
        """
        return (
            len(self.acc_buffer) == self.max_samples
            and len(self.gyro_buffer) == self.max_samples
        )

    def get_window(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Vrne trenutno okno za nadaljnjo obdelavo.

        Returns:
            acc_window:
                numpy array oblike (N, 3)

            gyro_window:
                numpy array oblike (N, 3)

        Če buffer še ni poln, vrne:
            None, None
        """
        if not self.is_ready():
            return None, None

        acc_window = np.array(self.acc_buffer, dtype=np.float32)
        gyro_window = np.array(self.gyro_buffer, dtype=np.float32)

        return acc_window, gyro_window

    def clear(self) -> None:
        """
        Počisti buffer.

        To pride prav ob reconnectu STM32 ali če želiš začeti novo meritev.
        """
        self.acc_buffer.clear()
        self.gyro_buffer.clear()

    def status(self) -> dict[str, int]:
        """
        Vrne stanje bufferja za debug izpis.

        Primer:
            {'acc_samples': 120, 'gyro_samples': 120, 'required_samples': 200}
        """
        return {
            "acc_samples": len(self.acc_buffer),
            "gyro_samples": len(self.gyro_buffer),
            "required_samples": self.max_samples,
        }