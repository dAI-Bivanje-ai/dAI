"""
Drseči buffer za realtime IMU podatke (ACC in GYRO).

Modul vsebuje SignalBuffer, ki iz paketov pobere ACC in GYRO vzorce, jih
pretvori v fizikalne enote in hrani zadnjih N vzorcev. Ko je dovolj poln,
vrne okni oblike (N, 3) za nadaljnjo obdelavo.
"""

from collections import deque

import numpy as np

ID_GYRO = 1
ID_ACC = 2


class SignalBuffer:
    """
    Buffer za realtime IMU podatke.

    - sprejema pakete iz LivePacketParser
    - iz njih pobere samo GYRO in ACC
    - vzorce pretvori v fizikalne enote
    - hrani zadnjih N vzorcev
    - vrne okno oblike (N, 3), ko je buffer poln
    """

    def __init__(
        self,
        window_seconds: float,
        acc_sample_rate: float,
        gyro_sample_rate: float,
        acc_resolution: float,
        gyro_resolution: float,
        ready_ratio: float = 0.95,
    ) -> None:
        """
        Args:
            window_seconds:
                Dolžina okna v sekundah.
            acc_sample_rate:
                Pričakovana vzorčevalna frekvenca pospeškometra v Hz.
            gyro_sample_rate:
                Pričakovana vzorčevalna frekvenca giroskopa v Hz.
            acc_resolution:
                Pretvorba surovih ACC vrednosti v g.
            gyro_resolution:
                Pretvorba surovih GYRO vrednosti v deg/s.
            ready_ratio:
                Delež pričakovanih vzorcev, pri katerem buffer že štejemo kot pripravljen.
        """
        self.window_seconds = window_seconds
        self.acc_sample_rate = acc_sample_rate
        self.gyro_sample_rate = gyro_sample_rate
        self.acc_resolution = acc_resolution
        self.gyro_resolution = gyro_resolution
        self.ready_ratio = ready_ratio

        self.acc_max_samples = int(window_seconds * acc_sample_rate)
        self.gyro_max_samples = int(window_seconds * gyro_sample_rate)

        # deque z maxlen avtomatsko briše najstarejše vzorce,
        # ko dodamo nove in je buffer že poln.
        self.acc_buffer = deque(maxlen=self.acc_max_samples)
        self.gyro_buffer = deque(maxlen=self.gyro_max_samples)

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
                    x * self.acc_resolution,
                    y * self.acc_resolution,
                    z * self.acc_resolution,
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
                    x * self.gyro_resolution,
                    y * self.gyro_resolution,
                    z * self.gyro_resolution,
                )
            )

    def is_ready(self) -> bool:
        """
        Preveri, ali buffer vsebuje dovolj vzorcev za eno okno.

        Returns:
            bool — True, ko ACC in GYRO dosežeta zahtevani delež vzorcev
        """
        acc_required = int(self.acc_max_samples * self.ready_ratio)
        gyro_required = int(self.gyro_max_samples * self.ready_ratio)

        return (
            len(self.acc_buffer) >= acc_required
            and len(self.gyro_buffer) >= gyro_required
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
            "required_acc_samples": self.acc_max_samples,
            "required_gyro_samples": self.gyro_max_samples,
        }
