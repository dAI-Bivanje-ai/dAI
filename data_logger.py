import struct

import numpy as np
import serial


class DataLogger:
    """Razred za branje in parsanje binarnih podatkov s plošče STM32."""

    DEFAULT_PORT: str = "/dev/tty.usbmodem207A356750471"
    DEFAULT_BAUDRATE: int = 115200

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baudrate: int = DEFAULT_BAUDRATE,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.ser = None

    def open(self) -> None:
        """Odpre serijski port za komunikacijo s ploščo."""
        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)

    def close(self) -> None:
        """Zapre serijski port, če je odprt."""
        if self.ser:
            self.ser.close()

    def read_raw(self, filename: str = "raw_data.bin") -> None:
        """
        Bere surove bajte s serijskega porta in jih shranjuje.

        Args:
            filename (str): Ime izhodne datoteke.
        """
        if not self.ser:
            raise RuntimeError("Port ni odprt")

        raw_data_file = open(filename, "wb")
        try:
            while True:
                data = self.ser.read(1024)
                if data:
                    print(data.hex(" "))
                    raw_data_file.write(data)
        except KeyboardInterrupt:
            pass
        finally:
            raw_data_file.close()
