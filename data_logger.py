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

    def find_sync_markers(self, data: bytes) -> list[int]:
        """
        Poišče pozicije sync markerjev (0xFF 0xFF) v podatkih.

        Args:
            data (bytes): Surovi bajti iz datoteke.

        Returns:
            list[int]: Seznam pozicij sync markerjev.
        """
        positions = []
        for i in range(len(data) - 1):
            if data[i] == 0xFF and data[i + 1] == 0xFF:
                positions.append(i)
        return positions

    def unstuff(self, stuffed_data: bytes) -> bytes:
        """
        Odstrani byte stuffing iz payloada paketa.

        Bajt 0xFE označuje escape sekvenco — naslednji bajt
        XOR-amo z 0xFE, da dobimo originalno vrednost.

        Args:
            stuffed_data (bytes): Stuffani bajti payloada.

        Returns:
            bytes: Originalni (unstuffani) bajti.
        """
        i = 0
        result = bytearray()
        while i < len(stuffed_data):
            if stuffed_data[i] == 0xFE:
                next_byte = stuffed_data[i + 1]
                original_byte = next_byte ^ 0xFE
                result.append(original_byte)
                i += 2
            else:
                result.append(stuffed_data[i])
                i += 1
        return bytes(result)

    def crc16_compute(self, data: bytes) -> int:
        """
        Izračuna CRC16-ANSI

        Args:
            data (bytes): Bajti za izračun CRC.

        Returns:
            int: Izračunana CRC16 vrednost.
        """
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc = crc >> 1
        return crc

    def parse_packet(self, raw_packet: bytes) -> dict | None:
        """
        Parsira en surov paket in vrne strukturirane podatke.

        Preveri sync marker, unstuffa payload, preveri CRC
        in parsira vse chunke s senzorskimi podatki.

        Args:
            raw_packet (bytes): Surovi bajti enega paketa
                (od sync markerja do naslednjega).

        Returns:
            dict | None: Slovar s podatki paketa ali None
                če je paket neveljaven.
        """
        if raw_packet[0:2] != b"\xff\xff":
            return None

        packet_counter = raw_packet[2]

        payload = self.unstuff(raw_packet[3:])

        timestamp = struct.unpack("<I", payload[0:4])[0]
        packet_size = struct.unpack("<H", payload[4:6])[0] + 1

        received_crc = struct.unpack("<H", payload[-2:])[0]
        computed_crc = self.crc16_compute(payload[:-2])
        if received_crc != computed_crc:
            return None

        chunks_data = payload[6:-2]
        pos = 0
        chunks = {}

        while pos < len(chunks_data):
            chunk_id = chunks_data[pos]
            chunk_size = struct.unpack("<H", chunks_data[pos + 1 : pos + 3])[0] + 1
            chunk_data = chunks_data[pos + 4 : pos + 4 + chunk_size]

            samples = []
            for i in range(0, chunk_size, 6):
                x, y, z = struct.unpack("<hhh", chunk_data[i : i + 6])
                samples.append((x, y, z))

            chunks[chunk_id] = samples
            pos += 4 + chunk_size

        return {
            "packet_counter": packet_counter,
            "timestamp": timestamp,
            "chunks": chunks,
        }
