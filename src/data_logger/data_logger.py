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

            if chunk_id == 0x04:
                samples = list(chunk_data)
            else:
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

    def parse_file(self, filename: str) -> list[dict]:
        """
        Prebere binarno datoteko in poišče pakete.

        Args:
            filename (str): Pot do .bin datoteke.

        Returns:
            list[int]: Seznam pozicij sync markerjev.
        """
        with open(filename, "rb") as f:
            data = f.read()

        cleaned = bytearray()
        i = 0
        while i < len(data):

            if data[i] == 0x50 and data[i : i + 6] == b"Packet":

                end = data.find(b"\r\n", i)

                if end != -1:
                    i = end + 2
                    continue

            cleaned.append(data[i])
            i += 1

        positions = self.find_sync_markers(bytes(cleaned))
        packets = []

        for j in range(len(positions)):
            if j < len(positions) - 1:
                raw_packet = bytes(cleaned[positions[j] : positions[j + 1]])
            else:
                raw_packet = bytes(cleaned[positions[j] :])

            result = self.parse_packet(raw_packet)
            if result:
                packets.append(result)
        """
        print(f"Veljavnih paketov: {len(packets)}/{len(positions)}")
        for p in packets[:3]:
            print(
                f"  ts={p['timestamp']}ms, "
                f"counter={p['packet_counter']}, "
                f"chunki={list(p['chunks'].keys())}"
            )
        """

        return packets

    def save_data(self, filename: str, packets: list[dict]) -> None:
        """
        Shrani parsirane pakete v .npz datoteko.

        Iz paketov zbere timestampe in vzorce za vsak
        senzor ter jih shrani kot numpy arraye.

        Args:
            filename (str): Ime izhodne .npz datoteke.
            packets (list[dict]): Seznam parsiranih paketov
                iz parse_file().
        """
        t_gyro = []
        y_gyro = []
        t_acc = []
        y_acc = []
        t_mag = []
        y_mag = []
        t_mic = []
        y_mic = []

        for packet in packets:
            ts = packet["timestamp"]

            if 1 in packet["chunks"]:
                for sample in packet["chunks"][1]:
                    t_gyro.append(ts)
                    y_gyro.append(sample)

            if 2 in packet["chunks"]:
                for sample in packet["chunks"][2]:
                    t_acc.append(ts)
                    y_acc.append(sample)

            if 3 in packet["chunks"]:
                for sample in packet["chunks"][3]:
                    t_mag.append(ts)
                    y_mag.append(sample)
            if 4 in packet["chunks"]:
                for i, sample in enumerate(packet["chunks"][4]):
                    t_mic.append(ts + i * (1000 / 8000))
                    y_mic.append(sample)

        np.savez(
            filename,
            t_gyro=np.array(t_gyro),
            y_gyro=np.array(y_gyro),
            t_acc=np.array(t_acc),
            y_acc=np.array(y_acc),
            t_mag=np.array(t_mag),
            y_mag=np.array(y_mag),
            t_mic=np.array(t_mic),
            y_mic=np.array(y_mic),
        )


if __name__ == "__main__":
    logger = DataLogger()
    logger.open()
    logger.read_raw("seja.bin")
    logger.close()
