import logging
import struct

import numpy as np

from src.data_logger.data_logger import DataLogger


SYNC = b"\xff\xff"

ID_GYRO = 1
ID_ACC = 2
ID_MIC = 4

SAMPLE_SIZE = 6  # int16 x, y, z


class LivePacketParser:
    """
    Realtime parser za STM32 stream.

    SerialReader vrača samo surove bajte:
        b"..."

    Ta parser iz teh bajtov naredi pakete:
        {
            "packet_counter": int,
            "timestamp": int,
            "chunks": {
                1: [(x, y, z), ...],  # gyro
                2: [(x, y, z), ...],  # acc
                4: [sample, ...],     # mic
            }
        }

    Glavna razlika od DataLoggerja:
    - DataLogger bere cel .bin file naenkrat
    - LivePacketParser dobiva bajte po kosih iz serial porta zato mora imeti svoj buffer
    """

    def __init__(self, max_buffer_size: int = 200000) -> None:
        # hranjenje bajtov iz serial porta ker en read() nii nujno cel paket
        self.buffer = bytearray()
        # buffer ne sme rasti v neskoncnost
        self.max_buffer_size = max_buffer_size

        self.data_logger = DataLogger()

        self.total_packets = 0
        self.valid_packets = 0
        self.invalid_packets = 0

    def feed(self, data: bytes) -> list[dict]:
        """
        Glavna realtime funkcija.

        Vhod:
            en kos bajtov iz serial readerja

        Izhod:
            seznam vseh kompletnih paketov, ki jih je parser uspel sestaviti


        En serial read() lahko vsebuje:
            - 0 celih paketov
            - 1 cel paket
            - več celih paketov
        """
        if not data:
            return []
        # dodamo nove bajte k starim v bufferju
        self.buffer.extend(data)
        # ce buffer postane prevelik, pomeni da dolgo nismo našli sync markerja
        # počistimo da ne zmanjka RAM-a
        if len(self.buffer) > self.max_buffer_size:
            logging.warning("Parser buffer je prevelik. Čistim buffer.")
            self.buffer.clear()
            return []

        packets = []

        while True:
            # poišče zacetek paketa
            start = self.buffer.find(SYNC)

            # če sync markerja ni, trenutni bajti niso uporabni.
            if start == -1:
                self.buffer.clear()
                return packets

            # Če je pred sync markerjem garbage, ga odstranimo.
            if start > 0:
                del self.buffer[:start]

            # poišče naslednji sync marker
            next_start = self.buffer.find(SYNC, len(SYNC))

            # če naslednjega SM še ni, trenutni paket še ni cel
            # počakamo naslednji feed()
            if next_start == -1:
                return packets
            # vse med prvim in drugim SM je en raw paket
            raw_packet = bytes(self.buffer[:next_start])
            # paket odstranimo iz bufferja
            del self.buffer[:next_start]

            self.total_packets += 1
            # pretvorba raw paketa v dict
            packet = self.parse_packet(raw_packet)
            # če paket ni veljaven ga preskočimo
            if packet is None:
                self.invalid_packets += 1
                continue

            self.valid_packets += 1
            packets.append(packet)

    def parse_packet(self, raw_packet: bytes) -> dict | None:
        """
        Parsira en cel raw paket.

        1. preverjanje SYNC markerja
        2. unstuff payload-a
        3. branje timestampa in velikosti
        4. CRC preverjanje
        5. parsing chunkov
        """
        # min. dolzina: sync + counter + ts + size + crc
        if len(raw_packet) < 12:
            return None
        # paket se mora zaceti z FF FF
        if raw_packet[:2] != SYNC:
            return None
        # 3ji B je packet counter
        packet_counter = raw_packet[2]

        # payload je vse po FF FF + packet counter
        payload = self.data_logger.unstuff(raw_packet[3:])

        # mora vsebovati vsaj ts 4B, packet_size 2B, crc 2B
        if len(payload) < 8:
            return None

        try:
            # prvi 4 bajti ts v ms
            timestamp = struct.unpack("<I", payload[0:4])[0]
            # velikost paketa 2B
            packet_size = struct.unpack("<H", payload[4:6])[0] + 1
            # zadnja 2B crc
            received_crc = struct.unpack("<H", payload[-2:])[0]

        except struct.error:
            return None
        # izracuna crc cez payload brez zadnjih 2B
        computed_crc = self.data_logger.crc16_compute(payload[:-2])

        # če se CRC ne ujema je paket poskodovan
        if received_crc != computed_crc:
            return None

        if packet_size != len(payload):
            logging.debug(
                "Packet size mismatch: header=%s actual=%s",
                packet_size,
                len(payload),
            )

        chunks = self.parse_chunks(payload[6:-2])

        if chunks is None:
            return None

        return {
            "packet_counter": packet_counter,
            "timestamp": timestamp,
            "chunks": chunks,
        }

    def parse_chunks(self, chunks_data: bytes) -> dict[int, list] | None:
        """
        Parsira vse chunke iz paketa.

        Chunk header:
            chunk_id    1 B
            chunk_size  2 B   zapisano kot size - 1
            reserved    1 B
            data        N B

        Podprto:
            1 -> gyro: int16 x, y, z
            2 -> acc:  int16 x, y, z
            4 -> mic:  int8 samples

        Magnetometer ignoriramo, ker ga v realtime projektu ne uporabljata več.
        """
        chunks: dict[int, list] = {}
        pos = 0

        while pos < len(chunks_data):
            # vsak chunk mora imet vsaj 4B header
            if pos + 4 > len(chunks_data):
                return None

            chunk_id = chunks_data[pos]

            try:
                chunk_size = (
                    struct.unpack(
                        "<H",
                        chunks_data[pos + 1 : pos + 3],
                    )[0]
                    + 1
                )

            except struct.error:
                return None

            data_start = pos + 4
            data_end = data_start + chunk_size

            if data_end > len(chunks_data):
                return None

            chunk_data = chunks_data[data_start:data_end]

            # Mikrofon je int8 mono stream.
            if chunk_id == ID_MIC:
                samples = np.frombuffer(chunk_data, dtype=np.int8).tolist()

            # Gyro in acc sta: int16 x, y, z.
            elif chunk_id in (ID_GYRO, ID_ACC):
                if chunk_size % SAMPLE_SIZE != 0:
                    return None

                samples = []

                for i in range(0, chunk_size, SAMPLE_SIZE):
                    try:
                        x, y, z = struct.unpack(
                            "<hhh",
                            chunk_data[i : i + SAMPLE_SIZE],
                        )
                    except struct.error:
                        return None

                    samples.append((x, y, z))

            else:
                # Nepodprt chunk ignoriramo.
                samples = []
            # Shranimo samo chunke, ki imajo podatke.
            if samples:
                chunks[chunk_id] = samples
            # Premaknemo se na naslednji chunk.
            pos = data_end

        return chunks

    def stats(self) -> dict[str, int]:
        """Vrne statistiko parserja za debugging."""
        return {
            "total_packets": self.total_packets,
            "valid_packets": self.valid_packets,
            "invalid_packets": self.invalid_packets,
            "buffer_size": len(self.buffer),
        }
