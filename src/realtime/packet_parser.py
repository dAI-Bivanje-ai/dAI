import logging


from src.data_logger.data_logger import DataLogger


SYNC = b"\xff\xff"

ID_GYRO = 1
ID_ACC = 2
ID_MIC = 4


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

    def strip_debug_lines(self) -> None:
        i = 0
        cleaned = bytearray()
        while i < len(self.buffer):
            if self.buffer[i] == 0x50 and self.buffer[i : i + 6] == b"Packet":
                end = self.buffer.find(b"\r\n", i)

                if end != -1:
                    i = end + 2
                    continue
            cleaned.append(self.buffer[i])
            i += 1
        self.buffer = cleaned

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
        self.strip_debug_lines()
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
            packet = self.data_logger.parse_packet(raw_packet)
            # če paket ni veljaven ga preskočimo
            if packet is None:
                self.invalid_packets += 1
                continue

            self.valid_packets += 1
            packets.append(packet)

    def stats(self) -> dict[str, int]:
        """Vrne statistiko parserja za debugging."""
        return {
            "total_packets": self.total_packets,
            "valid_packets": self.valid_packets,
            "invalid_packets": self.invalid_packets,
            "buffer_size": len(self.buffer),
        }
