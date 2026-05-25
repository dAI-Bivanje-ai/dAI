import logging
import time
from collections.abc import Iterator

import serial
import serial.tools.list_ports

# USB identifikator STM32 Virtual COM port-a
# lsusb
STM32_VID = "0483"
STM32_PID = "5740"
# hitrost
DEFAULT_BAUDRATE = 115200
# koliko B prebere naenkrat
READ_SIZE = 2048


class LiveSerialReader:
    """
    Realtime reader za STM32.

    Naloge readerja:
    - poišče STM32 napravo
    - vzpostavi serial povezavo
    - konfigurira stream
    - bere surove bajte
    - ob odklopu ne pade
    """

    def __init__(self, baudrate: int = DEFAULT_BAUDRATE) -> None:
        self.baudrate = baudrate
        self.port: str | None = None
        self.connection: serial.Serial | None = None
        self.running = False

    def find_stm32_port(self) -> str | None:
        """
        Poišče STM32 napravo med vsemi serial napravami.

        Vrne:
        - '/dev/ttyACM0'
        - '/dev/ttyACM1'
        - None če naprave ni
        """
        for port in serial.tools.list_ports.comports():
            hwid = port.hwid.upper()

            if f"VID:PID={STM32_VID}:{STM32_PID}" in hwid:
                return port.device

        return None

    def connect(self) -> bool:
        """
        Poskusi vzpostaviti serial povezavo.

        Vrne:
        True  -> uspešna povezava
        False -> povezava ni uspela
        """
        # poisce port
        detected_port = self.find_stm32_port()

        if detected_port is None:
            logging.info("STM32 ni zaznan.")
            return False

        try:
            # ustvari serial povezavo
            self.connection = serial.Serial(
                detected_port,
                self.baudrate,
                timeout=1,
            )
            # shrani port
            self.port = detected_port
            logging.info("STM32 povezan na %s", self.port)
            return True

        except serial.SerialException as exc:
            logging.error("Napaka pri povezavi z STM32: %s", exc)
            self.connection = None
            self.port = None
            return False

    def is_connected(self) -> bool:
        """Preveri, ali je povezava odprta."""
        return self.connection is not None and self.connection.is_open

    def disconnect(self) -> None:
        """Zapre serijsko povezavo."""
        if self.connection and self.connection.is_open:
            self.connection.close()

        self.connection = None
        self.port = None
        logging.info("STM32 povezava zaprta.")

    def start_stream(self) -> None:
        """Konfigurira STM32 in zažene stream podatkov."""
        if not self.is_connected():
            raise RuntimeError("STM32 ni povezan.")

        assert self.connection is not None

        # ustavimo morebitni prejšnji stream
        self.connection.write(b"OFF\r\n")
        time.sleep(0.5)

        # vklopimo giroskop
        self.connection.write(b"CONFIG gyro 1\r\n")
        time.sleep(0.1)

        # vklopimo pospeškometer
        self.connection.write(b"CONFIG accel 1\r\n")
        time.sleep(0.1)

        # Magnetometra ne rabimo.
        self.connection.write(b"CONFIG mag 0\r\n")
        time.sleep(0.1)

        # Mikrofon za zdaj izklopljen, ker najprej stabiliziramo IMU stream.
        self.connection.write(b"CONFIG mic 0\r\n")
        time.sleep(0.1)

        # počistimo stare podatke v bufferju
        self.connection.reset_input_buffer()

        # zaženemo stream
        self.connection.write(b"STREAM\r\n")
        time.sleep(0.5)

        # ponovno počistimo buffer
        self.connection.reset_input_buffer()
        logging.info("STM32 stream zagnan.")

    def stop_stream(self) -> None:
        """Ustavi stream."""
        if self.is_connected():
            assert self.connection is not None
            self.connection.write(b"OFF\r\n")
            logging.info("STM32 stream ustavljen.")

    def read_chunk(self, size: int = READ_SIZE) -> bytes:
        """
        Prebere blok surovih bajtov.

        Primer:
            ff ff 01 3a 22 ...
        """
        if not self.is_connected():
            raise RuntimeError("STM32 ni povezan.")

        assert self.connection is not None
        return self.connection.read(size)

    def read_stream(self) -> Iterator[bytes]:
        """
        Neprekinjeno bere surove bajte.

        Če se STM32 odklopi, program ne pade, ampak čaka na ponovni priklop.
        """
        self.running = True
        # glavna realtime zanka
        while self.running:
            # če STM ni povezan
            if not self.is_connected():
                # poskus povezave
                if not self.connect():
                    time.sleep(1)
                    continue

                try:
                    # po povezavi zazene stream
                    self.start_stream()
                except (RuntimeError, serial.SerialException) as exc:
                    logging.error("Napaka pri zagonu streama: %s", exc)
                    self.disconnect()
                    time.sleep(1)
                    continue

            try:
                # prebere blok bajtov
                data = self.read_chunk()
                if data:
                    # če smo dobili dejanske pod. vrne raw bajte
                    yield data

            except serial.SerialException as exc:
                logging.error("Napaka pri branju ali odklop STM32: %s", exc)
                self.disconnect()
                time.sleep(1)

    def stop(self) -> None:
        """Ustavi reader."""
        self.running = False
        self.stop_stream()
        self.disconnect()


def main() -> None:
    # konfiguracija loggerja
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    # ustvari reader
    reader = LiveSerialReader()

    try:
        # realtime branje streama
        for chunk in reader.read_stream():
            # izpis velikosti prejetega bloka
            print(f"Prejeto bajtov: {len(chunk)}")

    except KeyboardInterrupt:
        print("Ustavljanje realtime readerja...")

    finally:
        reader.stop()


if __name__ == "__main__":
    main()
