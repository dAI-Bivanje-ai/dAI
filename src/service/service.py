import socket
import threading
import serial.tools.list_ports
import time
import serial
import logging
from src.data_logger.data_logger import DataLogger
from pathlib import Path

import re

# Datoteke shranimo v trenutno delovno pot storitve (WorkingDirectory v systemd).
WORK_DIR = Path.cwd()
# shrani v datoteko znotraj working direcotrya
DATA_DIR = WORK_DIR / "stm32_data"
# mapa se ustvari samodejno
DATA_DIR.mkdir(parents=True, exist_ok=True)


HOST = "127.0.0.1"
PORT = 5000
STM32_VID = "0483"
STM32_PID = "5740"
INTERVAL = 1
stm32_port: str | None = None
stm32_lock = threading.Lock()  # ščiti spremenljivko stm32_port

# sčiti dejansko serijsko komunikacijo — samo ena operacija na STM32 naenkrat.
stm32_io_lock = threading.Lock()

connected_clients: list = []
clients_lock = threading.Lock()


def handle_client(conn, addr):

    with conn:
        with stm32_lock:
            port = stm32_port
        if port is None:
            conn.sendall(b"Connected to SPO STM32 service - No STM32 detected\n")
        else:
            conn.sendall(b"Connected to SPO STM32 service\n")

        with clients_lock:
            connected_clients.append(conn)

        try:
            while True:

                data = conn.recv(1024)

                if not data:
                    break

                command = data.decode(errors="ignore").strip()

                if command == "STATUS":
                    with stm32_lock:
                        port = stm32_port
                    if port:
                        response = "STM32 is connected\n"
                    else:
                        response = "FAIL: STM32 is not connected\n"

                elif command == "GET_LAST":
                    with stm32_lock:
                        port = stm32_port
                    if port is None:
                        response = "FAIL: STM32 is not connected\n"
                    else:
                        try:
                            with stm32_io_lock:
                                get_files_from_stm32(port, which="last")
                            response = "Last file from STM32 has been processed\n"
                        except Exception as e:
                            logging.exception("GET_LAST neuspešen")
                            response = f"FAIL: {e}\n"

                elif command == "GET_ALL":
                    with stm32_lock:
                        port = stm32_port
                    if port is None:
                        response = "FAIL: STM32 is not connected\n"
                    else:
                        try:
                            with stm32_io_lock:
                                get_files_from_stm32(port, which="all")
                            response = "All files from STM32 are processed\n"
                        except Exception as e:
                            logging.exception("GET_ALL neuspešen")
                            response = f"FAIL: {e}\n"

                elif command.startswith("GET_FILE|"):
                    with stm32_lock:
                        port = stm32_port
                    if port is None:
                        response = "FAIL: STM32 is not connected\n"
                    else:
                        filename = command.split("|", 1)[1]
                        try:
                            with stm32_io_lock:
                                get_files_from_stm32(
                                    port, which="file", filename=filename
                                )
                            response = (
                                f"File {filename} from STM32 has been processed\n"
                            )
                        except Exception as e:
                            logging.exception("GET_FILE neuspešen")
                            response = f"FAIL: {e}\n"

                elif command == "DELETE":
                    with stm32_lock:
                        port = stm32_port
                    if port is None:
                        response = "FAIL: STM32 is not connected\n"
                    else:
                        try:
                            with stm32_io_lock:
                                stm32_delete(port)
                            response = "All files on STM32 are deleted\n"
                        except Exception as e:
                            logging.exception("DELETE neuspešen")
                            response = f"FAIL: {e}\n"

                else:
                    response = f"UNKNOWN: {command}\n"

                conn.sendall(response.encode())
        except OSError:
            pass
        finally:
            with clients_lock:
                # broadcast() je mogoce conn že odstranil kot mrtvega
                if conn in connected_clients:
                    connected_clients.remove(conn)


# skenira USB porte, najde stm32
def find_stm32_port():
    for port in serial.tools.list_ports.comports():
        if f"VID:PID={STM32_VID}:{STM32_PID}" in port.hwid.upper():
            return port.device

    return None


# konstantno preverjanje
def stm32_monitor():
    global stm32_port
    while True:
        detected = find_stm32_port()
        with stm32_lock:
            # shranimo originalni port
            prev = stm32_port
            stm32_port = detected
        if prev is None and detected:
            broadcast(f"STM32 detected at {detected}\n")
        elif prev and detected is None:
            broadcast("STM32 has disconnected\n")

        time.sleep(INTERVAL)


# broadcasta vsem, tisti ki so odklopljeni jih odstrani
def broadcast(message: str):

    with clients_lock:
        dead_clients = []
        for conn in connected_clients:
            try:
                conn.sendall(message.encode())
            except OSError:
                dead_clients.append(conn)

        for conn in dead_clients:
            connected_clients.remove(conn)


def stm32_open(port: str) -> DataLogger:
    data_logger = DataLogger(port=port)
    data_logger.open()
    time.sleep(0.5)

    # Pobrišemo vse stare podatke, ki so že čakali v bufferju.
    data_logger.ser.reset_input_buffer()

    # Damo firmware-u še malo časa, da je port stabilen pred LIST/GET/DELETE.
    time.sleep(0.5)
    return data_logger


def stm32_close(logger: DataLogger) -> None:
    # logger.ser.write(b"LOG\r\n")  # preklopimo v LOG mode - zacne streamanje!
    #time.sleep(0.2)
    logger.close()


def read_until_idle(logger: DataLogger, idle_timeout: float = 2.0) -> bytes:

    prev_timeout = logger.ser.timeout
    logger.ser.timeout = idle_timeout
    chunks = bytearray()
    try:
        while True:
            chunk = logger.ser.read(4096)
            if not chunk:  # idle_timeout sekund tišine = konec
                break
            chunks.extend(chunk)
    finally:
        logger.ser.timeout = prev_timeout
    return bytes(chunks)


def stm32_list_files(logger: DataLogger) -> list[str]:
    """
    Pošlje ukaz LIST na STM32 in iz odgovora izlušči imena .BIN datotek.

    STM32 ne vrne samo imen datotek:

        Listing files...
        Volume is FAT32
        LOG001.BIN     2673
        LOG002.BIN     13527

    Ne preverja konca vrstice z ".BIN",
    Vrstica se konča z velikostjo datoteke:  2673, 13527
    """

    # Pošljemo ukaz LIST na STM32.
    logger.ser.write(b"LIST\r\n")

    # read_until_idle bere, dokler 2s ne prejema novih podatkov.
    response = read_until_idle(logger, idle_timeout=2.0).decode(errors="ignore")

    # debug izpis v terminalu - journalctl
    logging.info("STM32 LIST response:\n%s", response)

    # Če STM32 vrne ERROR, potem ne nadaljujemo,
    if "ERROR" in response:
        raise RuntimeError(response.strip())

    # iscemo LOG + 0023 + .BIN
    files = re.findall(r"LOG\d+\.BIN", response, flags=re.IGNORECASE)

    # ker se v serial izpisu lahko isto ime pojavi veckrat 
    # nardimo seznam brez duplikatov in ohranimo vrstni red.
    unique_files = []
    seen = set()

    for file in files:
        # vsa imena na CAPS
        file = file.upper()

        # Če tega imena še nismo dodali, ga dodamo
        if file not in seen:
            seen.add(file)
            unique_files.append(file)

    # Debug izpis
    logging.info("Parsed STM32 files: %s", unique_files)

    # Vrne seznam dat. - uporabljajo GET_LAST, GET_ALL in GET_FILE.
    return unique_files

def validate_filename(filename: str) -> str:
    """
    Preveri, da je ime datoteke varno in v pričakovani obliki.

    Dovolimo samo imena tipa:
        LOG001.BIN
        LOG23.BIN
        LOG999.BIN

    Preprečimo nevarne poti:
        ../../nekaj
        /etc/passwd
        test.txt
    """

    if filename is None:
        raise RuntimeError("Missing filename")

    # Odstrani presledke in spremeni v uppercase
    filename = filename.strip().upper()

    # Sprejmemo samo LOG + številke + .BIN.
    # Prepreči pisanje izven DATA_DIR.
    if not re.fullmatch(r"LOG\d+\.BIN", filename):
        raise RuntimeError(f"Invalid filename: {filename}")

    return filename

def stm32_get_file(logger: DataLogger, filename: str) -> Path:
    """
    Iz STM32 prenese eno .BIN datoteko in jo shrani v DATA_DIR.

    Primer:
        filename = LOG001.BIN
        shrani v: stm32_data/LOG001.BIN
    """

    filename = validate_filename(filename)

    logging.info("Requesting file from STM32: %s", filename)

    # STM32 ukaz za prenos datoteke
    logger.ser.write(f"GET {filename}\r\n".encode())

    # pri prenosu dat, daljši timeout, ker je dat lahko velika
    data = read_until_idle(logger, idle_timeout=4.0)

    # če nismo prejeli podatkov ne smemo ustvarjati datoteke
    if not data:
        raise RuntimeError(f"No data received for {filename}")
    
    # Če STM32 vrne tekstovno napako, zaznamo in ne nadaljujemo s parsiranjem.
    text_preview = data[:200].decode(errors="ignore")
    if "ERROR" in text_preview or "FAIL" in text_preview:
        raise RuntimeError(f"STM32 returned error while reading {filename}: {text_preview.strip()}")
    
    # Shranimo v namensko mapo
    path = DATA_DIR / filename
    path.write_bytes(data)

    logging.info("Saved raw STM32 file to: %s", path)

    return path


def stm32_process_file(logger: DataLogger, filename: str) -> None:
    """
    Prenese .BIN datoteko iz STM32, jo sparsa in shrani obdelano .npz datoteko.

    Primer:
        stm32_data/LOG001.BIN
        stm32_data/LOG001.npz
    """
    # preverimo filename
    filename = validate_filename(filename)
    # prenesemo .BIN dat iz STM32
    path = stm32_get_file(logger, filename)

    logging.info("Parsing file: %s", path)

    # parsiranje .bin v pakete
    packets = logger.parse_file(str(path))

    # če ni veljavnega paketa ne shranimo praznega .npz
    if not packets:
        raise RuntimeError(f"No valid packets parsed from {filename}")

    # isto ime druga končnica
    npz_path = DATA_DIR / (path.stem + ".npz")

    # shranimo obdelane podatke
    logger.save_data(str(npz_path), packets)

    logging.info("Saved processed NPZ file to: %s", npz_path)


def get_files_from_stm32(port: str, which: str = "all", filename: str | None = None):
    """
    Glavna funkcija za GET_ALL, GET_LAST in GET_FILE.

    Najprej odpre STM32, naredi LIST, potem glede na parameter 
    which - prenese eno ali več datotek.
    """

    logger = stm32_open(port)

    try:
        # vprašamo STM32 katere dat obstajajo
        files = stm32_list_files(logger)
        # če STM32 nima nobene LOGxxx.BIN dat ne nadaljujemo
        if not files:
            raise RuntimeError("No files on STM32")

        if which == "all":
            # GET_ALL - obdelamo vse dat
            for file in files:
                stm32_process_file(logger, file)
        elif which == "last":
            # GET_LAST - obdelamo zadnjo dat iz seznama
            stm32_process_file(logger, files[-1])
        elif which == "file":
            # GET_FILE - obdelamo določeno dat
            # preveri filename
            filename = validate_filename(filename)

            # Preverimo, če dat. res obstaja na STM32.
            if filename not in files:
                raise RuntimeError(f"File {filename} not found on STM32")

            stm32_process_file(logger, filename)
        else:
            # Če nekdo pokliče napačen mode 
            raise RuntimeError(f"Invalid transfer mode: {which}")
    finally:
        # Port vedno zapremo, drugače naslednji ukaz ob kakšni napaki ne more več odpreti porta
        stm32_close(logger)


def stm32_delete(port: str) -> None:
    """
    Pošlje ukaz DELETE na STM32 in preveri odgovor.
    """
    logger = stm32_open(port)

    try:
        logging.info("Sending DELETE command to STM32")
        logger.ser.write(b"DELETE\r\n")

        # Preberemo odgovor STM32-ja, da vidimo, ali je ukaz uspel.
        response = read_until_idle(logger, idle_timeout=3.0).decode(errors="ignore")

        logging.info("STM32 DELETE response:\n%s", response)

        if "ERROR" in response or "FAIL" in response:
            raise RuntimeError(response.strip())

    finally:
        
        stm32_close(logger)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info(
        "Starting SPO STM32 service on %s:%s (WORK_DIR=%s)", HOST, PORT, WORK_DIR
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen()

        threading.Thread(target=stm32_monitor, daemon=True).start()
        while True:
            conn, addr = server.accept()
            thread = threading.Thread(
                # handle_client teče v svojem threadu
                target=handle_client,
                args=(conn, addr),
                daemon=True,
            )
            thread.start()


if __name__ == "__main__":
    main()