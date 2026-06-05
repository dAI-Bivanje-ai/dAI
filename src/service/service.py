import socket
import threading
import serial.tools.list_ports
import time
import serial
from src.data_logger.data_logger import DataLogger
from pathlib import Path

WORK_DIR = Path("/opt/app")


HOST = "127.0.0.1"
PORT = 5000
STM32_VID = "0483"
STM32_PID = "5740"
INTERVAL = 1
stm32_port: str | None = None
stm32_lock = threading.Lock()

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

                command = data.decode().strip()

                if command == "STATUS":
                    with stm32_lock:
                        port = stm32_port
                    if port:
                        response = "STM32 is connected\n"
                    else:
                        response = "FAIL: STM32 is not connected\n"

                elif command == "STOP":
                    response = "Stopping service\n"
                    conn.sendall(response.encode())
                    break

                elif command == "GET_LAST":
                    with stm32_lock:
                        port = stm32_port
                    if port is None:
                        response = "FAIL: STM32 is not connected\n"
                    else:
                        try:
                            get_files_from_stm32(port, which="last")
                            response = "Last file from STM32 has been processed\n"
                        except Exception as e:
                            response = f"FAIL: {e}\n"

                elif command == "GET_ALL":
                    with stm32_lock:
                        port = stm32_port
                    if port is None:
                        response = "FAIL: STM32 is not connected\n"
                    else:
                        try:
                            get_files_from_stm32(port, which="all")
                            response = "All files from STM32 are processed\n"
                        except Exception as e:
                            response = f"FAIL: {e}\n"

                elif command.startswith("GET_FILE|"):
                    with stm32_lock:
                        port = stm32_port
                    if port is None:
                        response = "FAIL: STM32 is not connected\n"
                    else:
                        filename = command.split("|", 1)[1]
                        try:
                            get_files_from_stm32(port, which="file", filename=filename)
                            response = f"File {filename} from STM32 has been processed\n"
                        except Exception as e:
                            response = f"FAIL: {e}\n"

                elif command == "DELETE":
                    with stm32_lock:
                        port = stm32_port
                    if port is None:
                        response = "FAIL: STM32 is not connected\n"
                    else:
                        try:
                            stm32_delete(port)
                            response = "All files on STM32 are deleted\n"
                        except Exception as e:
                            response = f"FAIL: {e}\n"

                else:
                    response = f"UNKNOWN: {command}\n"

                conn.sendall(response.encode())
        except OSError:
            pass
        finally:
            with clients_lock:
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
    data_logger.ser.write(b"OFF\r\n")
    time.sleep(1)
    data_logger.ser.reset_input_buffer()
    return data_logger


def stm32_close(logger: DataLogger) -> None:
    logger.ser.write(b"LOG\r\n")  # preklopimo v LOG mode
    time.sleep(0.2)
    logger.close()


def stm32_list_files(logger: DataLogger) -> list[str]:
    logger.ser.write(b"LIST\r\n")
    time.sleep(0.5)
    response = logger.ser.read_all().decode(errors="ignore")
    files = []
    for line in response.splitlines():
        line = line.strip()
        if line.endswith(".BIN"):
            files.append(line)
    return files


def stm32_get_file(logger: DataLogger, filename: str) -> Path:
    logger.ser.write(f"GET {filename}\r\n".encode())
    time.sleep(1)
    data = logger.ser.read_all()
    path = WORK_DIR / filename
    path.write_bytes(data)
    return path


def stm32_process_file(logger: DataLogger, filename: str) -> None:
    path = stm32_get_file(logger, filename)
    packets = logger.parse_file(str(path))
    npz_path = WORK_DIR / (path.stem + ".npz")
    logger.save_data(str(npz_path), packets)


def get_files_from_stm32(port: str, which: str = "all", filename: str | None = None):
    logger = stm32_open(port)
    files = stm32_list_files(logger)

    if which == "all":
        for file in files:
            stm32_process_file(logger, file)
    elif which == "last":
        stm32_process_file(logger, files[-1])
    elif which == "file":
        stm32_process_file(logger, filename)

    stm32_close(logger)


def stm32_delete(port: str) -> None:
    logger = stm32_open(port)
    logger.ser.write(b"DELETE\r\n")
    time.sleep(1)
    stm32_close(logger)


def main():

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
