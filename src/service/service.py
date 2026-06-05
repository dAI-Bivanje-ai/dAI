import socket
import threading
import serial.tools.list_ports
import time

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
        conn.sendall(b"Connected to SPO STM32 service\n")

        while True:

            data = conn.recv(1024)

            if not data:
                break

            command = data.decode().strip()

            if command == "STATUS":
                response = "SPO STM32 service is running\n"

            elif command == "STOP":
                response = "Stopping service\n"
                conn.sendall(response.encode())
                break

            else:

                response = f"UNKNOWN: {command}\n"

            conn.sendall(response.encode())


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
    find_stm32_port()
