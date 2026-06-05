import socket
import threading


HOST = "127.0.0.1"
PORT = 5000


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
