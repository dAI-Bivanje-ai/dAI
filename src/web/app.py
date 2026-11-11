import threading
import time
import logging

from flask import Flask, jsonify, render_template

from src.config import WEB_HOST, WEB_PORT
from src.realtime.main_rt import run_pipeline
from src.realtime.serial_reader import LiveSerialReader

current_state = {"imu": None, "mic": None, "rms": 0, "time": 0, "error": False, "connected": False}


def website_save(imu, mic, rms):
    current_state["imu"] = imu
    current_state["mic"] = mic
    current_state["rms"] = rms
    current_state["time"] = time.time()


def pipeline_runner(reader):
    try:
        run_pipeline(website_save, reader)
    except Exception:
        logging.exception("Pipeline se je ustavil")
        current_state["error"] = True


app = Flask(__name__)


@app.route("/")
def run_app():
    return render_template("index.html")


@app.route("/api/state")
def get_state():
    current_state["connected"] = reader.is_connected()
    return jsonify(current_state)


if __name__ == "__main__":
    reader = LiveSerialReader()

    pipeline_thread = threading.Thread(target=pipeline_runner, args=(reader,))
    pipeline_thread.start()

    app.run(host=WEB_HOST, port=WEB_PORT)

    reader.running = False
    pipeline_thread.join()
