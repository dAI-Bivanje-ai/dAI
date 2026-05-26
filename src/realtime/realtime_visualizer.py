import numpy as np
import matplotlib.pyplot as plt


class RealtimeSpectrogramVisualizer:
    """
    Realtime prikaz spektrogramov za pospeškometer in giroskop.

    Vhod v update():
        acc_spec  -> spektrogram pospeškometra oblike (freq_bins, segment_length, 3)
        gyro_spec -> spektrogram giroskopa oblike (freq_bins, segment_length, 3)
    """

    def __init__(self) -> None:
        plt.ion()

        self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 7))

        self.acc_img = None
        self.gyro_img = None

        self.update_counter = 0

        self.setup_plot()

        plt.tight_layout()
        plt.show(block=False)

    def setup_plot(self) -> None:
        self.ax[0].set_title("Pospeškometer - spektrogram")
        self.ax[0].set_xlabel("Časovni segment")
        self.ax[0].set_ylabel("Frekvenčni bin")

        self.ax[1].set_title("Giroskop - spektrogram")
        self.ax[1].set_xlabel("Časovni segment")
        self.ax[1].set_ylabel("Frekvenčni bin")

    def update(
        self,
        acc_spec: np.ndarray,
        gyro_spec: np.ndarray,
        time_sec: float | None = None,
    ) -> None:
        self.update_counter += 1

        acc_data = self.calculate_magnitude(acc_spec)
        gyro_data = self.calculate_magnitude(gyro_spec)

        if self.acc_img is None:
            self.acc_img = self.ax[0].imshow(
                acc_data,
                aspect="auto",
                origin="lower",
                vmin=0,
                vmax=1,
            )

            self.gyro_img = self.ax[1].imshow(
                gyro_data,
                aspect="auto",
                origin="lower",
                vmin=0,
                vmax=1,
            )

            self.fig.colorbar(self.acc_img, ax=self.ax[0])
            self.fig.colorbar(self.gyro_img, ax=self.ax[1])

        else:
            self.acc_img.set_data(acc_data)
            self.gyro_img.set_data(gyro_data)

        if time_sec is not None:
            self.fig.suptitle(
                f"Realtime spektrogram | čas: {time_sec:.2f} s | update: {self.update_counter}"
            )
        else:
            self.fig.suptitle(
                f"Realtime spektrogram | update: {self.update_counter}"
            )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        plt.pause(0.001)

    def calculate_magnitude(self, spectrogram: np.ndarray) -> np.ndarray:
        x = spectrogram[:, :, 0]
        y = spectrogram[:, :, 1]
        z = spectrogram[:, :, 2]

        return np.sqrt(x**2 + y**2 + z**2)

    def is_open(self) -> bool:
        return plt.fignum_exists(self.fig.number)