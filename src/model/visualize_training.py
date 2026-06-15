from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

import json
import numpy as np
import matplotlib.pyplot as plt


def load_history(path):
    with open(path) as f:
        return json.load(f)


def plot_loss(ax, epochs, loss, title, color):
    ax.plot(epochs, loss, color=color, label="Train loss")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoha")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_accuracy(ax, epochs, train_acc, val_acc, title, color):
    best_epoch = int(np.argmax(val_acc)) + 1
    best_val = max(val_acc)

    ax.plot(epochs, train_acc, color=color, label="Train acc")
    ax.plot(epochs, val_acc, color="darkorange", linestyle="--", label="Val acc")

    ax.axvline(x=best_epoch, color="darkorange", linestyle=":", alpha=0.6)
    ax.annotate(
        f"best: {best_val:.3f}\n(epoha {best_epoch})",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + len(epochs) * 0.05, best_val - 0.12),
        color="darkorange",
        fontsize=8,
    )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoha")
    ax.set_ylabel("Točnost")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


def plot_overfitting(ax, epochs, train_acc, val_acc, title, color):
    gap = [t - v for t, v in zip(train_acc, val_acc)]
    ax.plot(epochs, gap, color=color)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoha")
    ax.set_ylabel("train_acc − val_acc")
    ax.grid(True, alpha=0.3)


def main():
    h_imu = load_history(ROOT_DIR / "models" / "history.json")
    h_mic = load_history(ROOT_DIR / "models" / "history_mic.json")

    epochs_imu = range(1, len(h_imu["train_loss"]) + 1)
    epochs_mic = range(1, len(h_mic["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    plot_loss(
        axes[0][0],
        epochs_imu,
        h_imu["train_loss"],
        "IMU model (ACC + GYRO) — Loss",
        color="steelblue",
    )
    plot_accuracy(
        axes[0][1],
        epochs_imu,
        h_imu["train_acc"],
        h_imu["val_acc"],
        "IMU model (ACC + GYRO) — Točnost",
        color="steelblue",
    )
    plot_overfitting(
        axes[0][2],
        epochs_imu,
        h_imu["train_acc"],
        h_imu["val_acc"],
        "IMU model (ACC + GYRO) — Overfitting gap",
        color="steelblue",
    )
    plot_loss(
        axes[1][0],
        epochs_mic,
        h_mic["train_loss"],
        "Mikrofon model — Loss",
        color="seagreen",
    )
    plot_accuracy(
        axes[1][1],
        epochs_mic,
        h_mic["train_acc"],
        h_mic["val_acc"],
        "Mikrofon model — Točnost",
        color="seagreen",
    )
    plot_overfitting(
        axes[1][2],
        epochs_mic,
        h_mic["train_acc"],
        h_mic["val_acc"],
        "Mikrofon model — Overfitting gap",
        color="seagreen",
    )

    plt.tight_layout()
    out_path = ROOT_DIR / "models" / "training_plot_combined.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Shranjeno: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
