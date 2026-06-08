from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from torch.utils.data import DataLoader

from src.model.cnn_model import CNNModel as IMUModel
from src.model.cnn_model_mic import CNNModel as MicModel
from src.model.dataset_cnn import IMUDataset
from src.model.dataset_cnn_mic import MicDataset

VAL_NPZ_IMU = str(ROOT_DIR / "val_dataset.npz")
VAL_NPZ_MIC = str(ROOT_DIR / "val_dataset_mic.npz")

IMU_MODEL_PATH = str(ROOT_DIR / "models" / "imu_cnn.pt")
MIC_MODEL_PATH = str(ROOT_DIR / "models" / "mic_cnn.pt")

IMU_CLASSES = ["Delo", "Telefon"]
MIC_CLASSES = ["Glasba", "Pogovor"]


def compute_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def plot_confusion_matrix(cm, class_names, title, ax):
    total = cm.sum()
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="Dejanski razred",
        xlabel="Napovedani razred",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = 100 * cm[i, j] / total if total > 0 else 0
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    ax.set_xlim(-0.5, len(class_names) - 0.5)
    ax.set_ylim(len(class_names) - 0.5, -0.5)


def evaluate_imu():
    dataset = IMUDataset(VAL_NPZ_IMU)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = IMUModel(num_classes=2)
    model.load_state_dict(torch.load(IMU_MODEL_PATH, map_location="cpu"))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for (acc, gyro), y in loader:
            outputs = model(acc, gyro)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())

    return np.array(all_labels), np.array(all_preds)


def evaluate_mic():
    dataset = MicDataset(VAL_NPZ_MIC)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    checkpoint = torch.load(MIC_MODEL_PATH, map_location="cpu")
    model = MicModel(num_classes=2)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())

    return np.array(all_labels), np.array(all_preds)


def main():
    y_true_imu, y_pred_imu = evaluate_imu()
    y_true_mic, y_pred_mic = evaluate_mic()

    cm_imu = compute_confusion_matrix(y_true_imu, y_pred_imu, num_classes=2)
    cm_mic = compute_confusion_matrix(y_true_mic, y_pred_mic, num_classes=2)

    acc_imu = (y_true_imu == y_pred_imu).mean() * 100
    acc_mic = (y_true_mic == y_pred_mic).mean() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    plot_confusion_matrix(
        cm_imu,
        IMU_CLASSES,
        f"IMU model – Confusion Matrix\nVal accuracy: {acc_imu:.1f}%",
        ax1,
    )
    plot_confusion_matrix(
        cm_mic,
        MIC_CLASSES,
        f"Mikrofon model – Confusion Matrix\nVal accuracy: {acc_mic:.1f}%",
        ax2,
    )

    plt.tight_layout()
    out_path = ROOT_DIR / "models" / "confusion_matrices.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Shranjeno: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
