import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.model.dataset_cnn import IMUDataset
from src.model.cnn_model import CNNModel


dataset = IMUDataset("dataset.npz")

print("Število primerov:", len(dataset))

(acc, gyro), y = dataset[0]

print("ACC shape:", acc.shape)
print("GYRO shape:", gyro.shape)
print("Label:", y)

model = CNNModel(num_classes=5)

acc = acc.unsqueeze(0)
gyro = gyro.unsqueeze(0)

output = model(acc, gyro)

print("Output shape:", output.shape)
print("Output:", output)