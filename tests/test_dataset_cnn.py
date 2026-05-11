import numpy as np

from src.model.dataset_cnn import IMUDataset


def test_imu_dataset_shape(tmp_path):
    path = tmp_path / "dataset.npz"

    x_acc = np.random.randint(0, 255, size=(10, 51, 3), dtype=np.uint8)
    x_gyro = np.random.randint(0, 255, size=(10, 51, 3), dtype=np.uint8)
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

    np.savez(path, X_acc=x_acc, X_gyro=x_gyro, y=y)

    dataset = IMUDataset(path, sensor="acc")
    x, label = dataset[0]

    assert len(dataset) == 10
    assert x.shape == (1, 51, 3)
    assert label.item() in [0, 1, 2, 3, 4]