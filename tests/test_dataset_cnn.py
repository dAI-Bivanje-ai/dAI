# import numpy as np

# from src.model.dataset_cnn import IMUDataset


# def test_imu_dataset_shape(tmp_path):
#     path = tmp_path / "dataset.npz"

#     x_acc = np.random.rand(10, 51, 3).astype(np.float32)
#     x_gyro = np.random.rand(10, 51, 3).astype(np.float32)
#     y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

#     np.savez(path, X_acc=x_acc, X_gyro=x_gyro, y=y)

#     dataset = IMUDataset(path)

#     (acc, gyro), label = dataset[0]

#     assert len(dataset) == 10

#     assert acc.shape == (1, 51, 3)
#     assert gyro.shape == (1, 51, 3)

#     assert label.item() in [0, 1, 2, 3, 4]
