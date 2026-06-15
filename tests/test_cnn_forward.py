import torch

from src.model.cnn_model import CNNModel


def test_cnn_model_forward_shape():
    model = CNNModel(num_classes=5)

    acc = torch.randn(2, 3, 51, 3)
    gyro = torch.randn(2, 3, 51, 3)

    output = model(acc, gyro)

    assert output.shape == (2, 5)
