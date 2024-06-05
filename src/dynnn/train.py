from typing import Callable

from dynnn.train.dnn_train import dnn_train


def train(args: dict, data: dict, plot_loss_callback: Callable | None = None):
    return dnn_train(args, data, plot_loss_callback)
