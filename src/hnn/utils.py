import torch


def L2_loss(u, v):
    return (u - v).pow(2).mean()
