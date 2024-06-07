import os
import math
import json
import statistics
import sys
import torch
from torchdyn.numerics.odeint import odeint
from typing import Sequence


def l2_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return (y_true - y_pred).pow(2).mean()


def get_timepoints(
    t_span_min: int, t_span_max: int, time_scale: int = 30
) -> torch.Tensor:
    return torch.linspace(
        t_span_min, t_span_max, int(time_scale * (t_span_max - t_span_min))
    )


def permutation_tensor() -> torch.Tensor:
    """
    Constructs the Levi-Civita permutation tensor for 3 dimensions.
    """
    P = torch.zeros((3, 3, 3))
    P[0, 1, 2] = 1
    P[1, 2, 0] = 1
    P[2, 0, 1] = 1
    P[2, 1, 0] = -1
    P[1, 0, 2] = -1
    P[0, 2, 1] = -1
    return P


def integrate_model(
    model,
    t_span_min: int,
    t_span_max: int,
    y0: torch.Tensor,
    time_scale: int = 30,
    **kwargs,
):
    def fun(t, x):
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        _x = x.clone().detach().requires_grad_()
        dx = model.forward(_x).data
        return dx

    t = get_timepoints(t_span_min, t_span_max, time_scale)
    return odeint(fun, t=t, y0=y0, **kwargs)


MODEL_BASE_DIR = sys.path[0] + "/../models"


def save_model(model: torch.nn.Module, run_id: str):
    """
    Save model to disk
    """
    if not os.path.exists(MODEL_BASE_DIR):
        os.makedirs(MODEL_BASE_DIR)

    file_path = f"{MODEL_BASE_DIR}/dynnn-{run_id}.pt"
    print("Saving model to", file_path)
    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save(file_path)


def save_stats(stats: dict, run_id: str):
    """
    Save model stats
    """
    if not os.path.exists(MODEL_BASE_DIR):
        os.makedirs(MODEL_BASE_DIR)

    file_path = f"{MODEL_BASE_DIR}/stats-dynnn-{run_id}.json"
    print("Saving stats to", file_path)
    json.dump(stats, open(file_path, "w"))


def load_model(file_or_timestamp: str) -> torch.nn.Module:
    """
    Load model from disk
    """
    model_file = file_or_timestamp
    if not model_file.endswith(".pt"):
        model_file += ".pt"
    if not model_file.startswith("dynnn-"):
        model_file = f"dynnn-{model_file}"

    file_path = f"{MODEL_BASE_DIR}/{model_file}"
    model = torch.jit.load(file_path)
    model.eval()
    return model


def load_stats(file_or_timestamp: str) -> dict[str, list]:
    """
    Load stats from disk
    """
    stats_file = file_or_timestamp
    if not stats_file.endswith(".json"):
        stats_file += ".json"
    if not stats_file.startswith("stats-dynnn-"):
        stats_file = f"stats-dynnn-{stats_file}"

    return json.load(open(f"{MODEL_BASE_DIR}/{stats_file}"))
