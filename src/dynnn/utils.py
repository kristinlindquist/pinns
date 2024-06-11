import os
import math
import json
import pickle
import statistics
import sys
import torch
from torchdyn.numerics.odeint import odeint
from typing import Any, Callable, Sequence

MODEL_BASE_DIR = sys.path[0] + "/../models"
DATA_BASE_DIR = sys.path[0] + "/../data"


def load_data(data_file: str) -> Any:
    """
    Load data file from disk

    Args:
        data_file (str): data file name (without path)
    """
    file_path = f"{DATA_BASE_DIR}/{data_file}"
    print(f"Loading data from {file_path}")
    with open(file_path, "rb") as file:
        data = pickle.loads(file.read())

    return data


def save_data(data: Any, data_file: str) -> str:
    """
    Save data file to disk

    Args:
        data (Any): data file to save
        data_file (str): data file name (without path)
    """
    if not os.path.exists(DATA_BASE_DIR):
        os.makedirs(DATA_BASE_DIR)

    file_path = f"{DATA_BASE_DIR}/{data_file}"
    with open(file_path, "wb") as file:
        pickle.dump(data, file)

    return file_path


def load_or_create_data(
    data_file: str, create_if_nx: Callable[[], Any] | None = None
) -> Any:
    """
    Load data file from disk, optionally creating new data if not found

    Args:
        data_file (str): data file name (without path)
        create_if_nx (Callable[[], Any]): function to create new data if not found

    Returns:
        Any: loaded or created data
    """
    try:
        return load_data(data_file)
    except FileNotFoundError:
        print(f"Data file {data_file} not found.")
        if create_if_nx is not None:
            print(f"Creating new data...")
            data = create_if_nx()
            save_data(data, data_file)

        return data


def save_model(model: torch.nn.Module, run_id: str):
    """
    Save model to disk

    Args:
        model (torch.nn.Module): model to save
        run_id (str): a unique identifier for the model
    """
    if not os.path.exists(MODEL_BASE_DIR):
        os.makedirs(MODEL_BASE_DIR)

    file_path = f"{MODEL_BASE_DIR}/dynnn-{run_id}.pt"
    print("Saving model to", file_path)
    torch.save(model, file_path)


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
    model = torch.load("file_path.pth")
    model.eval()
    return model


def flatten_dict(nested_dict: dict, prefix: str = "") -> dict:
    """
    Flattens a nested dictionary into a single-level dictionary.

    Args:
        nested_dict (dict): nested dictionary
        prefix (str): prefix for keys

    Returns:
        dict: flattened dictionary
    """
    flat_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            flat_dict.update(flatten_dict(value, prefix=prefix + key + "."))
        else:
            flat_dict[prefix + key] = value
    return flat_dict


def unflatten_dict(flat_dict: dict) -> dict:
    """
    Unflattens a single-level dictionary into a nested dictionary.

    Args:
        flat_dict (dict): flattened dictionary

    Returns:
        dict: nested dictionary
    """
    nested_dict = {}
    for key, value in flat_dict.items():
        parts = key.split(".")
        current_dict = nested_dict
        for part in parts[:-1]:
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]
        current_dict[parts[-1]] = value
    return nested_dict


def coerce_int(value: Any, allow_none: bool = False) -> int | None:
    """
    Coerce value to an integer

    Args:
        value (Any): value to coerce
        allow_none (bool): allow None values (otherwise, None -> 0.0)

    Returns:
        int | None: coerced integer value
    """
    if value is None:
        if allow_none:
            return None
        return 0
    return int(value)
