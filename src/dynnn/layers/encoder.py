from enum import Enum
import torch

from dynnn.utils import flatten_dict, unflatten_dict


def encode_value(value) -> int | float:
    """
    Encodes a value into a number
    """
    if isinstance(value, torch.Tensor):
        return value.mean().item()

    if isinstance(value, int | float):
        return value

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, tuple):
        return hash(value)  # not reversible

    raise ValueError(f"Unsupported value type: {type(value)}")


def decode_value(value):
    """
    Decodes a value from a tensor or keeps it unchanged.
    """
    if isinstance(value, torch.Tensor):
        return value.item()
    else:
        return value


def encode_params(params: dict) -> torch.Tensor:
    """
    Encodes a nested dictionary of params into a tensor.
    """
    flat_params = flatten_dict(params)
    params_tensor = torch.tensor(
        [encode_value(v) for v in flat_params.values()], dtype=torch.float32
    )
    return params_tensor


def decode_params(params_tensor: torch.Tensor, params_template: dict) -> dict:
    """
    Decodes a tensor into a nested dictionary of params.
    """
    decoded_values = params_tensor.split(split_size=1)
    flat_params = {
        key: decode_value(val)
        for key, val in zip(flatten_dict(params_template).keys(), decoded_values)
    }
    nested_params = unflatten_dict(flat_params)
    return nested_params
