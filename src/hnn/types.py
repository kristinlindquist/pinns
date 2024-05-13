from typing import Callable
import torch
from pydantic import BaseModel, ConfigDict

HamiltonianFunction = Callable[[torch.Tensor], torch.Tensor]


class HamiltonianField(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    meta: dict
    x: torch.Tensor
    dx: torch.Tensor


class TrajectoryArgs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    t_span: tuple[int, int]
    timescale: int = 5
    noise_std: float = 0.05


class FieldArgs(BaseModel):
    xmin: float = -1.2
    xmax: float = 1.2
    ymin: float = -1.2
    ymax: float = 1.2
    gridsize: int = 20


class DatasetArgs(BaseModel):
    num_samples: int = 30
    test_split: float = 0.5
