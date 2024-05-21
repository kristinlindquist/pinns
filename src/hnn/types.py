from typing import Callable
import torch
from pydantic import BaseModel, ConfigDict

HamiltonianFunction = Callable[[torch.Tensor], torch.Tensor]


class TrajectoryArgs(BaseModel):
    y0: torch.Tensor
    masses: torch.Tensor
    model_config = ConfigDict(arbitrary_types_allowed=True)
    timescale: int = 10
    noise_std: float = 0.0
    model: torch.nn.Module | None = None


class DatasetArgs(BaseModel):
    num_samples: int = 10
    test_split: float = 0.7


class Trajectory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    r: torch.Tensor
    v: torch.Tensor
    dr: torch.Tensor
    dv: torch.Tensor
    t: torch.Tensor
