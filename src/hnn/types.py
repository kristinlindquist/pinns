from typing import Callable
import torch
from pydantic import BaseModel, ConfigDict

HamiltonianFunction = Callable[[torch.Tensor], torch.Tensor]


class TrajectoryArgs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    y0: torch.Tensor
    masses: torch.Tensor
    timescale: int = 10
    noise_std: float = 0.01
    model: torch.nn.Module | None = None


class DatasetArgs(BaseModel):
    num_samples: int = 30
    test_split: float = 0.7


class Trajectory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    r: torch.Tensor
    v: torch.Tensor
    dr: torch.Tensor
    dv: torch.Tensor
    t: torch.Tensor
