from typing import Callable
import torch
from pydantic import BaseModel, ConfigDict

HamiltonianFunction = Callable[[torch.Tensor], torch.Tensor]


class TrajectoryArgs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    y0: torch.Tensor
    masses: torch.Tensor
    time_scale: int = 5
    model: torch.nn.Module | None = None


class DatasetArgs(BaseModel):
    num_samples: int = 40
    test_split: float = 0.8


class ModelArgs(BaseModel):
    domain: tuple[int, int] = (0, 10)
    t_span: tuple[int, int] = (0, 30)
    use_lagrangian: bool = False


class Trajectory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    r: torch.Tensor
    v: torch.Tensor
    dr: torch.Tensor
    dv: torch.Tensor
    t: torch.Tensor
