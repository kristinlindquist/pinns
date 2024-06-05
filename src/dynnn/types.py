from typing import Callable, Literal
import torch
from pydantic import BaseModel, ConfigDict

SystemFunction = Callable[[torch.Tensor], torch.Tensor]
SystemType = Literal["lagrangian", "hamiltonian"]

# On choosing an ODE solver: https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/
OdeSolver = Literal[
    "tsit5", "dopri5", "alf", "euler", "midpoint", "rk4", "ieuler", "symplectic"
]


class DatasetArgs(BaseModel):
    num_samples: int = 20
    test_split: float = 0.8


class ModelArgs(BaseModel):
    domain: tuple[int, int] = (0, 10)
    t_span: tuple[int, int] = (0, 100)
    system_type: SystemType = "hamiltonian"


class Trajectory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    q: torch.Tensor
    p: torch.Tensor
    dq: torch.Tensor
    dp: torch.Tensor
    t: torch.Tensor


class TrajectoryArgs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    y0: torch.Tensor
    masses: torch.Tensor
    time_scale: int = 3
    model: torch.nn.Module | None = None
    odeint_rtol: float = 1e-10
    odeint_atol: float = 1e-6
    odeint_solver: OdeSolver = "tsit5"
