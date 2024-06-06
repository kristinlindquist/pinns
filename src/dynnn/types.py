from typing import Callable, Literal
import torch
from pydantic import BaseModel, ConfigDict

GeneratorFunction = Callable[[torch.Tensor], torch.Tensor]
GeneratorType = Literal["lagrangian", "hamiltonian"]

# On choosing an ODE solver: https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/
OdeSolver = Literal[
    "tsit5", "dopri5", "alf", "euler", "midpoint", "rk4", "ieuler", "symplectic"
]


class DatasetArgs(BaseModel):
    num_samples: int = 20
    test_split: float = 0.8


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
    domain: tuple[int, int] = (0, 10)
    t_span: tuple[int, int] = (0, 100)
    generator_type: GeneratorType = "hamiltonian"


class PinnStats(BaseModel):
    train_loss: list[float] = []
    test_loss: list[float] = []
    train_additional_loss: list[float] = []
    test_additional_loss: list[float] = []

    @staticmethod
    def _calc_mean(values: list[float]) -> float:
        if len(values) == 0:
            return 0.0
        return math.mean(values)

    @staticmethod
    def _calc_min(values: list[float]) -> float:
        if len(values) == 0:
            return 0.0
        return min(values)

    @property
    def min_train_loss(self) -> float:
        return self._calc_min(self.train_loss)

    @property
    def min_test_loss(self) -> float:
        return self._calc_min(self.test_loss)
