from typing import Any, Callable, Literal
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator, validator
from enum import Enum

GeneratorFunction = Callable[[torch.Tensor], torch.Tensor]
InitialConditionsFunction = Callable[
    [int, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


class GeneratorType(Enum):
    LAGRANGIAN = 1
    HAMILTONIAN = 2

    @classmethod
    def _missing_(cls, value):
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"{value} is not a valid GeneratorType")


class OdeSolver(Enum):
    # On choosing an ODE solver: https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/
    TSIT5 = 1
    DOPRI5 = 2
    ALF = 3
    EULER = 4
    MIDPOINT = 5
    RK4 = 6
    IEULER = 7
    SYMPLECTIC = 8

    @classmethod
    def _missing_(cls, value):
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"{value} is not a valid OdeSolver")


class ModelArgs(BaseModel):
    domain_min: int = 0
    domain_max: int = 10

    @validator("domain_min")
    def domain_min_check(cls, v):
        return round(v, 0)

    @validator("domain_max")
    def domain_domain_check(cls, v):
        return round(v, 0)


class DatasetArgs(BaseModel):
    num_samples: int = 2
    test_split: float = 0.8

    @validator("num_samples")
    def num_samples_check(cls, v):
        if v < 1:
            return 1
        return round(v, 0)

    @validator("test_split")
    def test_split_check(cls, v):
        return round(v, 2)

    @property
    def filename(self) -> str:
        filename_parts = [
            f"num_samples_{self.num_samples}",
            f"test_split_{self.test_split}",
        ]
        return "_".join(filename_parts)


class Trajectory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    q: torch.Tensor
    p: torch.Tensor
    dq: torch.Tensor
    dp: torch.Tensor
    t: torch.Tensor


class TrajectoryArgs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    y0: torch.Tensor | None = None
    masses: torch.Tensor | None = None
    n_bodies: int | None = None
    n_dims: int = 3
    time_scale: int = 3
    model: torch.nn.Module | None = None
    odeint_rtol: float = 1e-10
    odeint_atol: float = 1e-6
    odeint_solver: OdeSolver = OdeSolver.TSIT5
    t_span_min: int = 0
    t_span_max: int = 3
    generator_type: GeneratorType = GeneratorType.HAMILTONIAN

    @validator("n_bodies")
    def n_bodies_check(cls, v):
        if v < 1:
            return 1
        return round(v, 0)

    @validator("n_dims")
    def n_dims_check(cls, v):
        if v < 1:
            return 1
        return round(v, 0)

    @validator("time_scale")
    def time_scale_check(cls, v):
        return round(v, 0)

    @validator("odeint_rtol")
    def odeint_rtol_check(cls, v):
        return round(v, 0)

    @validator("odeint_atol")
    def odeint_atol_check(cls, v):
        return round(v, 0)

    @validator("t_span_min")
    def t_span_min_check(cls, v):
        return round(v, 0)

    @validator("t_span_max")
    def t_span_max_check(cls, v):
        return round(v, 0)

    @property
    def filename(self) -> str:
        # TODO: masses, y0, domain
        filename_parts = [
            f"n_bodies_{self.n_bodies}",
            f"n_dims_{self.n_dims}",
            f"time_scale_{self.time_scale}",
            f"odeint_rtol_{self.odeint_rtol}",
            f"odeint_atol_{self.odeint_atol}",
            f"odeint_solver_{self.odeint_solver.name.lower()}",
            f"t_span_{self.t_span_min}-{self.t_span_max}",
            f"generator_type_{self.generator_type.name.lower()}",
        ]
        return "_".join(filename_parts)

    @model_validator(mode="before")
    def post_update(cls, values: dict[str, Any]) -> dict[str, Any]:
        if values.get("masses") is None and values.get("n_bodies") is None:
            raise ValueError("Either masses or n_bodies must be provided")

        if values.get("y0") is not None and values.get("masses") is None:
            raise ValueError("y0 and masses must be provided together")

        if values.get("masses") is not None:
            n_bodies = len(values["masses"])
            if values.get("n_bodies", n_bodies) != n_bodies:
                raise ValueError(
                    f"Number of bodies ({values['n_bodies']}) must match the number of masses ({n_bodies})"
                )
            values["n_bodies"] = n_bodies

        return values


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

    def encode(self) -> tuple[float, float]:
        return (
            self.min_train_loss,
            self.min_test_loss,
        )
