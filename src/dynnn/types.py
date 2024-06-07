from typing import Any, Callable, Literal
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator, validator
from enum import Enum
from dynnn.utils import unflatten_dict

GeneratorFunction = Callable[[torch.Tensor], torch.Tensor]
InitialConditionsFunction = Callable[
    [int, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


def rl_param(attr):
    attr.metadata["is_rl_trainable"] = True


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


class HasSimulatorParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """
    Base class for models that have rl-learnable simulator parameters
    """

    @classmethod
    @property
    def rl_fields(cls):
        return [
            field_name
            for field_name, field in cls.model_fields.items()
            if (field.json_schema_extra or {}).get("decorator") == rl_param
        ]

    def encode_for_rl(self) -> dict:
        return {field_name: getattr(self, field_name) for field_name in self.rl_fields}

    @classmethod
    def decode_from_rl(cls, encoded: dict) -> "HasSimulatorParams":
        return cls(**{k: v for k, v in encoded.items() if k in cls.rl_fields})

    @property
    def filename(self) -> str:
        filename_parts = [f"{k}-{v}" for k, v in self.encode_for_rl().items()]
        return "_".join(filename_parts)


class ModelArgs(HasSimulatorParams):
    domain_min: int = Field(0, decorator=rl_param)
    domain_max: int = Field(10, decorator=rl_param)

    @validator("domain_min")
    def domain_min_check(cls, v):
        return round(v, 0)

    @validator("domain_max")
    def domain_domain_check(cls, v):
        return round(v, 0)


class DatasetArgs(HasSimulatorParams):
    num_samples: int = Field(2, decorator=rl_param)
    test_split: float = 0.8

    @validator("num_samples")
    def num_samples_check(cls, v):
        if v < 1:
            return 1
        return round(v, 0)

    @validator("test_split")
    def test_split_check(cls, v):
        return round(v, 2)


class TrajectoryArgs(HasSimulatorParams):
    y0: torch.Tensor | None = None
    masses: torch.Tensor | None = None
    n_bodies: int | None = Field(None, decorator=rl_param)
    n_dims: int = Field(3, decorator=rl_param)
    time_scale: int = Field(3, decorator=rl_param)
    model: torch.nn.Module | None = None
    odeint_rtol: float = Field(1e-10, decorator=rl_param)
    odeint_atol: float = Field(1e-6, decorator=rl_param)
    odeint_solver: OdeSolver = OdeSolver.TSIT5
    t_span_min: int = Field(0, decorator=rl_param)
    t_span_max: int = Field(3, decorator=rl_param)
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

    @model_validator(mode="before")
    def pre_update(cls, values: dict[str, Any]) -> dict[str, Any]:
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


class SimulatorParams(BaseModel):
    dataset_args: DatasetArgs
    trajectory_args: TrajectoryArgs
    model_args: ModelArgs

    @property
    def filename(self) -> str:
        filename_parts = [
            self.dataset_args.filename,
            self.trajectory_args.filename,
            self.model_args.filename,
        ]
        return "_".join(filename_parts)

    def encode(self):
        return {
            **self.dataset_args.encode_for_rl(),
            **self.trajectory_args.encode_for_rl(),
            **self.model_args.encode_for_rl(),
        }

    @staticmethod
    def load(encoded: dict) -> "SimulatorParams":
        dataset_args = DatasetArgs.decode_from_rl(encoded["dataset_args"])
        trajectory_args = TrajectoryArgs.decode_from_rl(encoded["trajectory_args"])
        model_args = ModelArgs.decode_from_rl(encoded.get("model_args", {}))
        return SimulatorParams(
            dataset_args=dataset_args,
            trajectory_args=trajectory_args,
            model_args=model_args,
        )


class PinnStats(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    train_loss: list[torch.Tensor] = []
    test_loss: list[torch.Tensor] = []
    train_additional_loss: list[torch.Tensor] = []
    test_additional_loss: list[torch.Tensor] = []

    @staticmethod
    def _calc_mean(values: list[torch.Tensor]) -> torch.Tensor:
        if len(values) == 0:
            return torch.tensor(0.0)
        return torch.cat(values).mean()

    @staticmethod
    def _calc_min(values: list[torch.Tensor]) -> torch.Tensor:
        if len(values) == 0:
            return torch.tensor(0.0)
        return torch.cat(values).min()

    @property
    def min_train_loss(self) -> torch.Tensor:
        return self._calc_min(self.train_loss)

    @property
    def min_test_loss(self) -> torch.Tensor:
        return self._calc_min(self.test_loss)

    def encode(self) -> tuple[float, float]:
        return (
            self.min_train_loss,
            self.min_test_loss,
        )


class Trajectory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    q: torch.Tensor
    p: torch.Tensor
    dq: torch.Tensor
    dp: torch.Tensor
    t: torch.Tensor
