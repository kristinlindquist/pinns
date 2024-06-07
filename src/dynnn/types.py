from typing import Annotated, Any, Callable, Literal
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
    validator,
)
from pydantic.functional_validators import AfterValidator
from enum import Enum

GeneratorFunction = Callable[[torch.Tensor], torch.Tensor]
InitialConditionsFunction = Callable[
    [int, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


def float_tensor(value) -> float | torch.Tensor:
    if isinstance(value, float):
        return value

    if isinstance(value, torch.Tensor):
        return torch.clamp(value, min=cls.ge, max=cls.le).item()

    raise ValueError(f"Unsupported value type: {type(value)}")


FloatTensor = Annotated[float, AfterValidator(float_tensor)]


def long_tensor(value) -> int | torch.Tensor:
    if isinstance(value, int):
        return value

    if isinstance(value, torch.Tensor):
        return torch.clamp(value, min=cls.ge, max=cls.le).item()

    raise ValueError(f"Unsupported value type: {type(value)}")


LongTensor = Annotated[int, AfterValidator(long_tensor)]


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
    domain_min: LongTensor = Field(0, decorator=rl_param, ge=-1000, le=1000)
    domain_max: LongTensor = Field(10, decorator=rl_param)

    @validator("domain_max")
    def domain_max_gt_min(cls, domain_max, values, **kwargs):
        if domain_max <= values.get("domain_min"):
            return domain_max + (values.get("domain_min") - domain_max) + 1
        return domain_max


class DatasetArgs(HasSimulatorParams):
    num_samples: int | torch.Tensor = Field(2, decorator=rl_param, ge=1, le=1000)
    test_split: float | torch.Tensor = Field(0.8, ge=0.1, le=0.9)


class TrajectoryArgs(HasSimulatorParams):
    y0: torch.Tensor | None = None
    masses: torch.Tensor | None = None
    n_bodies: LongTensor | None = Field(None, decorator=rl_param, ge=1, le=10000)
    n_dims: LongTensor = Field(3, decorator=rl_param, ge=1, le=6)
    time_scale: LongTensor = Field(3, decorator=rl_param, ge=1, le=1000)
    model: torch.nn.Module | None = None
    odeint_rtol: FloatTensor = Field(1e-10, decorator=rl_param, ge=1e-12, le=1e-3)
    odeint_atol: FloatTensor = Field(1e-6, decorator=rl_param, ge=1e-12, le=1e-3)
    odeint_solver: OdeSolver = OdeSolver.TSIT5
    t_span_min: LongTensor = Field(0, decorator=rl_param, ge=0, le=1000)
    t_span_max: LongTensor = Field(3, decorator=rl_param)
    generator_type: GeneratorType = GeneratorType.HAMILTONIAN

    @validator("t_span_max")
    def t_span_max_gt_min(cls, t_span_max, values, **kwargs):
        if t_span_max <= values["t_span_min"]:
            return t_span_max + (values["t_span_min"] - t_span_max) + 1
        return t_span_max

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
    def load(encoded: dict[str, torch.Tensor]) -> "SimulatorParams":
        return SimulatorParams(
            dataset_args=DatasetArgs.decode_from_rl(encoded["dataset_args"]),
            trajectory_args=TrajectoryArgs.decode_from_rl(encoded["trajectory_args"]),
            model_args=ModelArgs.decode_from_rl(encoded.get("model_args", {})),
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
