from typing import Annotated, Any, Callable, Literal
import torch
import torch.nn.functional as F
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
    validator,
)
from pydantic.functional_validators import BeforeValidator
from enum import Enum
from functools import partial


def coerce_int(value: Any, allow_none: bool = False) -> int | None:
    if value is None:
        if allow_none:
            return None
        return 0
    return int(value)


ForcedInt = Annotated[int, BeforeValidator(coerce_int)]
ForcedIntOrNone = Annotated[int, BeforeValidator(partial(coerce_int, allow_none=True))]


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


def rl_param(attr):
    attr.metadata["is_rl_trainable"] = True


class HasSimulatorParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """
    Base class for models that have rl-learnable simulator parameters
    """

    @classmethod
    def rl_fields(cls) -> dict[str, type]:
        return {
            field_name: field.annotation
            for field_name, field in cls.model_fields.items()
            if (field.json_schema_extra or {}).get("decorator") == rl_param
        }

    @property
    def rl_param_sizes(self) -> dict[str, tuple[int, int]]:
        return {
            # TODO: metadata=[Ge(ge=0.1), Le(le=0.9)]
            field_name: (
                field.metadata[0].__getstate__()[0],  # pulls 0.1 from Ge(ge=0.1)
                field.metadata[1].__getstate__()[0],
            )
            for field_name, field in self.model_fields.items()
            if (field.json_schema_extra or {}).get("decorator") == rl_param
        }

    def encode_rl_params(self) -> dict:
        return {
            field_name: getattr(self, field_name) for field_name in self.rl_fields()
        }

    @classmethod
    def load_rl_params(cls, encoded: dict):
        return cls(**{f: v for f, v in encoded.items() if f in cls.rl_fields()})

    @classmethod
    def calc_rl_param_loss(cls, encoded: dict) -> torch.Tensor:
        actual = torch.stack([encoded[f] for f in cls.rl_fields()])

        adjusted = cls(**{f: v for f, v in encoded.items() if f in cls.rl_fields()})
        adjusted = torch.stack([getattr(adjusted, f) for f in cls.rl_fields()])

        res = F.mse_loss(actual, adjusted)
        return res

    @property
    def filename(self) -> str:
        filename_parts = [f"{k}-{v}" for k, v in self.encode_rl_params().items()]
        return "_".join(filename_parts)


class TrainingArgs(HasSimulatorParams):
    n_epochs: ForcedInt = Field(50, decorator=rl_param, ge=50, le=200)
    steps_per_epoch: int = Field(100, ge=20, le=10000)
    learning_rate: float = Field(1e-3, ge=1e-6, le=1e-1)  # rl_param
    weight_decay: float = Field(1e-4, ge=1e-6, le=1e-1)  # rl_param
    tolerance: int = 1e-1
    patience: int = 10
    min_epochs: int = 5


class ModelArgs(HasSimulatorParams):
    domain_min: ForcedInt = Field(0, decorator=rl_param, ge=-1000, le=1000)
    domain_max: ForcedInt = Field(10, decorator=rl_param, ge=-1000, le=1000)
    vector_field_type: Literal["solenoidal", "conservative", "port", "both"] = (
        "conservative"
    )
    hidden_dim: ForcedInt = Field(500, decorator=rl_param, ge=8, le=4096)
    use_invariant_layer: bool = False

    @model_validator(mode="after")
    def pre_update(cls, values: "ModelArgs") -> "ModelArgs":
        if values.domain_max <= values.domain_min:
            values.domain_max = (
                values.domain_max + (values.domain_min - values.domain_max) + 1
            )

        return values


class DatasetArgs(HasSimulatorParams):
    n_samples: ForcedInt = Field(2, decorator=rl_param, ge=2, le=5)
    test_split: float = Field(0.8, ge=0.1, le=0.9)


MAX_N_BODIES = 50
MIN_N_BODIES = 2


class TrajectoryArgs(HasSimulatorParams):
    y0: torch.Tensor | None = None
    masses: torch.Tensor | None = None
    n_bodies: ForcedIntOrNone = Field(
        5, decorator=rl_param, ge=MIN_N_BODIES, le=MAX_N_BODIES
    )
    n_dims: ForcedInt = Field(3, ge=1, le=6)  # decorator=rl_param)
    time_scale: ForcedInt = Field(3, decorator=rl_param, ge=1, le=10)
    model: torch.nn.Module | None = None
    odeint_rtol: float = Field(1e-10, ge=1e-12, le=1e-3)  # , decorator=rl_param)
    odeint_atol: float = Field(1e-6, ge=1e-12, le=1e-3)  # , decorator=rl_param)
    odeint_solver: OdeSolver = OdeSolver.TSIT5
    t_span_min: ForcedInt = Field(0, decorator=rl_param, ge=0, le=3)
    t_span_max: ForcedInt = Field(3, decorator=rl_param, ge=4, le=15)
    generator_type: GeneratorType = GeneratorType.HAMILTONIAN

    @model_validator(mode="after")
    def pre_update(cls, values: dict[str, Any]) -> dict[str, Any]:
        if values.masses is None and values.n_bodies is None:
            raise ValueError("Either masses or n_bodies must be provided")

        if values.y0 is not None and values.masses is None:
            raise ValueError("y0 and masses must be provided together")

        if values.masses is not None:
            n_bodies = len(values.masses)
            if values.n_bodies != n_bodies:
                print(f"Warning: setting n_bodies to {n_bodies}")
            values.n_bodies = n_bodies

        if values.t_span_max <= values.t_span_min:
            values.t_span_max = (
                values.t_span_max + (values.t_span_min - values.t_span_max) + 1
            )

        return values


class SimulatorParams(BaseModel):
    dataset_args: DatasetArgs = DatasetArgs()
    trajectory_args: TrajectoryArgs = TrajectoryArgs()
    model_args: ModelArgs = ModelArgs()
    training_args: TrainingArgs = TrainingArgs()

    @property
    def filename(self) -> str:
        filename_parts = [
            self.dataset_args.filename,
            self.trajectory_args.filename,
            self.model_args.filename,
            self.training_args.filename,
        ]
        return "_".join(filename_parts)

    @property
    def rl_param_sizes(self) -> dict[str, dict[str, tuple[int, int]]]:
        return {
            "dataset_args": self.dataset_args.rl_param_sizes,
            "trajectory_args": self.trajectory_args.rl_param_sizes,
            "model_args": self.model_args.rl_param_sizes,
            "training_args": self.training_args.rl_param_sizes,
        }

    def encode_rl_params(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "dataset_args": self.dataset_args.encode_rl_params(),
            "trajectory_args": self.trajectory_args.encode_rl_params(),
            "model_args": self.model_args.encode_rl_params(),
            "training_args": self.training_args.encode_rl_params(),
        }

    @classmethod
    def load_rl_params(cls, encoded: dict[str, torch.Tensor]) -> "SimulatorParams":
        return cls(
            dataset_args=DatasetArgs.load_rl_params(encoded["dataset_args"]),
            trajectory_args=TrajectoryArgs.load_rl_params(encoded["trajectory_args"]),
            model_args=ModelArgs.load_rl_params(encoded.get("model_args", {})),
            training_args=TrainingArgs.load_rl_params(encoded["training_args"]),
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
    masses: torch.Tensor


class ParameterLossError(ValueError):
    def __init__(self, message, loss):
        super().__init__(message)
        self.loss = loss
