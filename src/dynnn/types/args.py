from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from pydantic.functional_validators import BeforeValidator
import torch
import torch.nn.functional as F
from typing import Any, Literal


from .enums import GeneratorType, OdeSolver
from .types import ForcedInt, ForcedIntOrNone

MAX_N_BODIES = 50
MIN_N_BODIES = 2


def rl_param(attr):
    attr.metadata["is_rl_trainable"] = True


class HasSimulatorArgs(BaseModel):
    """
    Base class for models that have rl-learnable simulator parameters
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

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


class PinnTrainingArgs(HasSimulatorArgs):
    """
    Arguments for training a PINN model
    """

    n_epochs: ForcedInt = Field(5, ge=50, le=200)  # decorator=rl_param
    learning_rate: float = Field(1e-3, ge=1e-6, le=1e-1)  # rl_param
    weight_decay: float = Field(1e-4, ge=1e-6, le=1e-1)  # rl_param
    patience: ForcedInt = Field(10, ge=2, le=100, decorator=rl_param)

    tolerance: float = 1e-1
    steps_per_epoch: int = Field(10, ge=10, le=10000)
    min_epochs: int = 5


class PinnModelArgs(HasSimulatorArgs):
    """
    Arguments for the PINN model
    """

    domain_min: ForcedInt = Field(0, decorator=rl_param, ge=-1000, le=1000)
    domain_max: ForcedInt = Field(10, decorator=rl_param, ge=-1000, le=1000)
    vector_field_type: Literal["solenoidal", "conservative", "port", "both"] = (
        "conservative"
    )
    hidden_dim: ForcedInt = Field(500, decorator=rl_param, ge=8, le=4096)
    use_invariant_layer: bool = False

    @model_validator(mode="after")
    def pre_update(cls, values: "PinnModelArgs") -> "PinnModelArgs":
        if values.domain_max <= values.domain_min:
            values.domain_max = (
                values.domain_max + (values.domain_min - values.domain_max) + 1
            )

        return values


class DatasetArgs(HasSimulatorArgs):
    """
    Arguments for generating a dataset of trajectories of a dynamical system
    """

    n_samples: ForcedInt = Field(2, decorator=rl_param, ge=2, le=5)
    test_split: float = Field(0.8, ge=0.1, le=0.9)


class TrajectoryArgs(HasSimulatorArgs):
    """
    Arguments for generating a trajectory of a dynamical system with an equation of motion
    """

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
