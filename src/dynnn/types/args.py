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

from dynnn.utils import round_to_mantissa

from .enums import GeneratorType, OdeSolverType, VectorField
from .types import ForcedInt, ForcedIntOrNone

MAX_N_BODIES = 1000
MIN_N_BODIES = 2


def RlParam(attr):
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
            if (field.json_schema_extra or {}).get("decorator") == RlParam
        }

    @property
    def rl_param_sizes(self) -> dict[str, tuple[int, int]]:
        return {
            # TODO: metadata=[Ge(ge=0.1), Le(le=0.9)]
            field_name: (
                field.metadata[0].__getstate__()[0],  # extracts 0.1 from Ge(ge=0.1)
                field.metadata[1].__getstate__()[0],
            )
            for field_name, field in self.model_fields.items()
            if (field.json_schema_extra or {}).get("decorator") == RlParam
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

    n_epochs: ForcedInt = Field(5, ge=50, le=200, decorator=RlParam)
    steps_per_epoch: int = Field(200, ge=100, le=2000)

    learning_rate: float = Field(1e-3, ge=1e-7, le=1e-1)
    weight_decay: float = Field(1e-4, ge=1e-7, le=1e-1)

    ### early stopping parameters ###
    # minimum improvement in loss to avoid incrementing early stopping counter
    tolerance: float = 1e-1

    # number of insufficiently improving epochs before stopping
    patience: ForcedInt = Field(10, ge=2, le=100, decorator=RlParam)

    # min epochs before considering early stopping
    min_epochs: int = 5

    ### end early stopping parameters ###


class PinnModelArgs(HasSimulatorArgs):
    """
    Arguments for the PINN model
    """

    domain_min: ForcedInt = Field(0, decorator=RlParam, ge=-1000, le=0)
    domain_max: ForcedInt = Field(10, decorator=RlParam, ge=1, le=1000)

    # type of vector field to attempt to learn
    vector_field_type: VectorField = VectorField.PORT

    # canonical neural network dims (cannot be RlParams)
    canonical_input_dim: int = Field(128, ge=64, le=4096)
    canonical_hidden_dim: int = Field(512, ge=64, le=4096)

    # use invariant layers to improve learning rate
    use_invariant_layer: bool = True

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

    # number of distinct trajectories to generate
    n_samples: ForcedInt = Field(5, decorator=RlParam, ge=5, le=25)

    test_split: float = Field(0.8, ge=0.1, le=0.9)


class TrajectoryArgs(HasSimulatorArgs):
    """
    Arguments for generating a trajectory of a dynamical system with an equation of motion
    """

    # initial conditions
    y0: torch.Tensor | None = None
    masses: torch.Tensor | None = None

    # model to use, if any
    model: torch.nn.Module | None = None

    n_bodies: ForcedIntOrNone = Field(
        15, decorator=RlParam, ge=MIN_N_BODIES, le=MAX_N_BODIES
    )
    n_dims: ForcedInt = Field(3, ge=1, le=6)

    # type of EOM generator
    generator_type: GeneratorType = GeneratorType.HAMILTONIAN

    # time parameters
    time_scale: ForcedInt = Field(10, decorator=RlParam, ge=5, le=100)
    t_span_min: ForcedInt = Field(0, ge=0, le=3)  # decorator=RlParam
    t_span_max: ForcedInt = Field(50, decorator=RlParam, ge=5, le=500)

    # ODE solver parameters
    odeint_rtol: float = Field(1e-10, ge=1e-12, le=1e-5, decorator=RlParam)
    odeint_atol: float = Field(1e-6, ge=1e-12, le=1e-5, decorator=RlParam)
    odeint_solver: OdeSolverType = OdeSolverType.TSIT5

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

        values.odeint_rtol = round_to_mantissa(values.odeint_rtol)
        values.odeint_atol = round_to_mantissa(values.odeint_atol)

        return values
