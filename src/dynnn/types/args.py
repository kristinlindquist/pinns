from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from pydantic.functional_validators import BeforeValidator
from pydash import merge
import torch
import torch.nn.functional as F
from typing import Any, Callable, Literal

from dynnn.encoding import encode_params, flatten_dict, unflatten_params
from dynnn.utils import round_to_mantissa

from .enums import GeneratorType, OdeSolverType, VectorField, enum_validator
from .stats import ModelStats
from .types import Dataset, ForcedInt, ForcedIntOrNone

MAX_N_BODIES = 200  # TODO: 1000
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
    def load_encoded(cls, encoded: dict, existing_args: "HasSimulatorArgs"):
        return cls(**{**encoded, **existing_args.model_dump()})

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


class TrainingArgs(HasSimulatorArgs):
    """
    Arguments for training a model
    """

    n_epochs: ForcedInt = Field(5, ge=5, le=200, decorator=RlParam)
    steps_per_epoch: int = Field(400, ge=100, le=2000)

    learning_rate: float = Field(1e-3, ge=1e-7, le=1e-1, decorator=RlParam)
    weight_decay: float = Field(1e-4, ge=1e-7, le=1e-1, decorator=RlParam)

    plot_loss_callback: Callable | None = None


class EarlyStoppingTrainingArgs(TrainingArgs):
    """
    Arguments for training a model
    """

    # minimum improvement in loss to avoid incrementing early stopping counter
    tolerance: float = 1e-1

    # number of insufficiently improving epochs before stopping
    patience: ForcedInt = Field(10, ge=2, le=100, decorator=RlParam)

    # min epochs before considering early stopping
    min_epochs: int = 5


class MechanicsArgs(HasSimulatorArgs):
    domain_min: ForcedInt = Field(0, decorator=RlParam, ge=-1000, le=0)
    domain_max: ForcedInt = Field(10, decorator=RlParam, ge=1, le=1000)

    @model_validator(mode="after")
    def pre_update(cls, values: "MechanicsArgs") -> "MechanicsArgs":
        if values.domain_max <= values.domain_min:
            values.domain_max = (
                values.domain_max + (values.domain_min - values.domain_max) + 1
            )

        return values


class PinnTrainingArgs(EarlyStoppingTrainingArgs):
    """
    Arguments for training a PINN model
    """

    loss_fn: Callable | None = None

    # type of vector field to attempt to learn
    # TODO: no PORT
    vector_field_type: VectorField = Field(
        VectorField.NONE,
        ge=min(VectorField),
        le=max(VectorField),
        # decorator=RlParam,
    )

    # use invariant layers to improve learning rate
    use_invariant_layer: bool = False


class SimulatorTrainingArgs(TrainingArgs):
    num_experiments: ForcedInt = Field(
        50, ge=25, le=200, description="number of RL param switch experiments"
    )
    max_simulator_steps: ForcedInt = Field(
        250, ge=100, le=2000, description="max steps within an experiment"
    )


class PinnModelArgs(HasSimulatorArgs):
    """
    Arguments for the PINN model
    """

    canonical_input_dim: int = Field(128, ge=64, le=4096)
    canonical_hidden_dim: int = Field(512, ge=64, le=4096)
    n_dims: ForcedInt = Field(3, ge=1, le=6)


class DatasetArgs(HasSimulatorArgs):
    """
    Arguments for generating a dataset of trajectories of a dynamical system
    """

    # number of distinct trajectories to generate
    n_samples: ForcedInt = Field(2, decorator=RlParam, ge=2, le=25)

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
        5, decorator=RlParam, ge=MIN_N_BODIES, le=MAX_N_BODIES
    )
    n_dims: ForcedInt = Field(3, ge=1, le=6)

    # type of EOM generator
    generator_type: GeneratorType = Field(
        GeneratorType.HAMILTONIAN,
        ge=min(GeneratorType),
        le=max(GeneratorType),
        # decorator=RlParam,
    )

    # time parameters
    time_scale: ForcedInt = Field(2, decorator=RlParam, ge=1, le=50)
    t_span_min: ForcedInt = Field(0, ge=0, le=3)  # decorator=RlParam
    t_span_max: ForcedInt = Field(10, decorator=RlParam, ge=5, le=500)

    # ODE solver parameters
    odeint_rtol: float = Field(1e-7, ge=1e-14, le=1e-5, decorator=RlParam)
    odeint_atol: float = Field(1e-7, ge=1e-14, le=1e-5, decorator=RlParam)
    odeint_solver: OdeSolverType = Field(
        OdeSolverType.SYMPLECTIC,
        ge=min(OdeSolverType),
        le=max(OdeSolverType),
        decorator=RlParam,
    )

    """
    A symplectic ODE solver of order p means that the global error between
    the numerical and true solution is ~ to p-th power of the step size.
    i.e. if step size is reduced by a factor of h, the global error will decrease by a factor of ~h^p
    """
    odeint_order: ForcedInt = Field(2, ge=1, le=4)  # decorator=RlParam

    @model_validator(mode="before")
    def validate_enums(cls, values: dict[str, Any]) -> dict[str, Any]:
        """
        Validate enum fields (e.g. turn int values into enums)
        """
        for name, field in cls.__fields__.items():
            if name in values:
                values[name] = enum_validator(cls, values[name], field)
        return values

    @model_validator(mode="after")
    def post_update(cls, values: "TrajectoryArgs") -> "TrajectoryArgs":
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


class SimulatorArgs(BaseModel):
    dataset_args: DatasetArgs = DatasetArgs()
    mechanics_args: MechanicsArgs = MechanicsArgs()
    trajectory_args: TrajectoryArgs = TrajectoryArgs()
    training_args: PinnTrainingArgs = PinnTrainingArgs()

    @property
    def filename(self) -> str:
        filename_parts = [
            self.dataset_args.filename,
            self.mechanics_args.filename,
            self.trajectory_args.filename,
            self.training_args.filename,
        ]
        return "_".join(filename_parts)

    @property
    def rl_param_sizes(self) -> dict[str, dict[str, tuple[int, int]]]:
        return {
            "dataset_args": self.dataset_args.rl_param_sizes,
            "mechanics_args": self.mechanics_args.rl_param_sizes,
            "trajectory_args": self.trajectory_args.rl_param_sizes,
            "training_args": self.training_args.rl_param_sizes,
        }

    def encode_rl_params(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "dataset_args": self.dataset_args.encode_rl_params(),
            "mechanics_args": self.mechanics_args.encode_rl_params(),
            "trajectory_args": self.trajectory_args.encode_rl_params(),
            "training_args": self.training_args.encode_rl_params(),
        }

    @classmethod
    def load_encoded(
        cls, encoded: dict[str, torch.Tensor], existing_args: "SimulatorArgs"
    ) -> "SimulatorArgs":
        return cls(
            **{
                f: v.annotation.load_encoded(encoded[f], getattr(existing_args, f))
                for f, v in cls.__fields__.items()
            }
        )


class SimulatorState(BaseModel):
    params: SimulatorArgs
    stats: ModelStats = ModelStats()
    sim_duration: float = 0.0

    @property
    def rl_param_sizes(self) -> dict[str, dict[str, Any]]:
        return {
            "params": self.params.rl_param_sizes,
            "stats": 2,
            "sim_duration": 2,
        }

    @property
    def rl_param_sizes_flat(self) -> dict[str, tuple[int, int]]:
        return flatten_dict(self.rl_param_sizes)

    @property
    def num_rl_params(self) -> int:
        return len(self.rl_param_sizes_flat)

    def encode_rl_params(self) -> tuple[torch.Tensor, dict]:
        attributes = {
            "params": self.params.encode_rl_params(),
            "stats": self.stats.encode(),
            "sim_duration": self.sim_duration,
        }

        return encode_params(flatten_dict(attributes)), attributes

    @classmethod
    def load_encoded(
        cls, encoded_state: dict, existing_state: "SimulatorState", template: dict
    ) -> "SimulatorState":
        """
        Includes existing state because not all params are rl params (and we don't want to lose those as defaults)
        TODO: this is confusing.
        """
        scalar_dict = unflatten_params(encoded_state, template, decode_tensors=True)
        return cls(
            params=SimulatorArgs.load_encoded(
                scalar_dict["params"], existing_state.params
            ),
            stats=scalar_dict.get("stats", {}),
            sim_duration=scalar_dict.get("sim_duration", 0.0),
        )
