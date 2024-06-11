from typing import Any
import torch
from pydantic import BaseModel

from dynnn.encoding import encode_params, flatten_dict, unflatten_params
from dynnn.types import (
    DatasetArgs,
    PinnModelArgs,
    PinnStats,
    TrajectoryArgs,
    PinnTrainingArgs,
)


class SimulatorArgs(BaseModel):
    dataset_args: DatasetArgs = DatasetArgs()
    trajectory_args: TrajectoryArgs = TrajectoryArgs()
    model_args: PinnModelArgs = PinnModelArgs()
    training_args: PinnTrainingArgs = PinnTrainingArgs()

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
    def load_rl_params(cls, encoded: dict[str, torch.Tensor]) -> "SimulatorArgs":
        return cls(
            dataset_args=DatasetArgs.load_rl_params(encoded["dataset_args"]),
            trajectory_args=TrajectoryArgs.load_rl_params(encoded["trajectory_args"]),
            model_args=PinnModelArgs.load_rl_params(encoded.get("model_args", {})),
            training_args=PinnTrainingArgs.load_rl_params(encoded["training_args"]),
        )


class SimulatorState(BaseModel):
    params: SimulatorArgs
    stats: PinnStats = PinnStats()
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
    def load_rl_params(cls, encoded: dict, template: dict) -> "SimulatorArgs":
        scalar_dict = unflatten_params(encoded, template, decode_tensors=True)
        return cls(
            params=SimulatorArgs.load_rl_params(scalar_dict["params"]),
            stats=scalar_dict.get("stats", {}),
            sim_duration=scalar_dict.get("sim_duration", 0.0),
        )
