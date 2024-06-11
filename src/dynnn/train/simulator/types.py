from typing import Any
import torch
from pydantic import BaseModel

from dynnn.encoding import encode_params, flatten_dict, unflatten_params
from dynnn.types import (
    PinnStats,
    SimulatorParams,
)


class SimulatorState(BaseModel):
    params: SimulatorParams
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
    def load_rl_params(cls, encoded: dict, template: dict) -> "SimulatorParams":
        scalar_dict = unflatten_params(encoded, template, decode_tensors=True)
        return cls(
            params=SimulatorParams.load_rl_params(scalar_dict["params"]),
            stats=scalar_dict.get("stats", {}),
            sim_duration=scalar_dict.get("sim_duration", 0.0),
        )
