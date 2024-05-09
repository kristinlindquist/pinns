from typing import overload
from multimethod import multidispatch
import torch


from hnn.dynamics import HamiltonianDynamics
from hnn.types import DatasetArgs, TrajectoryArgs, HamiltonianField


def get_default_y0(radius: float = 1.3) -> torch.Tensor:
    y0 = torch.rand(2) * 2.0 - 1
    y0 = y0 / torch.sqrt((y0**2).sum()) * radius
    return y0


DEFAULT_TRAJECTORY_ARGS = {"t_span": (0, 10), "y0": get_default_y0()}


def pendulum_fn(coords: torch.Tensor) -> torch.Tensor:
    """
    Pendulum Hamiltonian
    """
    q, p = torch.tensor_split(coords, 2)
    H = 3 * (1 - torch.cos(q)) + p**2
    return H


class PendulumHamiltonianDynamics(HamiltonianDynamics):
    def __init__(self):
        super(PendulumHamiltonianDynamics, self).__init__(pendulum_fn)

    @overload
    @HamiltonianDynamics.get_dataset.register
    def _(self, args: dict, trajectory_args: dict):
        traj_args = TrajectoryArgs(**{**DEFAULT_TRAJECTORY_ARGS, **args})
        return self.get_dataset(DatasetArgs(**args), traj_args)

    @overload
    @HamiltonianDynamics.get_trajectory.register
    def _(self, args: dict, ode_args: dict = {}) -> HamiltonianField:
        traj_args = TrajectoryArgs(**{**DEFAULT_TRAJECTORY_ARGS, **args})
        return self.get_trajectory(traj_args, ode_args)
