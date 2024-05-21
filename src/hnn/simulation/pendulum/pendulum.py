from typing import overload
from multimethod import multidispatch
import torch


from hnn.dynamics import HamiltonianDynamics
from hnn.types import DatasetArgs, TrajectoryArgs, HamiltonianField


def get_default_y0(radius: float = 1.3) -> torch.Tensor:
    y0 = torch.rand(2) * 2.0 - 1
    y0 = y0 / torch.sqrt((y0**2).sum()) * radius
    return y0


def pendulum_fn(ps_coords: torch.Tensor) -> torch.Tensor:
    """
    Pendulum Hamiltonian
    """
    r, v = torch.tensor_split(ps_coords, 2)
    H = 3 * (1 - torch.cos(r)) + v**2
    return H


class PendulumHamiltonianDynamics(HamiltonianDynamics):
    def __init__(self):
        y0 = get_default_y0()
        super(PendulumHamiltonianDynamics, self).__init__(
            pendulum_fn, y0=y0, t_span=(0, 10)
        )
