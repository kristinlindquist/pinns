from typing import overload
from multimethod import multidispatch
import torch

from hnn.dynamics import HamiltonianDynamics
from hnn.types import DatasetArgs, TrajectoryArgs, HamiltonianField


def get_default_y0() -> torch.Tensor:
    return torch.rand(2)


DEFAULT_TRAJECTORY_ARGS = {"t_span": (0, 10), "y0": get_default_y0()}


def lennard_jones_potential(
    q: torch.Tensor, σ: float = 1.0, ε: float = 1.0
) -> torch.Tensor:
    """
    Lennard-Jones potential function.
    V(r) = 4 * ε * ((σ / r)^12 - (σ / r)^6

    Args:
        q (torch.Tensor): Tensor containing positions of particles.
        σ (float): "the distance at which the particle-particle potential energy V is zero"
        ε (float): "depth of the potential well"

    Returns:
        torch.Tensor: Total potential energy (v).

    See https://en.wikipedia.org/wiki/Lennard-Jones_potential
    """
    n_particles = q.shape[0] // 2
    q = q.view(n_particles, -1)

    v = 0.0
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            r = torch.norm(q[i] - q[j])
            r6 = (σ / r) ** 6
            r12 = r6 * r6
            v += 4 * ε * (r12 - r6)
    return v


def mve_ensemble_fn(
    coords: torch.Tensor, masses: torch.Tensor, potential_fn=lennard_jones_potential
) -> torch.Tensor:
    """
    Hamiltonian for a generalized MVE ensemble.

    Args:
        coords (torch.Tensor): Tensor containing the coordinates (positions and momenta).
        masses (torch.Tensor): Tensor containing the masses of each particle.
        potential_fn (callable): Function that computes the potential energy given positions.

    Returns:
        torch.Tensor: Hamiltonian (Total energy) of the system.
    """
    q, p = torch.tensor_split(coords, 2)

    # Compute kinetic energy
    kinetic_energy = torch.sum(p**2 / (2 * masses.repeat(2)))

    # Compute potential energy
    potential_energy = potential_fn(q)

    # Hamiltonian (Total Energy)
    H = kinetic_energy + potential_energy
    return H


def get_mve_ensemble_fn(masses: torch.Tensor, potential_fn):
    def _mve_ensemble_fn(coords: torch.Tensor) -> torch.Tensor:
        return mve_ensemble_fn(coords, masses, potential_fn)

    return _mve_ensemble_fn


class MveEnsembleHamiltonianDynamics(HamiltonianDynamics):
    def __init__(self):
        masses = torch.tensor([1.0, 1.5, 2.0])
        mve_ensemble_fn = get_mve_ensemble_fn(
            masses, potential_fn=lennard_jones_potential
        )
        super(MveEnsembleHamiltonianDynamics, self).__init__(mve_ensemble_fn)

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
