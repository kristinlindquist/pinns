from typing import overload
from multimethod import multidispatch
import torch

from hnn.dynamics import HamiltonianDynamics
from hnn.types import DatasetArgs, TrajectoryArgs, HamiltonianField


def get_initial_conditions(
    n_bodies: int = 10,
    n_dims: int = 2,
    width: int = 100,
    height: int = 100,
    temp: float = 75.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # initialize positions
    q = torch.rand(n_bodies, n_dims) * torch.tensor([width, height])

    # initialize velocities (Maxwell-Boltzmann distribution scaled by temp)
    p = torch.randn(n_bodies, n_dims) * torch.sqrt(torch.tensor([temp]))

    # Ensure zero total momentum
    total_momentum = p.sum(0)
    p -= total_momentum / n_bodies  # TODO: this is velocity, not momentum!

    coords = torch.stack([q, p], dim=1)  # n_bodies x len([q, p]) x n_dims

    masses = torch.ones(n_bodies)

    return coords, masses


DEFAULT_TRAJECTORY_ARGS = {"t_span": (0, 7)}


def lennard_jones_potential(
    positions: torch.Tensor, σ: float = 1.0, ε: float = 1.0
) -> torch.Tensor:
    """
    Lennard-Jones potential function.
    V(r) = 4 * ε * ((σ / r)^12 - (σ / r)^6

    Args:
        positions (torch.Tensor): Tensor containing positions of particles
            size:
                n_particles x 2 or
                n_particles x timepoints x 2
        σ (float): "the distance at which the particle-particle potential energy V is zero"
        ε (float): "depth of the potential well"

    Returns:
        torch.Tensor: Total potential energy (v).

    See https://en.wikipedia.org/wiki/Lennard-Jones_potential
    """
    n_particles = positions.shape[0]
    norm_dim = len(positions.shape) - 2

    v = 0.0
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            r = torch.linalg.vector_norm(positions[i] - positions[j], dim=norm_dim)
            r12 = (σ / r) ** 12
            r6 = (σ / r) ** 6
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
    # Split coordinates into positions and momenta (num_particles x num_dims)
    q, p = [v.squeeze() for v in torch.split(coords, 1, dim=1)]

    # Compute kinetic energy
    kinetic_energy = torch.sum(p**2 / (2 * masses.unsqueeze(1)))

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
    def __init__(self, n_bodies: int = 5):
        y0, masses = get_initial_conditions(n_bodies)
        self.y0 = y0
        mve_ensemble_fn = get_mve_ensemble_fn(
            masses, potential_fn=lennard_jones_potential
        )
        super(MveEnsembleHamiltonianDynamics, self).__init__(mve_ensemble_fn)

    @overload
    @HamiltonianDynamics.get_dataset.register
    def _(self, args: dict, trajectory_args: dict):
        traj_args = TrajectoryArgs(**{**DEFAULT_TRAJECTORY_ARGS, **trajectory_args})
        return self.get_dataset(DatasetArgs(**args), traj_args, {"y0": self.y0})

    @overload
    @HamiltonianDynamics.get_trajectory.register
    def _(self, args: dict, ode_args: dict = {}) -> HamiltonianField:
        traj_args = TrajectoryArgs(**{**DEFAULT_TRAJECTORY_ARGS, **args})
        return self.get_trajectory(traj_args, {"y0": self.y0, **ode_args})
