from typing import overload
from multimethod import multidispatch
import torch

from hnn.dynamics import HamiltonianDynamics
from hnn.types import DatasetArgs, TrajectoryArgs, HamiltonianField


def get_initial_conditions(
    n_bodies: int,
    n_dims: int = 2,
    width: int = 100,
    height: int = 100,
    temp: float = 75.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate initial conditions for a system of particles.
    """
    # initialize masses
    masses = torch.ones(n_bodies)

    # initialize positions
    q = torch.rand(n_bodies, n_dims) * torch.tensor([width, height])

    # initialize velocities (Maxwell-Boltzmann distribution scaled by temp)
    v = torch.randn(n_bodies, n_dims) * torch.sqrt(torch.tensor([temp]))

    # Ensure zero total momentum
    total_momentum = v.sum(0)
    p = v - (total_momentum / n_bodies)

    coords = torch.stack([q, p], dim=1)  # n_bodies x 2 x n_dims

    return coords, masses


DEFAULT_TRAJECTORY_ARGS = {"t_span": (0, 10)}


def calc_lennard_jones_potential(
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
    return_dim = len(positions.shape) - 2

    v = 0.0
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            r = torch.linalg.vector_norm(positions[i] - positions[j], dim=return_dim)
            r12 = (σ / r) ** 12
            r6 = (σ / r) ** 6
            v += 4 * ε * (r12 - r6)
    return v


def calc_kinetic_energy(velocities: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
    """
    Compute the kinetic energy of a system.
    = 1/2 mv^2

    Args:
        velocities (torch.Tensor): Tensor containing xy velocities of particles
            size:
                n_particles x 2 or
                n_particles x timepoints x 2
        masses (torch.Tensor): Tensor containing masses of particles
    """
    # norm is sqrt(x^2 + y^2)
    scalar_velocities = torch.norm(velocities, dim=-1)

    return_dim = 0 if len(scalar_velocities.shape) > 1 else None
    for i in range(len(scalar_velocities.shape) - 1):
        masses = masses.unsqueeze(-1)

    kinetics = scalar_velocities**2 / (2 * masses)
    return torch.sum(kinetics, dim=return_dim)


def mve_ensemble_fn(
    coords: torch.Tensor,
    masses: torch.Tensor,
    potential_fn=calc_lennard_jones_potential,
) -> torch.Tensor:
    """
    Hamiltonian for a generalized MVE ensemble.

    Args:
        coords (torch.Tensor): Coordinates (positions and momenta) (n_particles x 2 x n_dims)
        masses (torch.Tensor): Masses of each particle (n_particles)
        potential_fn (callable): Function that computes the potential energy given positions

    Returns:
        torch.Tensor: Hamiltonian (Total energy) of the system.
    """
    # Split coordinates into positions and velocity (num_particles x num_dims)
    q, p = [v.squeeze() for v in torch.split(coords, 1, dim=1)]

    # Compute kinetic energy
    kinetic_energy = calc_kinetic_energy(p, masses)

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
            masses, potential_fn=calc_lennard_jones_potential
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
