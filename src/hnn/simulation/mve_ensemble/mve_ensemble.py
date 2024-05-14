from typing import overload
from multimethod import multidispatch
import torch

from hnn.dynamics import HamiltonianDynamics
from hnn.types import DatasetArgs, TrajectoryArgs, HamiltonianField


def get_initial_conditions(
    n_bodies: int,
    n_dims: int = 3,
    width: int = 5,
    height: int = 5,
    depth: int = 5,
    temp: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate initial conditions for a system of particles.

    Args:
        n_bodies (int): Number of particles
        n_dims (int): Number of dimensions
        width (int): Width of the space
        height (int): Height of the space
        temp (float): Temperature of the system
    """
    # initialize masses
    masses = torch.ones(n_bodies)

    # initialize positions
    possible_dims = [width, height, depth]
    r = torch.rand(n_bodies, n_dims) * torch.tensor(possible_dims[:n_dims])

    # initialize velocities (simplified Maxwell-Boltzmann distribution scaled by temp)
    v = torch.randn(n_bodies, n_dims) * torch.sqrt(torch.tensor([temp]))

    # Ensure zero total momentum
    total_momentum = v.sum(0)
    v -= total_momentum / n_bodies

    state = torch.stack([r, v], dim=1)  # n_bodies x 2 x n_dims

    return state, masses


def calc_lennard_jones_potential(
    coords: torch.Tensor, σ: float = 1.0, ε: float = 1.0
) -> torch.Tensor:
    """
    Lennard-Jones potential function.
    V(r) = 4 * ε * ((σ / r)^12 - (σ / r)^6

    Args:
        coords (torch.Tensor): Tensor containing positions of particles
            size:
                n_bodies x 2 or
                n_bodies x timepoints x 2
        σ (float): "the distance at which the particle-particle potential energy V is zero"
        ε (float): "depth of the potential well"

    Returns:
        torch.Tensor: Total potential energy (v).

    See https://en.wikipedia.org/wiki/Lennard-Jones_potential
    """
    n_bodies = coords.shape[0]
    return_dim = len(coords.shape) - 2

    v = 0.0
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            r = torch.linalg.vector_norm(coords[i] - coords[j], dim=return_dim)
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
                n_bodies x 2 or
                n_bodies x timepoints x 2
        masses (torch.Tensor): Tensor containing masses of particles
    """
    if len(masses.shape) == 1:
        # Reshape from (n_bodies,) to (n_bodies, 1)
        masses = masses.unsqueeze(-1)

    # Ensure masses can broadcast correctly with velocities
    if len(velocities.shape) == 3:
        # Adjust for timepoints: (n_bodies, 1) to (n_bodies, 1, 1)
        masses = masses.unsqueeze(1)

    kinetics = 0.5 * masses * velocities**2
    kinetic_energy = torch.sum(kinetics, dim=-1)

    # Average the kinetic energies over all particles
    if kinetic_energy.dim() > 1:
        # If we have timepoints, average across all particles for each timepoint
        return kinetic_energy.mean(dim=0)

    return kinetic_energy.mean()


def mve_ensemble_fn(
    state: torch.Tensor,
    masses: torch.Tensor,
    potential_fn=calc_lennard_jones_potential,
) -> torch.Tensor:
    """
    Hamiltonian for a generalized MVE ensemble.

    Args:
        state (torch.Tensor): State (positions and velocities) (n_bodies x 2 x n_dims)
        masses (torch.Tensor): Masses of each particle (n_bodies)
        potential_fn (callable): Function that computes the potential energy given positions

    Returns:
        torch.Tensor: Hamiltonian (Total energy) of the system.
    """
    # Split coordinates into positions and momentum (num_particles x num_dims)
    r, v = [s.squeeze() for s in torch.split(state, 1, dim=1)]

    # Compute kinetic energy
    kinetic_energy = calc_kinetic_energy(v, masses)

    # Compute potential energy
    potential_energy = potential_fn(r)

    # Hamiltonian (Total Energy)
    H = kinetic_energy + potential_energy

    return H


def get_mve_ensemble_fn(masses: torch.Tensor, potential_fn):
    def _mve_ensemble_fn(state: torch.Tensor) -> torch.Tensor:
        return mve_ensemble_fn(state, masses, potential_fn)

    return _mve_ensemble_fn


DEFAULT_TRAJECTORY_ARGS = {"t_span": (0, 10)}


class MveEnsembleHamiltonianDynamics(HamiltonianDynamics):
    def __init__(self, n_bodies: int = 5, dimensions: tuple[int, int] = (0, 50)):
        y0, masses = get_initial_conditions(n_bodies)
        self.y0 = y0
        mve_ensemble_fn = get_mve_ensemble_fn(
            masses, potential_fn=calc_lennard_jones_potential
        )
        super(MveEnsembleHamiltonianDynamics, self).__init__(
            mve_ensemble_fn, dimensions
        )

    @overload
    @HamiltonianDynamics.get_dataset.register
    def _(self, args: dict, trajectory_args: dict):
        traj_args = TrajectoryArgs(**{**DEFAULT_TRAJECTORY_ARGS, **trajectory_args})
        return self.get_dataset(DatasetArgs(**args), traj_args, {"y0": self.y0})

    @overload
    @HamiltonianDynamics.get_trajectory.register
    def _(
        self,
        args: dict,
        ode_args: dict = {},
    ) -> HamiltonianField:
        traj_args = TrajectoryArgs(**{**DEFAULT_TRAJECTORY_ARGS, **args})
        return self.get_trajectory(traj_args, {"y0": self.y0, **ode_args})
