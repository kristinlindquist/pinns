from typing import overload, Callable
from multimethod import multidispatch
import torch
import torch.nn.functional as F
from functools import partial

from dynnn.mechanics import Mechanics
from dynnn.types import ModelArgs


def get_initial_conditions(
    n_bodies: int,
    n_dims: int,
    width: int = 4,
    height: int = 4,
    depth: int = 4,
    temp: float = 5.0,
    offset: int = 2,
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
    masses = torch.ones(n_bodies).requires_grad_()

    # initialize positions
    possible_dims = [width, height, depth]
    r = (torch.rand(n_bodies, n_dims) * torch.tensor(possible_dims[:n_dims])) + offset

    # initialize velocities (simplified Maxwell-Boltzmann distribution scaled by temp)
    v = torch.randn(n_bodies, n_dims) * torch.sqrt(torch.tensor([temp]))

    # Ensure zero total momentum
    total_momentum = v.sum(0)
    v -= total_momentum / n_bodies

    ps_coords = torch.stack([r, v], dim=1).requires_grad_()  # n_bodies x 2 x n_dims

    return ps_coords, masses


def calc_boundary_potential(
    positions: torch.Tensor,
    boundaries: tuple[float, float],
    steepness: float = 100.0,
    width: float = 0.05,
) -> torch.Tensor:
    """
    A conservative boundary potential that fades out smoothly.
    NOTE: Energy is not perfectly conserved around this boundary.

    Args:
        positions (torch.Tensor): Particle position coordinates (n_bodies x n_dims).
        boundaries (tuple[float, float]): Boundary positions (min and max) for each dimension.
        steepness (float): Steepness of the boundary potential.
        width (float): Width of the boundary region where the potential is applied.

    Returns:
        torch.Tensor: Total boundary potential for the given positions.
    """
    if len(positions.shape) != 2:
        raise ValueError("Positions must be n_bodies x n_dims")

    # Smoothly varying boundary potential function
    def smooth_boundary_potential(distance, width, steepness):
        return steepness * torch.exp(-((distance / width) ** 2))

    # Calculate distances to the minimum and maximum boundaries
    distance_min = torch.abs(positions - boundaries[0])
    distance_max = torch.abs(positions - boundaries[1])

    # Apply smooth boundary potential
    boundary_potential_min = smooth_boundary_potential(distance_min, width, steepness)
    boundary_potential_max = smooth_boundary_potential(distance_max, width, steepness)

    # Sum the boundary potentials across all dimensions and particles
    total_boundary_effect = boundary_potential_min.sum() + boundary_potential_max.sum()

    return total_boundary_effect


def calc_lennard_jones_potential(
    positions: torch.Tensor, σ: float = 1.0, ε: float = 1.0
) -> torch.Tensor:
    """
    Lennard-Jones potential function.
    V(r) = 4 * ε * ((σ / r)^12 - (σ / r)^6

    Args:
        positions (torch.Tensor): Tensor containing positions of particles
            size:
                n_bodies x n_dims or
                timepoints x n_bodies x n_dims
        σ (float): "the distance at which the particle-particle potential energy V is zero"
        ε (float): "depth of the potential well"

    Returns:
        torch.Tensor: Total potential energy (v).

    See https://en.wikipedia.org/wiki/Lennard-Jones_potential
    """
    distances = torch.cdist(positions, positions)
    lj = 4 * ε * ((σ / distances) ** 12 - (σ / distances) ** 6)
    lj.nan_to_num_(0.0)

    return lj.sum((-1, -2)) / 2


def calc_kinetic_energy(velocities: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
    """
    Compute the kinetic energy of a system.
    = 1/2 mv^2

    Args:
        velocities (torch.Tensor): Tensor containing xy velocities of particles
            size:
                n_bodies x num_dims or
                timepoints x n_bodies x num_dims
        masses (torch.Tensor): Tensor containing masses of particles
    """
    if len(masses.shape) == 1:
        # Reshape from (n_bodies,) to (n_bodies, 1)
        masses = masses.unsqueeze(-1)

    # Ensure masses can broadcast correctly with velocities
    if len(velocities.shape) >= 3:
        masses = masses.reshape(
            *([1] * (len(velocities.shape) - 2)), masses.shape[0], 1
        )

    kinetic_energy = torch.sum(0.5 * masses * velocities**2, dim=-1)

    # Average the kinetic energies over all particles
    if kinetic_energy.dim() > 1:
        # If we have timepoints, average across all particles for each timepoint
        return kinetic_energy.mean(dim=-1)

    return kinetic_energy.mean()


def calc_total_energy(
    r: torch.Tensor,
    v: torch.Tensor,
    masses: torch.Tensor,
    potential_fn=calc_lennard_jones_potential,
):
    """
    Compute the total energy of a system.
    """
    kinetic_energy = calc_kinetic_energy(v, masses)
    potential_energy = potential_fn(r)
    return kinetic_energy + potential_energy


def energy_conservation_loss(
    ps_coords: torch.Tensor,
    ps_coords_hat: torch.Tensor,
    masses: torch.Tensor,
    potential_fn=calc_lennard_jones_potential,
) -> torch.Tensor:
    """
    Compute loss for actual versus predicted total energy.

    TODO: calc actuals only once
    """
    r, v = [s.squeeze() for s in torch.split(ps_coords, 1, dim=-2)]
    r_hat, v_hat = [s.squeeze() for s in torch.split(ps_coords_hat, 1, dim=-2)]
    energy = calc_total_energy(r, v, masses, potential_fn)
    energy_hat = calc_total_energy(r_hat, v_hat, masses, potential_fn)
    return torch.pow(energy - energy_hat, 2)


def mve_ensemble_h_fn(
    ps_coords: torch.Tensor,
    masses: torch.Tensor,
    potential_fn=calc_lennard_jones_potential,
) -> torch.Tensor:
    """
    Hamiltonian for a generalized MVE ensemble.

    Args:
        ps_cooords (torch.Tensor): Particle phase space coordinates (n_bodies x 2 x n_dims)
        masses (torch.Tensor): Masses of each particle (n_bodies)
        potential_fn (callable): Function that computes the potential energy given positions

    Returns:
        torch.Tensor: Hamiltonian (Total energy) of the system.
    """
    r, v = [s.squeeze() for s in torch.split(ps_coords, 1, dim=1)]
    return calc_total_energy(r, v, masses, potential_fn)


def mve_ensemble_l_fn(
    r: torch.Tensor,
    v: torch.Tensor,
    masses: torch.Tensor,
    potential_fn=calc_lennard_jones_potential,
) -> torch.Tensor:
    """
    Lagrangian for a generalized MVE ensemble.

    Args:
        r (torch.Tensor): Particle position coordinates (n_bodies x 2 x n_dims)
        v (torch.Tensor): Velocities of each particle (n_bodies x n_dims)
        masses (torch.Tensor): Masses of each particle (n_bodies)
        potential_fn (callable): Function that computes the potential energy given positions
    """
    kinetic_energy = calc_kinetic_energy(v, masses)
    potential_energy = potential_fn(r)

    return kinetic_energy - potential_energy


class MveEnsembleMechanics(Mechanics):
    """
    Mechanics for an MVE ensemble.
    """

    def __init__(self, args: ModelArgs = ModelArgs()):
        # potential energy function
        # - Lennard-Jones potential
        # - Boundary potential
        def potential_fn(positions: torch.Tensor):
            bc_pe = calc_boundary_potential(positions, boundaries=args.domain)
            return calc_lennard_jones_potential(positions) + bc_pe

        self.potential_fn = potential_fn
        self.no_bc_potential_fn = partial(calc_lennard_jones_potential)

        _get_function = lambda masses: partial(
            (
                mve_ensemble_l_fn
                if args.system_type == "lagrangian"
                else mve_ensemble_h_fn
            ),
            masses=masses,
            potential_fn=self.potential_fn,
        )

        super(MveEnsembleMechanics, self).__init__(
            _get_function,
            domain=args.domain,
            t_span=args.t_span,
            system_type=args.system_type,
        )
