"""
MVE Ensemble Mechanics
(subclass of `Mechanics` for an MVE ensemble)
"""

from functools import partial
import torch
import torch.nn.functional as F
from typing import Callable

from dynnn.mechanics import Mechanics
from dynnn.utils import zero_mask
from dynnn.types import GeneratorType, MechanicsArgs


def get_initial_conditions(
    n_bodies: int,
    n_dims: int = 3,
    width: int = 4,
    height: int = 4,
    depth: int = 4,
    temp: float = 5.0,
    offset: int = 2,
    masses: torch.Tensor | None = None,
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
    if masses is None:
        masses = torch.ones(n_bodies).requires_grad_()

    if masses.shape[0] != n_bodies:
        raise ValueError("Masses must be the same length as the number of bodies")

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
    steepness: float = 2000.0,
    width: float = 0.01,
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
    if len(masses.shape) < 2:
        masses = masses.unsqueeze(-1)

    masses = masses.expand(velocities.shape)

    # sum along n_bodies dimension
    kinetic_energy = torch.sum(0.5 * masses * velocities**2, dim=-1)

    # Average the kinetic energies over all particles
    # If we have timepoints, average across all particles for each timepoint
    mean_dim = -1 if velocities.dim() >= 2 else 0

    return kinetic_energy.mean(dim=mean_dim)


def calc_total_energy(
    q: torch.Tensor,
    v: torch.Tensor,
    masses: torch.Tensor,
    potential_fn=calc_lennard_jones_potential,
):
    """
    Compute the total energy of a system.

    Returns a tensor of shape (timepoints)
    """
    kinetic_energy = calc_kinetic_energy(v, masses)
    potential_energy = potential_fn(q)
    return kinetic_energy + potential_energy


def calc_total_energy_per_cell(
    q: torch.Tensor,
    p: torch.Tensor,
    masses: torch.Tensor,
    grid_resolution: tuple[int, int, int],
    boundaries: tuple[float, float] | tuple[float, float, float, float, float, float],
    potential_fn=calc_lennard_jones_potential,
) -> torch.Tensor:
    """
    Calculate the total energy of the system per cell in a grid
        (grid as specified by grid_resolution and boundaries).

    Args:
        q (tensor): The positions of the particles. shape: (n_samples, n_timepoints, n_bodies, n_dims)
        p (tensor): The momenta of the particles. shape: (n_samples, n_timepoints, n_bodies, n_dims)
        masses (tensor): The masses of the particles. shape: (n_bodies)
        grid_resolution (tuple): The number of cells in each dimension. shape: (3)
        boundaries (tuple): The boundaries of the grid
        potential_fn (function): The potential function to use. Default: calc_lennard_jones_potential

    Returns:
        tensor: The total energy of the system per cell. shape: (n_timepoints, n_cells)
    """
    if len(boundaries) == 2:
        boundaries = boundaries * 3

    # Calculate the size of each cell
    cell_sizes = torch.tensor(
        [
            (boundaries[1] - boundaries[0]) / grid_resolution[0],
            (boundaries[3] - boundaries[2]) / grid_resolution[1],
            (boundaries[5] - boundaries[4]) / grid_resolution[2],
        ]
    )

    # Create a grid of cell indices
    cell_indices = torch.cartesian_prod(*[torch.arange(gr) for gr in grid_resolution])

    # Calc position ranges for each cell (n_cells, 2, num_dims)
    cell_ranges = torch.stack(
        [
            torch.tensor(boundaries[0::2]) + cell_indices * cell_sizes,
            torch.tensor(boundaries[0::2]) + (cell_indices + 1) * cell_sizes,
        ],
        dim=-2,
    )

    # Create masks for each cell based on position ranges
    # -> shape: (n_timepoints, n_bodies, n_dims, n_cells)
    masks = (
        (p[..., None, 0] >= cell_ranges[..., 0, 0])
        & (p[..., None, 0] < cell_ranges[..., 1, 0])
        & (p[..., None, 1] >= cell_ranges[..., 0, 1])
        & (p[..., None, 1] < cell_ranges[..., 1, 1])
        & (p[..., None, 2] >= cell_ranges[..., 0, 2])
        & (p[..., None, 2] < cell_ranges[..., 1, 2])
    )
    masks = masks.permute(-1, *range(masks.dim() - 1))

    # Calculate the total energy of the system for each cell
    energies = [
        calc_total_energy(zero_mask(q, mask), zero_mask(p, mask), masses, potential_fn)
        for mask in masks
    ]

    # shape: (n_timepoints, n_cells)
    return torch.stack(energies).T


def energy_conservation_loss(
    ps_coords: torch.Tensor,
    ps_coords_hat: torch.Tensor,
    masses: torch.Tensor,
    potential_fn: Callable = calc_lennard_jones_potential,
    loss_weight: float = 100,
) -> torch.Tensor:
    """
    Compute total energy difference of a system over time. Should be zero.

    Args:
        ps_coords (torch.Tensor): Actual phase space coords (n_bodies x 2 x n_dims)
        ps_coords_hat (torch.Tensor): Predicted phase space coords (n_bodies x 2 x n_dims)
        masses (torch.Tensor): Masses of each particle (n_bodies)
        potential_fn (callable): Function that computes the potential energy given positions
        loss_weight (float): Weight to apply to the loss
    """
    q, v = [s.squeeze() for s in torch.split(ps_coords_hat, 1, dim=-2)]

    # Compute the total energy of the system
    energy = calc_total_energy(q, v, masses, potential_fn)

    # Compute the difference in energy between each timepoint
    energy_diff = torch.abs(torch.diff(energy, dim=0)).sum()

    # Return the mean energy difference, scaled up
    return energy_diff.mean() * loss_weight


def mve_hamiltonian_fn(
    ps_coords: torch.Tensor,
    masses: torch.Tensor,
    potential_fn: Callable = calc_lennard_jones_potential,
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
    r, v = ps_coords[:, 0], ps_coords[:, 1]
    return calc_total_energy(r, v, masses, potential_fn)


def mve_lagrangian_fn(
    r: torch.Tensor,
    v: torch.Tensor,
    masses: torch.Tensor,
    potential_fn: Callable = calc_lennard_jones_potential,
) -> torch.Tensor:
    """
    Lagrangian for a generalized MVE ensemble.

    Args:
        r (torch.Tensor): Particle position coordinates (n_bodies x 2 x n_dims)
        v (torch.Tensor): Velocities of each particle (n_bodies x n_dims)
        masses (torch.Tensor): Masses of each particle (n_bodies)
        potential_fn (callable): Function that computes the potential energy given positions

    Returns:
        torch.Tensor: Lagrangian of the system.
    """
    T = calc_kinetic_energy(v, masses)
    V = potential_fn(r)

    return T - V


class MveEnsembleMechanics(Mechanics):
    """
    Mechanics for an MVE ensemble.
    """

    def __init__(self, args: MechanicsArgs = MechanicsArgs()):
        # potential energy function
        # - Lennard-Jones potential
        # - Boundary potential
        def potential_fn(positions: torch.Tensor):
            bc_pe = calc_boundary_potential(
                positions, boundaries=(args.domain_min, args.domain_max)
            )
            return calc_lennard_jones_potential(positions) + bc_pe

        self.potential_fn = potential_fn
        self.no_bc_potential_fn = partial(calc_lennard_jones_potential)
        self.domain_min = args.domain_min
        self.domain_max = args.domain_max

        _get_generator_fn = lambda masses, generator_type: partial(
            (
                mve_lagrangian_fn
                if generator_type == GeneratorType.LAGRANGIAN
                else mve_hamiltonian_fn
            ),
            masses=masses,
            potential_fn=self.potential_fn,
        )

        super(MveEnsembleMechanics, self).__init__(
            _get_generator_fn,
            get_initial_conditions=get_initial_conditions,
        )
