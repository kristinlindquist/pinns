from typing import overload, Callable
from multimethod import multidispatch
import torch
from functools import partial

from hnn.dynamics import HamiltonianDynamics


def get_initial_conditions(
    n_bodies: int,
    n_dims: int,
    width: int = 8,
    height: int = 8,
    depth: int = 8,
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

    ps_coords = torch.stack([r, v], dim=1)  # n_bodies x 2 x n_dims

    return ps_coords, masses


def calc_boundary_potential(
    positions: torch.Tensor,
    boundaries: tuple[float, float],
    steepness: float = 10.0,
    width: float = 0.15,
):
    """
    A conservative boundary potential that fades out smoothly
    NOTE: energy is not conserved around this boundary
    """

    def _boundary_potential(boundary):
        distance = torch.abs(positions - boundary).reshape(-1)
        mask = distance < width
        distance[mask] = steepness * (1 - (distance[mask] / width) ** 2) ** 2
        return distance

    boundary_effect = torch.sum(
        torch.stack([_boundary_potential(b) for b in boundaries])
    )
    return boundary_effect


def calc_lennard_jones_potential(
    positions: torch.Tensor, σ: float = 1.0, ε: float = 1.0
) -> torch.Tensor:
    """
    Lennard-Jones potential function.
    V(r) = 4 * ε * ((σ / r)^12 - (σ / r)^6

    Args:
        positions (torch.Tensor): Tensor containing positions of particles
            size:
                n_bodies x 2 or
                n_bodies x timepoints x 2
        σ (float): "the distance at which the particle-particle potential energy V is zero"
        ε (float): "depth of the potential well"

    Returns:
        torch.Tensor: Total potential energy (v).

    See https://en.wikipedia.org/wiki/Lennard-Jones_potential
    """
    n_bodies = positions.shape[0]
    return_dim = len(positions.shape) - 2

    v = 0.0
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
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
                n_bodies x 2 or
                n_bodies x timepoints x 2
        masses (torch.Tensor): Tensor containing masses of particles
    """
    if len(masses.shape) == 1:
        # Reshape from (n_bodies,) to (n_bodies, 1)
        masses = masses.unsqueeze(-1)

    # Ensure masses can broadcast correctly with velocities
    if len(velocities.shape) == 3:
        masses = masses.unsqueeze(1)

    kinetics = 0.5 * masses * velocities**2
    kinetic_energy = torch.sum(kinetics, dim=-1)

    # Average the kinetic energies over all particles
    if kinetic_energy.dim() > 1:
        # If we have timepoints, average across all particles for each timepoint
        return kinetic_energy.mean(dim=0)

    return kinetic_energy.mean()


def mve_ensemble_fn(
    ps_coords: torch.Tensor,
    masses: torch.Tensor,
    potential_fn=calc_lennard_jones_potential,
) -> torch.Tensor:
    """
    Hamiltonian for a generalized MVE ensemble.

    Args:
        ps_coords (torch.Tensor): Phase space coordinates (n_bodies x 2 x n_dims)
        masses (torch.Tensor): Masses of each particle (n_bodies)
        potential_fn (callable): Function that computes the potential energy given positions

    Returns:
        torch.Tensor: Hamiltonian (Total energy) of the system.
    """
    # Split coordinates into positions and momentum  (-> n_bodies x num_dim)
    r, v = [s.squeeze() for s in torch.split(ps_coords, 1, dim=1)]

    # Compute kinetic energy
    kinetic_energy = calc_kinetic_energy(v, masses)

    # Compute potential energy
    potential_energy = potential_fn(r)

    # Hamiltonian (Total Energy)
    H = kinetic_energy + potential_energy

    return H


class MveEnsembleHamiltonianDynamics(HamiltonianDynamics):
    def __init__(
        self,
        domain: tuple[int, int] = (0, 10),
        t_span: tuple[int, int] = (0, 20),
    ):
        # potential energy function
        # - Lennard-Jones potential
        # - Boundary potential
        def potential_fn(positions: torch.Tensor):
            bc_pe = calc_boundary_potential(positions, boundaries=domain)
            return calc_lennard_jones_potential(positions) + bc_pe

        self.potential_fn = potential_fn
        self.potential_wo_bc_fun = partial(calc_lennard_jones_potential)

        _get_function = lambda masses: partial(
            mve_ensemble_fn, masses=masses, potential_fn=self.potential_fn
        )

        super(MveEnsembleHamiltonianDynamics, self).__init__(
            _get_function, domain, t_span=t_span
        )
