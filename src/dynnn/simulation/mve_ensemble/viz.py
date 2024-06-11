"""
Visualization helpers for MVE ensemble
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Callable
import torch


def visualize_trajectory(
    positions: torch.Tensor, num_timepoints: int, domain: tuple[int, int] = (0, 10)
):
    """
    Visualize the trajectory of particles in 3D space

    Positions: (time_scale*t_span[1]) x n_bodies x n_dims

    Args:
        positions (torch.Tensor): Tensor containing xy positions of particles
            size:
                n_bodies x num_dims or
                timepoints x n_bodies x num_dims
        num_timepoints (int): Number of timepoints
        domain (tuple[int, int]): Tuple containing the domain limits

    Returns:
        animation object
    """
    fig = plt.figure(figsize=[7, 5], dpi=100)
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=20.0, azim=-35, roll=0)

    splot = ax.scatter([], [], [])

    def update(frame):
        ax.clear()
        ax.set_title("Trajectories")
        ax.set_xlim(*domain)
        ax.set_ylim(*domain)
        ax.set_zlim(*domain)

        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$z$")

        for i in range(positions.shape[1]):
            x, y, z = (
                positions[frame, i, 0],
                positions[frame, i, 1],
                positions[frame, i, 2],
            )
            ax.scatter(x.item(), y.item(), z.item())
        return []

    ani = FuncAnimation(fig, update, frames=num_timepoints, blit=True, repeat=False)
    return ani


def plot_energy(
    potential_energy: torch.Tensor,
    kinetic_energy: torch.Tensor,
    total_energy: torch.Tensor,
    time: torch.Tensor,
):
    """
    Plot the potential, kinetic, and total energy of the system
    """
    fig_e, ax_e = plt.subplots(figsize=[10, 4], dpi=100)
    plt.title("Energy")
    plt.xlabel("time")
    plt.plot(time, potential_energy, label="potential")
    plt.plot(time, kinetic_energy, label="kinetic")
    plt.plot(time, total_energy, label="total")
    plt.legend(fontsize=8)
