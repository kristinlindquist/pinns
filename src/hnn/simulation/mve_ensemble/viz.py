import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def visualize_trajectory(positions, num_timepoints, domain=(0, 10)):
    """
    Visualize the trajectory of particles in 3D space
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

        for i in range(positions.shape[0]):
            x, y, z = (
                positions[i, frame, 0],
                positions[i, frame, 1],
                positions[i, frame, 2],
            )
            ax.scatter(x.item(), y.item(), z.item())
        return []

    ani = FuncAnimation(fig, update, frames=num_timepoints, blit=True, repeat=False)
    return ani
