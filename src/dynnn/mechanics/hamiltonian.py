import torch
import torch.autograd.functional as AF

from dynnn.types import SystemFunction


def hamiltonian_equation_of_motion(
    hamiltonian_fn: SystemFunction,
    t: torch.Tensor,
    ps_coords: torch.Tensor,
    model: torch.nn.Module | None = None,
) -> torch.Tensor:
    """
    Hamiltonian equations of motion

    Args:
        hamiltonian_fn: hamiltonian function / generator (H = T + V)
        t: time
        ps_coords: phase space coordinates (n_bodies x 2 x n_dims)

    Returns:
        torch.Tensor: time derivative of the phase space coordinates
    """
    if model is not None:
        # model expects batch_size x (time_scale*t_span[1]) x n_bodies x 2 x n_dims
        _ps_coords = ps_coords.reshape(1, 1, *ps_coords.shape)
        dsdt = model.forward(_ps_coords).reshape(ps_coords.shape)
    else:
        dsdt = AF.jacobian(hamiltonian_fn, ps_coords, create_graph=True)

    dhdv, dhdr = dsdt[:, 0], dsdt[:, 1]
    dvdt = -dhdr
    drdt = dhdv
    S = torch.stack([dvdt, drdt], dim=1)

    return S
