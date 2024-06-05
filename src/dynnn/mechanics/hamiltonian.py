import torch
import torch.autograd.functional as AF

from dynnn.types import GeneratorFunction


def hamiltonian_equation_of_motion(
    generator_fn: GeneratorFunction,
    t: torch.Tensor,
    ps_coords: torch.Tensor,
    model: torch.nn.Module | None = None,
) -> torch.Tensor:
    """
    Hamiltonian equations of motion

    Args:
        generator_fn: hamiltonian generator function (H = T + V)
        t: time
        ps_coords: phase space coordinates (n_bodies x 2 x n_dims)

    Returns:
        torch.Tensor: time derivative of the phase space coordinates (in this case, the symplectic gradient)
    """
    if model is not None:
        # model expects batch_size x (time_scale*t_span[1]) x n_bodies x 2 x n_dims
        _ps_coords = ps_coords.reshape(1, 1, *ps_coords.shape)
        dsdt = model.forward(_ps_coords).reshape(ps_coords.shape)
    else:
        dsdt = AF.jacobian(generator_fn, ps_coords, create_graph=True)

    # because (dq/dt - ∂H/∂p = 0) and (dp/dt + ∂H/∂q = 0)
    # we can return (∂H/∂p, -∂H/∂q) which should equal the training set (dq/dt, dv/dt)
    dhdq, dhdp = dsdt[:, 0], dsdt[:, 1]
    dpdt = -dhdq
    dqdt = dhdp
    S = torch.stack([dqdt, dpdt], dim=1)

    return S
