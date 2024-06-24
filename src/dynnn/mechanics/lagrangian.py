import torch
from typing import Callable
import torch.autograd.functional as AF


def _lagrangian_equation_of_motion(
    generator_fn: Callable,
    t: torch.Tensor,
    ps_coords: torch.Tensor,
) -> torch.Tensor:
    """
    Lagrangian equation of motion (EOM) / Euler-Lagrange equation
    "The principle of least action"

    = d/dt (∂L/∂v) - ∂L/∂r = 0
    = accelerations = dv/dt = (∂²L/∂v²)^(-1) * (∂L/∂r - ∂²L/(∂r∂v) * v)

    Args:
        generator_fn (Callable): Lagrangian generator function (L = T - V)
        t (torch.Tensor): Time
        ps_coords (torch.Tensor): Phase space coordinates (n_bodies x 2 x n_dims)

    Returns:
        torch.Tensor: time derivative of the phase space coordinates (n_bodies x 2 x n_dims))

    Should be equivalent to this JAX code:
    ```
    q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
            @ (jax.grad(lagrangian, 0)(q, q_t)
                - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
    return dt*jnp.concatenate([q_t, q_tt])
    (from https://github.com/MilesCranmer/lagrangian_nns)
    ```
    """
    q, v = [t.squeeze() for t in torch.split(ps_coords, 1, dim=1)]

    # ∂L/∂q: 1st-order partial derivatives of L with respect to q
    # n_bodies x n_dims
    dLdq = torch.autograd.grad([generator_fn(q, v)], [q], create_graph=True)[0]
    assert dLdq is not None

    # ∂²L/∂v²
    dLdv = AF.hessian(lambda _v: generator_fn(q, _v), v, create_graph=True)
    dLdv_inv = torch.linalg.pinv(dLdv).reshape(dLdv.shape)

    # ∂²L/(∂r∂v) : gradient of L with respect to v changes with q
    dLdqdv = AF.jacobian(
        lambda _q: AF.jacobian(lambda _v: generator_fn(_q, _v), v, create_graph=True),
        q,
        create_graph=True,
    )
    # (∂²L/(∂q∂v)) * v
    dLdqdv_term = torch.einsum("ijkl,il->ij", dLdqdv, v)

    # (∂²L/∂v²)^(-1) * (∂L/∂q - ∂²L/(∂q∂v) * v)
    accelerations = torch.einsum("ijkl,il->ij", dLdv_inv, (dLdq - dLdqdv_term))

    # time derivative of the phase space coordinates
    d_ps_coords = torch.stack([v, accelerations], dim=1)

    return d_ps_coords


def lagrangian_equation_of_motion(
    generator_fn: Callable,
    t: torch.Tensor,
    ps_coords: torch.Tensor,
    model: torch.nn.Module = None,
) -> torch.Tensor:
    """
    Lagrangian equation of motion (EOM)

    Args:
        generator_fn (Callable): Lagrangian generator function (L = T - V)
        t (torch.Tensor): Time
        ps_coords (torch.Tensor): Phase space coordinates (n_bodies x 2 x n_dims)
        model (torch.nn.Module): model to use for time derivative

    Returns:
        torch.Tensor: time derivative of the phase space coordinates (n_bodies x 2 x n_dims)
    """
    if model is not None:
        # model expects batch_size x (time_scale*t_span[1]) x n_bodies x 2 x n_dims
        _ps_coords = ps_coords.reshape(1, 1, *ps_coords.shape)
        dsdt = model.forward(_ps_coords).reshape(ps_coords.shape)
        v, dv = dsdt[:, 0], dsdt[:, 1]
        return torch.stack([v, dv], dim=1)

    return _lagrangian_equation_of_motion(generator_fn, t, ps_coords)
