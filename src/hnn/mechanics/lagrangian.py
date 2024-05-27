import torch
from typing import Callable
import torch.autograd.functional as AF
import math


def lagrangian_dynamics_fn(
    lagrangian_fn: Callable,
    t: torch.Tensor,
    r: torch.Tensor,
    v: torch.Tensor,
    dt=1e-1,
) -> torch.Tensor:
    """
    Lagrangian dynamics function

    Returns tensor shape (n_bodies x 2 x n_dims)

    Should be equivalent to this JAX code:
    ```
        q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
                @ (jax.grad(lagrangian, 0)(q, q_t)
                    - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
        return dt*jnp.concatenate([q_t, q_tt])
    ```
    """
    r.requires_grad_(True)
    v.requires_grad_(True)

    # grad_r (∇L(q)): 1st-order partial derivatives of L with respect to r
    # n_bodies x n_dims
    grad_r = torch.autograd.grad(lagrangian_fn(r, v), r)[0]

    # Compute Hessian with respect to v, hessian_v
    # - 2nd-order partial derivatives of L with respect to v
    # - captures how the rate of change of L changes with v / curvature of L in v space
    # n_bodies x n_dims x n_bodies x n_dims
    hessian_v = AF.hessian(lambda _v: lagrangian_fn(r, _v), v)

    # double jacobian (second-order mixed partial derivatives)
    # gradient of L with respect to v changes with r
    # n_bodies x n_dims x n_bodies x n_dims
    jacobian = AF.jacobian(
        lambda _r: AF.jacobian(
            lambda _v: lagrangian_fn(_r, _v),
            v,
        ),
        r,
    )
    jvp_v = torch.einsum("ijkl,il->ij", jacobian, v)

    # dv = H**−1 ⋅ (∇rL−jvp)
    hessian_v_inv = torch.linalg.pinv(hessian_v).view(*hessian_v.shape)
    dv = torch.einsum("bjkl,bl->bj", hessian_v_inv, (grad_r - jvp_v))

    new_state = dt * torch.cat([v.unsqueeze(1), dv.unsqueeze(1)], dim=1)

    return new_state
