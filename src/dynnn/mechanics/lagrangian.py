import torch
from typing import Callable
import torch.autograd.functional as AF
import math
from itertools import permutations, combinations, combinations_with_replacement


def lagrangian_equation_of_motion(
    lagrangian_fn: Callable, t: torch.Tensor, ps_coords: torch.Tensor
) -> torch.Tensor:
    """
    Lagrangian equation of motion (EOM)
    "The prinicple of least action" - Euler-Lagrange equation

    Returns state update (size: n_bodies x 2 x n_dims)

    Should be equivalent to this JAX code:
    ```
    q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
            @ (jax.grad(lagrangian, 0)(q, q_t)
                - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
    return dt*jnp.concatenate([q_t, q_tt])
    (from https://github.com/MilesCranmer/lagrangian_nns)
    ```
    """
    r, v = [t.squeeze() for t in torch.split(ps_coords, 1, dim=1)]

    # grad_r (∇L(q)): 1st-order partial derivatives of L with respect to r
    # n_bodies x n_dims
    grad_r = torch.autograd.grad(lagrangian_fn(r, v), r, create_graph=True)[0]

    # Compute Hessian with respect to v, hessian_v
    # - 2nd-order partial derivatives of L with respect to v
    # - captures how the rate of change of L changes with v / curvature of L in v space
    hessian_v = AF.hessian(lambda _v: lagrangian_fn(r, _v), v, create_graph=True)

    # double jacobian (second-order mixed partial derivatives)
    # gradient of L with respect to v changes with r
    jacobian = AF.jacobian(
        lambda _r: AF.jacobian(lambda _v: lagrangian_fn(_r, _v), v, create_graph=True),
        r,
        create_graph=True,
    )
    jvp_v = torch.einsum("ijkl,il->ij", jacobian, v)

    # dv = H**−1 ⋅ (∇rL−jvp)
    hessian_v_inv = torch.linalg.pinv(hessian_v).reshape(*hessian_v.shape)
    dv = torch.einsum("ijkl,il->ij", hessian_v_inv, (grad_r - jvp_v))

    S = torch.stack([v, dv], dim=1)

    return S
