import torch
from typing import Callable
import torch.autograd.functional as AF
import math


#   q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
#           @ (jax.grad(lagrangian, 0)(q, q_t)
#              - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
#   return dt*jnp.concatenate([q_t, q_tt])
def lagrangian_dynamics_fn(
    lagrangian_fn: Callable,
    t: torch.Tensor,
    r: torch.Tensor,
    v: torch.Tensor,
    dt=1e-1,
):
    r.requires_grad_(True)
    v.requires_grad_(True)

    hessian = AF.hessian(lambda _v: lagrangian_fn(r, _v), v).view(
        r.shape[0], -1, v.shape[1]
    )

    grad = torch.autograd.grad(lagrangian_fn(r, v), r)[0]

    j = (
        AF.jacobian(
            lambda _r: AF.jacobian(
                lambda _v: lagrangian_fn(_r, _v),
                v,
            ),
            r,
        )
        # @ v.T
    ).view(r.shape[0], -1, v.shape[1])
    print("Hessian", hessian.shape, "J", j.shape, grad.unsqueeze(-2).shape)

    dv = hessian @ (grad.unsqueeze(-2) - j).transpose(1, 2)

    print("dv", dv.shape, v.shape)
    print("OK", v.view(*v.shape, 1).expand_as(dv).shape)

    res = dt * torch.concat([v.view(*v.shape, 1).expand_as(dv), dv])
    return res
