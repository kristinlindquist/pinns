import torch
from torchdyn.numerics.odeint import odeint


def get_timepoints(t_span: tuple[int, int], time_scale: int = 30) -> torch.Tensor:
    return torch.linspace(
        t_span[0], t_span[1], int(time_scale * (t_span[1] - t_span[0]))
    )


def permutation_tensor() -> torch.Tensor:
    """
    Constructs the Levi-Civita permutation tensor for 3 dimensions.
    """
    P = torch.zeros((3, 3, 3))
    P[0, 1, 2] = 1
    P[1, 2, 0] = 1
    P[2, 0, 1] = 1
    P[2, 1, 0] = -1
    P[1, 0, 2] = -1
    P[0, 2, 1] = -1
    return P


def integrate_model(
    model, t_span: tuple[int, int], y0: torch.Tensor, time_scale: int = 30, **kwargs
):
    def fun(t, x):
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        _x = x.clone().detach().requires_grad_()
        dx = model.forward(_x).data
        return dx

    t = get_timepoints(t_span, time_scale)
    return odeint(fun, t=t, y0=y0, **kwargs)
