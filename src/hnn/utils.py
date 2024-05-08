import torch
from torchdiffeq import odeint


def L2_loss(u, v) -> torch.Tensor:
    return (u - v).pow(2).mean()


def get_timepoints(t_span: tuple[int, int], timescale: int = 30) -> torch.Tensor:
    return torch.linspace(
        t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0]))
    )


def integrate_model(model, t_span: tuple[int, int], y0: int, timescale=30, **kwargs):
    def fun(t, x):
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        _x = x.clone().detach().requires_grad_()
        dx = model.time_derivative(_x).data
        return dx

    t = get_timepoints(t_span, timescale)
    return odeint(fun, t=t, y0=y0, **kwargs)
