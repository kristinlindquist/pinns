import torch
from torchdiffeq import odeint
import torch.autograd.functional as AF
from functorch import jacrev


def hamiltonian_fn(coords: torch.Tensor):
    """
    Pendulum Hamiltonian
    """
    q, p = torch.tensor_split(coords, 2)
    H = 3 * (1 - torch.cos(q)) + p**2
    return H


def dynamics_fn(t: torch.Tensor, coords: torch.Tensor):
    """
    Pendulum dynamics
    """
    # 1 x 2
    dcoords = AF.jacobian(hamiltonian_fn, coords)

    # 1
    dqdt, dpdt = dcoords.T

    # 1 x 2
    S = torch.cat([dpdt, -dqdt], axis=-1)
    return S


def get_default_y0(radius: float = 1.3):
    y0 = torch.rand(2) * 2.0 - 1
    y0 = y0 / torch.sqrt((y0**2).sum()) * radius
    return y0


def get_timepoints(t_span: tuple[int, int], timescale: int = 30):
    return torch.linspace(
        t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0]))
    )


def get_trajectory(
    t_span: tuple[int, int] = [0, 3],
    timescale=30,
    radius=torch.rand(1) + 1.3,
    y0=get_default_y0(),
    noise_std=0.1,
    **kwargs
):
    t = get_timepoints(t_span, timescale)
    ivp = odeint(dynamics_fn, t=t, y0=y0, rtol=1e-10, **kwargs)

    q, p = ivp[:, 0].unsqueeze(0), ivp[:, 1].unsqueeze(0)
    dydt = torch.stack([dynamics_fn(None, y) for y in ivp])

    # (t.length, 2) -> tup of (1, t.length)
    dqdt, dpdt = torch.tensor_split(dydt.transpose(0, 1), 2)

    # add noise
    q += torch.randn(*q.shape) * noise_std
    p += torch.randn(*p.shape) * noise_std

    # (1, t.length)
    return q, p, dqdt, dpdt, t


def get_dataset(samples: int = 50, test_split: float = 0.5, **kwargs):
    """
    Generate a dataset of pendulum trajectories
    """
    data = {"meta": locals()}
    torch.seed()
    xs, dxs = [], []
    for s in range(samples):
        x, y, dx, dy, t = get_trajectory(**kwargs)
        xs.append(torch.stack([x, y]).permute(1, 2, 0))
        dxs.append(torch.stack([dx, dy]).permute(1, 2, 0))

    # (num_samples, timescale, 2)
    data["x"] = torch.cat(xs)
    data["dx"] = torch.cat(dxs)

    # make a train/test split
    split_ix = int(len(data["x"]) * test_split)
    split_data = {}
    for k in ["x", "dx"]:
        split_data[k], split_data["test_" + k] = data[k][:split_ix], data[k][split_ix:]

    return split_data


def get_field(
    xmin: float = -1.2,
    xmax: float = 1.2,
    ymin: float = -1.2,
    ymax: float = 1.2,
    gridsize: int = 20,
):
    field = {"meta": locals()}

    # meshgrid to get vector field
    b, a = torch.meshgrid(
        torch.linspace(xmin, xmax, gridsize),
        torch.linspace(ymin, ymax, gridsize),
    )
    ys = torch.stack([b.flatten(), a.flatten()]).transpose(0, 1)

    # get vector directions
    dydt = [dynamics_fn(None, y) for y in ys]

    field["x"] = ys.unsqueeze(0)
    field["dx"] = torch.stack(dydt)

    return field


def get_vector_field(model, **kwargs):
    field = get_field(**kwargs)

    mesh_x = field["x"].requires_grad_()
    mesh_dx = model.time_derivative(mesh_x)

    return mesh_dx.data


def integrate_model(model, t_span: tuple[int, int], y0: int, timescale=30, **kwargs):
    def fun(t, x):
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        _x = x.clone().detach().requires_grad_()
        dx = model.time_derivative(_x).data
        return dx

    t = get_timepoints(t_span, timescale)
    return odeint(fun, t=t, y0=y0, **kwargs)
