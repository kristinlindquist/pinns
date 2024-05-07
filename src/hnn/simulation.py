import torch
from torchdiffeq import odeint
import torch.autograd.functional as AF
from functorch import jacrev


def hamiltonian_fn(coords):
    q, p = torch.tensor_split(coords, 2)
    H = 3 * (1 - torch.cos(q)) + p**2  # pendulum hamiltonian
    return H


def dynamics_fn(t, coords):
    dcoords = AF.jacobian(hamiltonian_fn, coords)
    dqdt, dpdt = torch.tensor_split(dcoords[0], 2)
    S = torch.cat([dpdt, -dqdt], axis=-1)
    return S


from xitorch.integrate import solve_ivp


def get_trajectory(
    t_span: tuple[int, int] = [0, 3],
    timescale=14,
    radius=None,
    y0=None,
    noise_std=0.1,
    **kwargs
):
    t_eval = torch.linspace(
        t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0]))
    )

    # get initial state
    if y0 is None:
        y0 = torch.rand(2) * 2.0 - 1
    if radius is None:
        radius = torch.rand(1) + 1.3  # sample a range of radii
    y0 = y0 / torch.sqrt((y0**2).sum()) * radius  ## set the appropriate radius

    ivp = odeint(dynamics_fn, t=t_eval, y0=y0, rtol=1e-10, **kwargs)  # t_eval

    q, p = ivp[:, 0], ivp[:, 1]
    dydt = [dynamics_fn(None, y) for y in ivp]
    dqdt, dpdt = torch.tensor_split(torch.stack(dydt).T, 2)

    # add noise
    q += torch.randn(*q.shape) * noise_std
    p += torch.randn(*p.shape) * noise_std

    return q, p, dqdt, dpdt, t_eval


def get_dataset(samples: int = 50, test_split: float = 0.5, **kwargs):
    data = {"meta": locals()}
    torch.seed()
    xs, dxs = [], []
    for s in range(samples):
        x, y, dx, dy, t = get_trajectory(**kwargs)
        xs.append(torch.stack([x, y]).T)
        dxs.append(torch.stack([dx, dy]).T)

    data["x"] = torch.cat(xs)
    data["dx"] = torch.cat(dxs).squeeze()

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
    ys = torch.stack([b.flatten(), a.flatten()]).T

    # get vector directions
    dydt = [dynamics_fn(None, y) for y in ys]

    field["x"] = ys
    field["dx"] = torch.stack(dydt)

    return field


def get_vector_field(model, **kwargs):
    field = get_field(**kwargs)

    mesh_x = field["x"].requires_grad_()
    mesh_dx = model.time_derivative(mesh_x)

    return mesh_dx.data


def integrate_model(model, t_span: tuple[int, int], y0: int, timescale=14, **kwargs):
    def fun(t, x):
        _x = x.clone().detach().requires_grad_()
        dx = model.time_derivative(_x).data.reshape(-1)
        return dx

    t_eval = torch.linspace(
        t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0]))
    )
    return odeint(fun, t=t_eval, y0=y0, **kwargs)
