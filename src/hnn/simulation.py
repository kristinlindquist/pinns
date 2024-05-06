import scipy.integrate
import torch
from torchdiffeq import odeint
import numpy as np

import autograd
import autograd.numpy as np

solve_ivp = scipy.integrate.solve_ivp


def hamiltonian_fn(coords):
    q, p = np.split(coords, 2)
    H = 3 * (1 - np.cos(q)) + p**2  # pendulum hamiltonian
    return H


def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = torch.from_numpy(dcoords)
    S = torch.cat([dpdt.unsqueeze(0), -dqdt.unsqueeze(0)], axis=-1)
    return S


def get_trajectory(
    t_span=[0, 3], timescale=15, radius=None, y0=None, noise_std=0.1, **kwargs
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

    spring_ivp = solve_ivp(
        fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs
    )
    q, p = torch.tensor(spring_ivp["y"][0]), torch.tensor(spring_ivp["y"][1])
    dydt = [dynamics_fn(None, y) for y in spring_ivp["y"].T]
    dqdt, dpdt = torch.stack(dydt).T

    # add noise
    q += torch.randn(*q.shape) * noise_std
    p += torch.randn(*p.shape) * noise_std
    return q, p, dqdt, dpdt, t_eval


def get_dataset(samples=50, test_split=0.5, **kwargs):
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
    data = split_data
    return data


def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    field = {"meta": locals()}

    # meshgrid to get vector field
    b, a = torch.meshgrid(
        torch.linspace(xmin, xmax, gridsize), torch.linspace(ymin, ymax, gridsize)
    )
    ys = torch.stack([b.flatten(), a.flatten()])

    # get vector directions
    dydt = [dynamics_fn(None, y.numpy()) for y in ys.T]
    dydt = torch.stack(dydt).T

    field["x"] = ys.T
    field["dx"] = dydt.T
    return field


def get_vector_field(model, **kwargs):
    field = get_field(**kwargs)
    np_mesh_x = field["x"]
    mesh_x = torch.tensor(np_mesh_x, requires_grad=True, dtype=torch.float32)
    mesh_dx = model.time_derivative(mesh_x)
    return mesh_dx.data.numpy()


def integrate_model(model, t_span, y0, **kwargs):
    def fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32).view(1, 2)
        dx = model.time_derivative(x).data.numpy().reshape(-1)
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)
