from typing import Callable, overload
import torch
from functools import partial
from torchdiffeq import odeint
import torch.autograd.functional as AF
from pydantic import BaseModel
from multimethod import multidispatch, multimethod


from hnn.types import (
    HamiltonianField,
    HamiltonianFunction,
    TrajectoryArgs,
    FieldArgs,
    DatasetArgs,
)
from hnn.utils import get_timepoints


class HamiltonianDynamics:
    """
    Hamiltonian dynamics class
    """

    def __init__(
        self,
        function: HamiltonianFunction,
    ):
        """
        Initialize the class

        Args:
            function: Hamiltonian function
        """
        self.function = function

    def dynamics_fn(self, t: torch.Tensor, coords: torch.Tensor):
        # 1 x 2
        dcoords = AF.jacobian(self.function, coords)

        # 1
        dqdt, dpdt = dcoords.T

        # 1 x 2
        S = torch.cat([dpdt, -dqdt], axis=-1)
        return S

    @multidispatch
    def get_field(self, args):
        return NotImplemented

    @overload
    @get_field.register
    def _(self, args: dict) -> HamiltonianField:
        return self.get_field(FieldArgs(**args))

    @overload
    @get_field.register
    def _(
        self,
        args: FieldArgs = FieldArgs(),
    ) -> HamiltonianField:
        xmin, xmax, ymin, ymax, gridsize = args.dict().values()

        # meshgrid to get vector field
        b, a = torch.meshgrid(
            torch.linspace(xmin, xmax, gridsize),
            torch.linspace(ymin, ymax, gridsize),
        )
        ys = torch.stack([b.flatten(), a.flatten()]).transpose(0, 1)

        # get vector directions
        dydt = [self.dynamics_fn(None, y) for y in ys]

        field = HamiltonianField(meta=locals(), x=ys.unsqueeze(0), dx=torch.stack(dydt))
        return field

    @multidispatch
    def get_vector_field(self, model, field_args):
        return NotImplemented

    @overload
    @get_vector_field.register
    def _(self, model: torch.nn.Module, field_args: dict = {}) -> torch.Tensor:
        return self.get_vector_field(model, FieldArgs(**field_args))

    @overload
    @get_vector_field.register
    def _(self, model: torch.nn.Module, field_args: FieldArgs) -> torch.Tensor:
        field = self.get_field(field_args)

        mesh_x = field.x.requires_grad_()
        mesh_dx = model.time_derivative(mesh_x)

        return mesh_dx.data

    @multidispatch
    def get_trajectory(self, args, ode_args):
        return NotImplemented

    @overload
    @get_trajectory.register
    def _(self, args: dict = {}, ode_args: dict = {}):
        return self.get_trajectory(TrajectoryArgs(**args), **ode_args)

    @overload
    @get_trajectory.register
    def _(self, args: TrajectoryArgs, ode_args: dict = {}):
        """
        Get a trajectory

        Args:
            args.t_span: Time span
            args.y0: Initial conditions
            args.timescale: Timescale
            args.noise_std: Noise standard deviation
            ode_args: Additional arguments
        """
        t_span, y0, timescale, noise_std = args.dict().values()

        t = get_timepoints(t_span, timescale)
        ivp = odeint(self.dynamics_fn, t=t, y0=y0, rtol=1e-10, **ode_args)

        q, p = ivp[:, 0].unsqueeze(0), ivp[:, 1].unsqueeze(0)
        dydt = torch.stack([self.dynamics_fn(None, y) for y in ivp])

        # (t.length, 2) -> tup of (1, t.length)
        dqdt, dpdt = torch.tensor_split(dydt.transpose(0, 1), 2)

        # add noise
        q += torch.randn(*q.shape) * noise_std
        p += torch.randn(*p.shape) * noise_std

        # (1, t.length)
        return q, p, dqdt, dpdt, t

    @multidispatch
    def get_dataset(self, args, trajectory_args):
        return NotImplemented

    @overload
    @get_dataset.register
    def _(self, args: dict = {}, trajectory_args: dict = {}):
        return self.get_dataset(DatasetArgs(**args), TrajectoryArgs(**trajectory_args))

    @overload
    @get_dataset.register
    def _(self, args: DatasetArgs, trajectory_args: TrajectoryArgs) -> dict:
        """
        Generate a dataset of trajectories

        Args:
        args.num_samples: Number of samples
        args.test_split: Test split
        trajectory_args: Additional arguments for the trajectory function
        """

        num_samples, test_split = args.dict().values()

        torch.seed()
        xs, dxs = [], []
        for s in range(num_samples):
            x, y, dx, dy, t = self.get_trajectory(trajectory_args)
            xs.append(torch.stack([x, y]).permute(1, 2, 0))
            dxs.append(torch.stack([dx, dy]).permute(1, 2, 0))

        # (num_samples, timescale, 2)
        data = {
            "meta": locals(),
            "x": torch.cat(xs),
            "dx": torch.cat(dxs),
        }

        # make a train/test split
        split_ix = int(len(data["x"]) * test_split)
        split_data = {}
        for k in ["x", "dx"]:
            split_data[k], split_data["test_" + k] = (
                data[k][:split_ix],
                data[k][split_ix:],
            )

        return split_data
