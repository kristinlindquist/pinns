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
        domain: tuple[int, int],
        t_span: tuple[int, int] = (0, 10),
    ):
        """
        Initialize the class

        Args:
            function: Hamiltonian function
            domain (tuple[int, int]): domain (boundary) for all dimensions
        """
        self.function = function
        self.domain = domain
        self.t_span = t_span

    def dynamics_fn(
        self,
        t: torch.Tensor,
        ps_coords: torch.Tensor,
        model: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        """
        Hamiltonian dynamics function

        Finds the Jacobian of the Hamiltonian function

        Args:
            t: Time
            ps_coords: phase space coordinates (n_bodies x 2 x num_dim)
            model: model to use for time derivative
        """
        # n_bodies x 2 x num_dim
        if model is not None:
            d_ps_coords = (
                # TODO: hacky
                model.time_derivative(ps_coords.unsqueeze(0).unsqueeze(0))
                .squeeze()
                .squeeze()
            )
        else:
            d_ps_coords = AF.jacobian(self.function, ps_coords)

        drdt, dvdt = [v.squeeze() for v in torch.split(d_ps_coords, 1, dim=1)]
        # dvdt = -dHdr; drdt = dHdv
        S = torch.stack([dvdt, -drdt], dim=1)

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
        mesh = torch.meshgrid(
            torch.linspace(xmin, xmax, gridsize),
            torch.linspace(ymin, ymax, gridsize),
            indexing="xy",
        )
        ps_coords = torch.stack([m.flatten() for m in mesh], dim=1)

        # -> n_bodies x 2 x num_dim
        dsdt = torch.stack([self.dynamics_fn(None, c) for c in ps_coords])

        field = HamiltonianField(meta=locals(), x=ps_coords, dx=dsdt)
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

        return mesh_dx.data.squeeze(0)

    @multidispatch
    def get_trajectory(self, args):
        return NotImplemented

    @overload
    @get_trajectory.register
    def _(self, args: dict = {}):
        return self.get_trajectory(TrajectoryArgs(**args))

    @overload
    @get_trajectory.register
    def _(
        self, args: TrajectoryArgs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a trajectory

        Args:
            args.y0: Initial conditions
            args.timescale: Timescale
            args.noise_std: Noise standard deviation
            args.dynamics_fn: Dynamics function (otherwise self.dynamics_fn is used)
        """
        y0, timescale, noise_std = args.y0, args.timescale, args.noise_std

        dynamics_fn = args.dynamics_fn or self.dynamics_fn

        t = get_timepoints(self.t_span, timescale)
        ivp = odeint(dynamics_fn, t=t, rtol=1e-10, y0=y0)

        # num_batches*t_span[1] x n_bodies x 2
        r, v = ivp[:, :, 0], ivp[:, :, 1]
        r += torch.randn(*r.shape) * noise_std  # add noise
        v += torch.randn(*v.shape) * noise_std  # add noise

        dsdt = torch.stack([dynamics_fn(None, s) for s in ivp])
        # -> num_batches*t_span[1] x n_bodies x num_dim
        drdt, dvdt = [d.squeeze() for d in torch.split(dsdt, 1, dim=2)]

        return r, v, drdt, dvdt, t

    @multidispatch
    def get_dataset(self, args, trajectory_args):
        return NotImplemented

    @overload
    @get_dataset.register
    def _(self, args: dict = {}, trajectory_args: dict = {}):
        return self.get_dataset(DatasetArgs(**args), TrajectoryArgs(**trajectory_args))

    @overload
    @get_dataset.register
    def _(
        self,
        args: DatasetArgs,
        trajectory_args: TrajectoryArgs,
    ) -> dict:
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
            r, v, dr, dv, t = self.get_trajectory(trajectory_args)

            # (timescale*t_span[1]) x n_bodies x 2 x num_dim
            xs.append(torch.stack([r, v], dim=2).unsqueeze(dim=0))

            # (timescale*t_span[1]) x n_bodies x 2 x num_dim
            dxs.append(torch.stack([dr, dv], dim=2).unsqueeze(dim=0))

        # batch_size x (timescale*t_span[1]) x n_bodies x 2 x num_dim
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
