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
        y0: torch.Tensor,
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
        self.y0 = y0
        self.t_span = t_span

    def dynamics_fn(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Hamiltonian dynamics function

        Finds the Jacobian of the Hamiltonian function
        """
        # n_bodies x 2 x num_dim
        d_state = AF.jacobian(self.function, state)

        drdt, dvdt = [v.squeeze() for v in torch.split(d_state, 1, dim=1)]
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
        b, a = torch.meshgrid(
            torch.linspace(xmin, xmax, gridsize),
            torch.linspace(ymin, ymax, gridsize),
            indexing="xy",
        )
        states = torch.stack([b.flatten(), a.flatten()], dim=1)

        # num_samples*t_span[1] x n_bodies
        dsdt = torch.stack([self.dynamics_fn(None, s) for s in states])

        field = HamiltonianField(meta=locals(), x=states, dx=dsdt)
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

        if field.x.dim() == 3:
            field.x = field.x.unsqueeze(0)

        mesh_x = field.x.requires_grad_()
        mesh_dx = model.time_derivative(mesh_x)

        return mesh_dx.data.squeeze(0)

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
            args.timescale: Timescale
            args.noise_std: Noise standard deviation
            ode_args: Additional arguments
        """
        timescale, noise_std = args.dict().values()

        t = get_timepoints(self.t_span, timescale)
        ivp = odeint(self.dynamics_fn, t=t, rtol=1e-10, y0=self.y0, **ode_args)

        # num_samples*t_span[1] x n_bodies x 2
        r, v = ivp[:, :, 0], ivp[:, :, 1]
        r += torch.randn(*r.shape) * noise_std  # add noise
        v += torch.randn(*v.shape) * noise_std  # add noise

        dsdt = torch.stack([self.dynamics_fn(None, s) for s in ivp])
        # -> num_samples*t_span[1] x n_bodies x num_dim
        drdt, dvdt = [d.squeeze() for d in torch.split(dsdt, 1, dim=2)]

        return r, v, drdt, dvdt, t

    @multidispatch
    def get_dataset(self, args, trajectory_args, ode_args):
        return NotImplemented

    @overload
    @get_dataset.register
    def _(self, args: dict = {}, trajectory_args: dict = {}, ode_args: dict = {}):
        return self.get_dataset(
            DatasetArgs(**args), TrajectoryArgs(**trajectory_args), ode_args
        )

    @overload
    @get_dataset.register
    def _(
        self, args: DatasetArgs, trajectory_args: TrajectoryArgs, ode_args: dict = {}
    ) -> dict:
        """
        Generate a dataset of trajectories

        Args:
        args.num_samples: Number of samples
        args.test_split: Test split
        trajectory_args: Additional arguments for the trajectory function
        ode_args: Additional arguments for the ODE solver
        """

        num_samples, test_split = args.dict().values()

        torch.seed()
        xs, dxs = [], []
        for s in range(num_samples):
            r, v, dr, dv, t = self.get_trajectory(trajectory_args, ode_args)

            # (timescale*t_span[1]) x n_bodies x 2 x num_dim
            xs.append(torch.stack([r, v], dim=2).unsqueeze(dim=0))

            # (timescale*t_span[1]) x n_bodies x 2 x num_dim
            dxs.append(torch.stack([dr, dv], dim=2).unsqueeze(dim=0))

        # num_samples x (timescale*t_span[1]) x n_bodies x 2 x num_dim
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
