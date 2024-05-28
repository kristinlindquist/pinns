from typing import Any, Callable, overload
import torch
from functools import partial
from torchdyn.numerics.odeint import odeint
import torch.autograd.functional as AF
from pydantic import BaseModel
from multimethod import multidispatch, multimethod


from hnn.mechanics.lagrangian import lagrangian_equation_of_motion as lagrangian_eom
from hnn.types import (
    HamiltonianFunction,
    TrajectoryArgs,
    DatasetArgs,
    Trajectory,
)
from hnn.utils import get_timepoints


class Mechanics:
    """
    Classical mechanics system class

    On choosing an ODE solver: https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/
    """

    def __init__(
        self,
        get_function: Callable[[Any], HamiltonianFunction],
        domain: tuple[int, int],
        t_span: tuple[int, int] = (0, 10),
        use_lagrangian: bool = False,
    ):
        """
        Initialize the class

        Args:
            get_function: function returning Hamiltonian function
            domain (tuple[int, int]): domain (boundary) for all dimensions
            t_span (tuple[int, int]): time span
            use_lagrangian (bool): whether to use the Lagrangian formulation
        """
        self.get_function = get_function
        self.domain = domain
        self.t_span = t_span
        self.use_lagrangian = use_lagrangian

    def dynamics_fn(
        self,
        t: torch.Tensor,
        ps_coords: torch.Tensor,
        model: torch.nn.Module | None = None,
        function_args: dict = {},
    ) -> torch.Tensor:
        """
        Dynamics function - finds the state update for the supplied function

        Args:
            t: Time
            ps_coords: phase space coordinates (n_bodies x 2 x n_dims)
            model: model to use for time derivative
            function_args: additional arguments for the Hamiltonian function
        """
        if t is not None:
            print(t)

        function = self.get_function(**function_args)

        if self.use_lagrangian:
            return lagrangian_eom(function, t, ps_coords)

        # n_bodies x 2 x n_dims
        if model is not None:
            # model expects batch_size x (time_scale*t_span[1]) x n_bodies x 2 x n_dims
            _ps_coords = ps_coords.unsqueeze(0).unsqueeze(0)
            drdt, dvdt = model.time_derivative(_ps_coords).squeeze().squeeze()
        else:
            dsdt = AF.jacobian(function, ps_coords)  # diff than jacobian(fun, (r, v))
            drdt, dvdt = [d.squeeze() for d in torch.split(dsdt, 1, dim=1)]

        S = torch.stack([dvdt, -drdt], dim=1)  # dvdt = -dHdr; drdt = dHdv

        return S

    @multidispatch
    def get_trajectory(self, args) -> Trajectory:
        return NotImplemented

    @overload
    @get_trajectory.register
    def _(self, args: dict = {}) -> Trajectory:
        return self.get_trajectory(TrajectoryArgs(**args))

    @overload
    @get_trajectory.register
    def _(self, args: TrajectoryArgs) -> Trajectory:
        """
        Get a trajectory

        Args:
            args.y0: Initial conditions
            args.masses: Masses
            args.time_scale: Time scale
            args.model: Model to use for time derivative (optional)
        """
        y0, time_scale = args.y0, args.time_scale
        dynamics_fn = partial(
            self.dynamics_fn,
            model=args.model,
            function_args={"masses": args.masses},
        )

        t = get_timepoints(self.t_span, time_scale)
        ivp = odeint(
            f=dynamics_fn,
            x=y0,
            t_span=t,
            solver="tsit5",  # tsit5, dopri5, alf, euler, midpoint, rk4, ieuler
            rtol=1e-10,
            atol=1e-7,
        )[1]

        # -> time_scale*t_span[1] x n_bodies x 2 x n_dims
        dsdt = torch.stack([dynamics_fn(None, dp) for dp in ivp])
        # -> time_scale*t_span[1] x n_bodies x n_dims
        drdt, dvdt = [d.squeeze() for d in torch.split(dsdt, 1, dim=2)]

        # -> time_scale*t_span[1] x n_bodies x 2
        r, v = ivp[:, :, 0], ivp[:, :, 1]

        return Trajectory(r=r, v=v, dr=drdt, dv=dvdt, t=t)

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
        xs, dxs, time = [], [], None
        for s in range(num_samples):
            r, v, dr, dv, t = self.get_trajectory(trajectory_args).dict().values()

            if time is None:
                time = t

            # (time_scale*t_span[1]) x n_bodies x 2 x n_dims
            xs.append(torch.stack([r, v], dim=2).unsqueeze(dim=0))

            # (time_scale*t_span[1]) x n_bodies x 2 x n_dims
            dxs.append(torch.stack([dr, dv], dim=2).unsqueeze(dim=0))

        # batch_size x (time_scale*t_span[1]) x n_bodies x 2 x n_dims
        data = {"meta": locals(), "x": torch.cat(xs), "dx": torch.cat(dxs)}

        # make a train/test split
        split_ix = int(len(data["x"]) * test_split)
        split_data = {"time": time}
        for k in ["x", "dx"]:
            split_data[k], split_data["test_" + k] = (
                data[k][:split_ix],
                data[k][split_ix:],
            )

        return split_data