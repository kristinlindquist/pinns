import os, pickle
import time
from typing import Any, Callable, overload
import torch
from functools import partial
from torchdyn.numerics.odeint import odeint, odeint_symplectic
import torch.autograd.functional as AF
from pydantic import BaseModel
from multimethod import multidispatch, multimethod
import uuid

from dynnn.mechanics.lagrangian import lagrangian_equation_of_motion as lagrangian_eom
from dynnn.mechanics.hamiltonian import (
    hamiltonian_equation_of_motion as hamiltonian_eom,
)
from dynnn.types import (
    DatasetArgs,
    GeneratorFunction,
    GeneratorType,
    OdeSolver,
    Trajectory,
    TrajectoryArgs,
)
from dynnn.utils import get_timepoints


class Mechanics:
    """
    Classical mechanics system class
    """

    def __init__(
        self,
        get_generator_fn: Callable[[Any], GeneratorFunction],
        get_initial_conditions: Callable[[int, Any], tuple[torch.Tensor, torch.Tensor]],
        domain_min: int = 0,
        domain_max: int = 10,
    ):
        """
        Initialize the class

        Args:
            get_generator_fn: function returning Hamiltonian function
            domain_min (int): minimum domain value
            domain_max (int): maximum domain value
        """
        self.get_generator_fn = get_generator_fn
        self.get_initial_conditions = get_initial_conditions
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.log = {}

    def dynamics_fn(
        self,
        t: torch.Tensor,
        ps_coords: torch.Tensor,
        generator_type: GeneratorType,
        model: torch.nn.Module | None = None,
        function_args: dict = {},
        traj_id: str = "",
    ) -> torch.Tensor:
        """
        Dynamics function - finds the state update for the supplied function

        Args:
            t: Time
            ps_coords: phase space coordinates (n_bodies x 2 x n_dims)
            model: model to use for time derivative
            function_args: additional arguments for the function
        """
        if isNaN(t):
            raise ValueError("t is NaN")

        if traj_id in self.log:
            self.log[traj_id].append(t)
            if len(self.log[traj_id]) % 500 == 0:
                print(
                    f"Trajectory {traj_id}: {len(self.log[traj_id])} steps (last t: {t})"
                )

        generator_fn = self.get_generator_fn(
            **function_args, generator_type=generator_type
        )
        eom_fn = (
            lagrangian_eom
            if generator_type == GeneratorType.LAGRANGIAN
            else hamiltonian_eom
        )

        return eom_fn(generator_fn, t, ps_coords, model)

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
            args.n_bodies: number of bodies (optional if y0 is provided)
            args.n_dims: number of dimensions (optional, default=3)
            args.y0: Initial conditions (optional; must be provided with masses)
            args.masses: Masses of the bodies (optional; must be provided with y0)
            args.t_span_min: Minimum time span
            args.t_span_max: Maximum time span
            args.odeint_solver: ODE solver
            args.odeint_rtol: Relative tolerance
            args.odeint_atol: Absolute tolerance
            args.generator_type: Generator type
            args.time_scale: Time scale
            args.model: Model to use for time derivative (optional)
        """
        if args.y0 is None:
            y0, masses = self.get_initial_conditions(args.n_bodies, masses=args.masses)
        elif args.masses is not None:
            y0, masses = args.y0, args.masses

        traj_id = uuid.uuid4().hex
        self.log[traj_id] = []

        dynamics_fn = partial(
            self.dynamics_fn,
            model=args.model,
            generator_type=args.generator_type,
            function_args={"masses": masses},
            traj_id=traj_id,
        )

        t = get_timepoints(args.t_span_min, args.t_span_max, args.time_scale)

        solve_ivp = (
            odeint_symplectic if args.odeint_solver == OdeSolver.SYMPLECTIC else odeint
        )
        ivp = solve_ivp(
            f=dynamics_fn,
            x=y0,
            t_span=t,
            solver=args.odeint_solver.name.lower(),
            rtol=args.odeint_rtol,
            atol=args.odeint_atol,
        )[1]

        # -> time_scale*t_span_max x n_bodies x 2 x n_dims
        dsdt = torch.stack([dynamics_fn(None, dp) for dp in ivp])
        # -> time_scale*t_span_max x n_bodies x n_dims
        dqdt, dpdt = [d.squeeze() for d in torch.split(dsdt, 1, dim=2)]

        # -> time_scale*t_span_max x n_bodies x 2
        q, p = ivp[:, :, 0], ivp[:, :, 1]

        self.log[traj_id] = None

        return Trajectory(q=q, p=p, dq=dqdt, dp=dpdt, t=t)

    @multidispatch
    def get_dataset(self, args, trajectory_args) -> tuple[dict, int]:
        return NotImplemented

    @overload
    @get_dataset.register
    def _(self, args: dict = {}, trajectory_args: dict = {}) -> tuple[dict, int]:
        return self.get_dataset(DatasetArgs(**args), TrajectoryArgs(**trajectory_args))

    @overload
    @get_dataset.register
    def _(
        self,
        args: DatasetArgs,
        trajectory_args: TrajectoryArgs,
    ) -> tuple[dict, int]:
        """
        Generate a dataset of trajectories
        (with pickle caching)

        Args:
        args.num_samples: Number of samples
        args.test_split: Test split
        trajectory_args: Additional arguments for the trajectory function
        """
        start = time.time()
        filename_parts = [
            args.filename,
            trajectory_args.filename,
        ]

        pickle_path = f"mve_data-{'-'.join(filename_parts)}.pkl"

        if os.path.exists(pickle_path):
            print(f"Loading data from {pickle_path}")
            with open(pickle_path, "rb") as file:
                data = pickle.loads(file.read())
        else:
            print(f"Creating data... ({pickle_path})")
            data = self._get_dataset(args, trajectory_args)
            print(f"Saving data to {pickle_path}")
            with open(pickle_path, "wb") as file:
                pickle.dump(data, file)

        runtime = time.time() - start

        return data, runtime

    def _get_dataset(
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
            q, p, dq, dp, t = self.get_trajectory(trajectory_args).dict().values()

            if time is None:
                time = t

            # (time_scale*t_span_max) x n_bodies x 2 x n_dims
            xs.append(torch.stack([q, p], dim=2).unsqueeze(dim=0))

            # (time_scale*t_span_max) x n_bodies x 2 x n_dims
            dxs.append(torch.stack([dq, dp], dim=2).unsqueeze(dim=0))

        # batch_size x (time_scale*t_span_max) x n_bodies x 2 x n_dims
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
