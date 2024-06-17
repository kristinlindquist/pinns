from functools import partial
import math
from multimethod import multidispatch
from pydantic import BaseModel
import time
import torch
import torch.autograd.functional as AF
from typing import Any, Callable, overload
import uuid

from dynnn.mechanics.lagrangian import lagrangian_equation_of_motion as lagrangian_eom
from dynnn.mechanics.hamiltonian import (
    hamiltonian_equation_of_motion as hamiltonian_eom,
)
from dynnn.types import (
    DatasetArgs,
    GeneratorFunction,
    GeneratorType,
    OdeSolverType,
    Trajectory,
    TrajectoryArgs,
)
from dynnn.utils import load_or_create_data


from .utils import get_timepoints

MAX_NAN_STEPS = 3
TRAJ_CHECK_STEPS = 500
TRAJ_MIN_PROGRESS = 1e-3
MAX_TRAJ_FAILS = 3


class Mechanics:
    """
    For modeling physical systems based on equations of motion.
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
            get_initial_conditions: function returning initial conditions
            domain_min (int): minimum domain value
            domain_max (int): maximum domain value
        """
        self.get_generator_fn = get_generator_fn
        self.get_initial_conditions = get_initial_conditions
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.log = {}

    def track_trajectory(self, traj_id: str, t: torch.Tensor | None) -> None:
        """
        Track trajectory progress (logging & detection that we're going nowhere)
        """
        # t == None means we're not called by ode_solver
        if t is None:
            return

        self.log[traj_id].append(t)
        total_steps = len(self.log[traj_id])
        if total_steps % TRAJ_CHECK_STEPS == 0:
            print(f"Trajectory {traj_id}: {total_steps} steps (last t: {t.item()})")

            if total_steps >= TRAJ_CHECK_STEPS * 2:
                # if the timestep is still very tiny, assume we're stuck
                progress = self.log[traj_id][-1] - self.log[traj_id][-TRAJ_CHECK_STEPS]
                if progress < TRAJ_MIN_PROGRESS:
                    print(
                        f"Trajectory {traj_id}: Not making progress ({progress}); giving up"
                    )
                    raise ValueError("Not making progress")

        # if too many values are NaNs, assume parameter set is invalid
        if len([t for t in self.log[traj_id] if math.isnan(t)]) > MAX_NAN_STEPS:
            print(f"Trajectory {traj_id}: Too many NaNs; giving up")
            raise ValueError("Too many NaNs")

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
            generator_type: type of generator (Hamiltonian or Lagrangian)
            model: model to use for time derivative
            function_args: additional arguments for the function
            traj_id: trajectory ID for logging

        Returns:
            time derivative of the phase space coordinates (n_bodies x 2 x n_dims)
            based on the specified generator function & equations of motion
        """
        # track trajectory, including logging and early stopping
        self.track_trajectory(traj_id, t)

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
        # generate initial conditions (y0/masses) if none provided
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

        # use an ODE solver to solve the equations of motion
        t = get_timepoints(args.t_span_min, args.t_span_max, args.time_scale)
        trajectory = args.odeint_solver.solve(
            f=dynamics_fn,
            x=y0,
            t_span=t,
            solver=args.odeint_solver.name.lower(),
            rtol=args.odeint_rtol,
            atol=args.odeint_atol,
        )[1]

        # clear log (only needed to track within odeint_solver)
        self.log[traj_id] = None

        # -> time_scale*t_span_max x n_bodies x 2 x n_dims
        # for each timepoint, get the time derivative of the phase space coordinates
        dsdt = torch.stack([dynamics_fn(None, st) for st in trajectory])

        # -> time_scale*t_span_max x n_bodies x n_dims
        dqdt, dpdt = [d.squeeze() for d in torch.split(dsdt, 1, dim=2)]

        # -> time_scale*t_span_max x n_bodies x 2
        q, p = trajectory[:, :, 0], trajectory[:, :, 1]

        return Trajectory(q=q, p=p, dq=dqdt, dp=dpdt, t=t, masses=masses)

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
        args.n_samples: Number of samples
        args.test_split: Test split
        trajectory_args: Additional arguments for the trajectory function
        """
        start = time.time()
        filename_parts = [
            args.filename,
            trajectory_args.filename,
        ]

        data_file = f"mve_data-{'-'.join(filename_parts)}.pkl"
        data = load_or_create_data(
            data_file, lambda: self._get_dataset(args, trajectory_args)
        )

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
        args.n_samples: Number of samples
        args.test_split: Test split
        trajectory_args: Additional arguments for the trajectory function
        """

        n_samples, test_split = args.dict().values()

        torch.seed()
        xs, dxs, time = [], [], None
        fail_count = 0
        count = 0
        while count < n_samples:
            # try to get a trajectory
            try:
                q, p, dq, dp, t, masses = (
                    self.get_trajectory(trajectory_args).dict().values()
                )
                count += 1
            except Exception as e:
                # if we fail too many times, bubble up the exception
                if fail_count >= MAX_TRAJ_FAILS:
                    raise e
                # otherwise, hope it is transient. try again.
                fail_count += 1
                continue

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
        split_data = {"masses": masses, "time": time}
        for k in ["x", "dx"]:
            split_data[k], split_data["test_" + k] = (
                data[k][:split_ix],
                data[k][split_ix:],
            )

        return split_data
