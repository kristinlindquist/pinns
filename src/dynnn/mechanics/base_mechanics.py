import concurrent.futures
from functools import partial
import logging
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
    Dataset,
    DatasetArgs,
    GeneratorFunction,
    GeneratorType,
    OdeSolverType,
    Trajectory,
    TrajectoryArgs,
)
from dynnn.utils import load_or_create_data, get_logger

from .utils import get_timepoints

logger = get_logger(__name__)

MAX_NAN_STEPS = 10
TRAJ_CHECK_STEPS = 500
TRAJ_MIN_PROGRESS = 1e-3
MAX_FAILS_PER_TRAJ = 3


class Mechanics:
    """
    For modeling physical systems based on equations of motion.
    """

    def __init__(
        self,
        get_generator_fn: Callable[[Any], GeneratorFunction],
        get_initial_conditions: Callable[[int, Any], tuple[torch.Tensor, torch.Tensor]],
    ):
        """
        Initialize the class

        Args:
            get_generator_fn: function returning Hamiltonian function
            get_initial_conditions: function returning initial conditions
        """
        self.get_generator_fn = get_generator_fn
        self.get_initial_conditions = get_initial_conditions
        self.log = {}

    def track_trajectory(
        self, traj_id: str, t: torch.Tensor | None, min_progress: int
    ) -> None:
        """
        Track trajectory progress (logging & detection that we're going nowhere)
        """
        # t == None means we're not called by ode_solver
        if t is None:
            return

        self.log[traj_id].append(t)
        total_steps = len(self.log[traj_id])
        if total_steps % TRAJ_CHECK_STEPS == 0:
            logger.debug(
                "Trajectory %s: %s steps (last t: %s)", traj_id, total_steps, t.item()
            )

            if total_steps >= (TRAJ_CHECK_STEPS * 2):
                # if the timestep is still very tiny, assume we're stuck
                progress = self.log[traj_id][-1] - self.log[traj_id][-TRAJ_CHECK_STEPS]
                if progress < min_progress:
                    logger.warn(
                        "Trajectory %s: Stalled (%s); giving up",
                        traj_id,
                        progress.item(),
                    )
                    raise RuntimeError("Trajectory stalled")

        # if too many values are NaNs, assume parameter set is invalid
        total_nans = len([t for t in self.log[traj_id] if torch.isnan(t)])
        if total_nans > MAX_NAN_STEPS:
            logger.warn(
                "Trajectory %s: Too many NaNs (%s); giving up", traj_id, total_nans
            )
            raise ValueError("Too many NaNs")

    def dynamics_fn(
        self,
        t: torch.Tensor,
        ps_coords: torch.Tensor,
        generator_type: GeneratorType,
        model: torch.nn.Module | None = None,
        function_args: dict = {},
        traj_id: str = "",
        min_progress: int = TRAJ_MIN_PROGRESS,
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
        self.track_trajectory(traj_id, t, min_progress)

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
            args.odeint_rtol: ode solver relative tolerance
            args.odeint_atol: ode solver absolute tolerance
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
            min_progress=TRAJ_MIN_PROGRESS / args.n_bodies,
        )
        dynamics_fn.order = args.odeint_order

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
    ) -> tuple[Dataset, int]:
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

    def get_trajectory_data(
        self, args: TrajectoryArgs, fail_count: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get trajectory data
        """
        try:
            q, p, dq, dp, t, masses = self.get_trajectory(args).dict().values()
        except Exception as e:
            if fail_count >= MAX_FAILS_PER_TRAJ:
                raise e
            else:
                return self.get_trajectory_data(args, fail_count + 1)

        # (time_scale*t_span_max) x n_bodies x 2 x n_dims
        x = torch.stack([q, p], dim=2).unsqueeze(dim=0)
        dx = torch.stack([dq, dp], dim=2).unsqueeze(dim=0)
        return x, dx, t

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

        torch.seed()
        n_samples, test_split = args.dict().values()
        xs, dxs, time = [], [], None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.get_trajectory_data, trajectory_args)
                for _ in range(n_samples)
            ]

            for future in concurrent.futures.as_completed(futures):
                x, dx, t = future.result()
                xs.append(x)
                dxs.append(dx)
                time = t
                logger.info("Data traj: %s of %s", len(results), n_samples)

        # batch_size x (time_scale*t_span_max) x n_bodies x 2 x n_dims
        data = {"x": torch.cat(xs), "dx": torch.cat(dxs)}

        # make a train/test split
        split_ix = int(len(data["x"]) * test_split)
        split_data = {"masses": masses, "time": time}
        for k in ["x", "dx"]:
            split_data[k], split_data["test_" + k] = (
                data[k][:split_ix],
                data[k][split_ix:],
            )

        return Dataset(**split_data)
