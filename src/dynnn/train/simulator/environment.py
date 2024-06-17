from pydantic import BaseModel
import statistics as math
import time
import torch
from typing import Callable

from dynnn.simulation.mve_ensemble import MveEnsembleMechanics
from dynnn.train.train_pinn import train_pinn

from .types import SimulatorState, PinnStats


class SimulatorEnv:
    """
    Environment for the simulator

    - reset: reset the environment to the initial state
    - step: apply an action (parameter configuration) to the simulator
    - simulate: simulate the environment with the given action
    - reward: compute the reward based on the state transition
    """

    def __init__(
        self,
        initial_state: SimulatorState,
        pinn: torch.nn.Module,
        pinn_loss_fn: Callable | None = None,
    ):
        self.initial_state = initial_state
        self.pinn = pinn
        self.pinn_loss_fn = pinn_loss_fn
        self.current_state = None

        self.run_id = time.time()

    def reset(self) -> SimulatorState:
        self.current_state = self.initial_state
        return self.current_state

    def step(self, action: SimulatorState) -> tuple[SimulatorState, float, bool]:
        # Apply the action (parameter configuration) to the simulator
        new_state = self.simulate(action)

        # Compute the reward based on the state transition
        reward = self.reward(self.current_state, new_state)

        self.current_state = new_state

        return new_state, reward

    def simulate(self, action: SimulatorState) -> SimulatorState:
        p = action.params
        mechanics = MveEnsembleMechanics(p.model_args)

        try:
            # generate EOM dataset
            data, sim_duration = mechanics.get_dataset(
                p.dataset_args, p.trajectory_args
            )
            _, stats = train_pinn(
                self.initial_state.params.training_args,
                data,
                run_id=self.run_id,
                model=self.pinn,
                loss_fn=self.pinn_loss_fn,
            )
        except Exception as e:
            print(f"Failed to generate dataset: {e}")
            # inf loss due to error
            stats = PinnStats(
                train_loss=[torch.tensor(1e10)],
                test_loss=[torch.tensor(1e10)],
            )
            sim_duration = 0.0

        return SimulatorState(
            params=p,
            stats=stats,
            sim_duration=sim_duration,
        )

    def reward(self, old_state: SimulatorState, new_state: SimulatorState) -> float:
        """
        Reward based on:
        - Variable Objective loss
        - Canonical loss
        - Computational cost
        """
        new_stats, old_stats = new_state.stats, old_state.stats

        var_loss_reduction = (
            old_stats.min_train_loss.detach() - new_stats.min_train_loss.detach()
        )
        runtime_penalty = (new_state.sim_duration - old_state.sim_duration) * 2000
        # canonical_loss_reduction = 0  # TODO

        print(f"Reward: {var_loss_reduction.item()}, (Runtime: {runtime_penalty})")

        return var_loss_reduction + runtime_penalty  # + canonical_loss_reduction
