import torch

from dynnn.simulation.mve_ensemble import MveEnsembleMechanics
from dynnn.types import ModelStats, SimulatorState, TrainLoop
from dynnn.utils import get_logger


logger = get_logger(__name__)


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
        train_loop: TrainLoop,
    ):
        self.initial_state = initial_state
        self.current_state = None
        self.train_loop = train_loop

    def reset(self) -> SimulatorState:
        self.current_state = self.initial_state
        return self.current_state

    def step(self, action: SimulatorState) -> tuple[SimulatorState, float]:
        # Apply the action (parameter configuration) to the simulator
        new_state = self.simulate(action)

        # Compute the reward based on the state transition
        reward = self.reward(self.current_state, new_state)

        self.current_state = new_state

        return new_state, reward

    def simulate(self, action: SimulatorState) -> SimulatorState:
        p = action.params
        mechanics = MveEnsembleMechanics(p.mechanics_args)

        try:
            # generate EOM dataset
            data, sim_duration = mechanics.get_dataset(
                p.dataset_args, p.trajectory_args
            )
            stats = self.train_loop(p.training_args, data)
        except Exception as e:
            logger.error("Failed to generate dataset: %s", e)
            # inf loss due to error
            stats = ModelStats(
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

        logger.info(
            f"Reward: {var_loss_reduction.item()}, (Runtime: {runtime_penalty})"
        )

        return var_loss_reduction + runtime_penalty  # + canonical_loss_reduction
