import statistics as math
import torch
from torch import nn

from dynnn.types import (
    SaveableModel,
    SimulatorState,
    SimulatorTrainingArgs,
    TrainLoop,
)
from dynnn.utils import get_logger

from .environment import SimulatorEnv

logger = get_logger(__name__)


def train_simulator(
    param_model: SaveableModel,
    args: SimulatorTrainingArgs,
    initial_state: SimulatorState,
    train_loop: TrainLoop,
):
    """
    Training loop to learn the simulator.
    Explore parameter space for the creation and learning of dynamical system.
    """
    # Initialize the rl environment
    # TODO: move up to task_model?
    env = SimulatorEnv(initial_state, train_loop)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        param_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    for experiment in range(args.num_experiments):
        state = env.reset()
        experiment_reward = []
        state_sequence = []

        for step in range(args.max_simulator_steps):
            optimizer.zero_grad()
            state_tensor, state_dict = state.encode_rl_params()

            state_sequence.append(state_tensor)

            # Get the action from the state
            action, unscaled_action, distribution = param_model(
                state_tensor, torch.stack(state_sequence)
            )

            # validate the action
            valid_action = SimulatorState.load_rl_params(action, state_dict)

            # Apply the action to the simulator
            next_state, reward = env.step(valid_action)

            # Compute the loss from the policy gradient
            log_prob = distribution.log_prob(unscaled_action)
            loss = -log_prob * reward
            logger.info(
                f"RL Loss: {loss.item()} (Reward: {reward.item()}, log_prob: {log_prob})"
            )
            loss.backward()

            nn.utils.clip_grad_norm_(param_model.parameters(), max_norm=1.0)
            optimizer.step()

            experiment_reward.append(reward.item())
            state = next_state

        param_model.save()

        logger.info(
            f"Experiment {experiment + 1}: Reward = {math.mean(experiment_reward):.2f}"
        )
