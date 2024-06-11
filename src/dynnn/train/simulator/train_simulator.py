from typing import Callable
import torch
import statistics as math

from dynnn.layers.parameter_search import ParameterSearchModel
from dynnn.layers.pinn import PINN
from dynnn.train.train_pinn import train_pinn
from dynnn.types import SimulatorParams

from .environment import SimulatorEnv
from .types import SimulatorState


def train_simulator(
    args: dict,
    canonical_data: dict = {},
    plot_loss_callback: Callable | None = None,
    pinn_loss_fn: Callable | None = None,
) -> tuple[torch.nn.Module, dict]:
    """
    Training loop to learn the simulator.
    Explore parameter space for the creation and learning of dynamical system.
    """
    # Initialize the simulator

    initial_state = SimulatorState(params=SimulatorParams())

    # Initialize the simulator model
    psm = ParameterSearchModel(
        state_dim=initial_state.num_rl_params,
        output_ranges=initial_state.rl_param_sizes_flat,
    )

    # Initialize the PINN model
    pinn = PINN(
        (100, 2, initial_state.params.trajectory_args.n_dims),
        initial_state.params.model_args,
    )

    # Initialize the rl environment
    env = SimulatorEnv(initial_state, pinn, pinn_loss_fn=pinn_loss_fn)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        psm.parameters(),
        lr=args.rl_learn_rate,
        weight_decay=args.rl_weight_decay,
    )

    for experiment in range(args.num_experiments):
        state = env.reset()
        experiment_reward = []

        for step in range(args.max_simulator_steps):
            optimizer.zero_grad()
            state_tensor, state_dict = state.encode_rl_params()

            # Get the action from the state
            action, unscaled_action, distribution = psm(state_tensor)

            # validate the action
            valid_action = SimulatorState.load_rl_params(action, state_dict)

            # Apply the action to the simulator
            next_state, reward = env.step(valid_action)

            # Compute the loss from the policy gradient
            log_prob = distribution.log_prob(unscaled_action)
            loss = -log_prob * reward
            loss.backward()

            torch.nn.utils.clip_grad_norm_(psm.parameters(), max_norm=1.0)
            optimizer.step()

            experiment_reward.append(reward.item())
            state = next_state

        print(
            f"Experiment {experiment + 1}: Reward = {math.mean(experiment_reward):.2f}"
        )

    return None, None
