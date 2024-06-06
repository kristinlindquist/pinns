from typing import Callable
import torch
from pydantic import BaseModel
import statistics as math

from dynnn.layers.simulator import SimulatorModel
from dynnn.layers.pinn import PINN
from dynnn.layers.encoder import decode_params, flatten_dict
from dynnn.simulation.mve_ensemble import MveEnsembleMechanics
from dynnn.train.dnn_train import dnn_train
from dynnn.types import ModelArgs

POSSIBLE_PARAMS = {
    "dataset_args": {
        "num_samples": 1,
        "test_split": 0.8,
    },
    "trajectory_args": {
        "y0": y0,
        "masses": masses,
        "time_scale": 3,
        "odeint_rtol": 1e-10,
        "odeint_atol": 1e-6,
        "odeint_solver": "tsit5",
    },
    "model_args": {
        "domain": (0, 10),
        "t_span": (0, 100),
        "generator_type": "hamiltonian",
    },
}


class SimulatorState(BaseModel):
    params: dict = POSSIBLE_PARAMS
    stats: PinnStats = PinnStats()
    simulation_duration: float


class SimulatorEnv:
    def __init__(
        self,
        initial_state: SimulatorState,
        pinn: torch.nn.Module,
        max_steps: int = 1000,
    ):
        self.initial_state = initial_state
        self.pinn = pinn
        self.max_steps = max_steps
        self.current_state = None
        self.current_step = 0
        self.prior_dist = None

    def reset(self) -> SimulatorState:
        self.current_state = self.initial_state
        self.current_step = 0
        return self.current_state

    def step(self, action: torch.Tensor) -> tuple[SimulatorState, float, bool]:
        # Apply the action (parameter configuration) to the simulator
        new_state = self.simulate_and_learn(action)

        # Compute the reward based on the state transition
        reward = self.reward(self.current_state, new_state)

        # Update the current state and step counter
        self.current_state = new_state
        self.current_step += 1

        # Check if the simulation has reached the maximum number of steps
        is_done = self.current_step >= self.max_steps

        return new_state, reward, is_done

    def simulate_and_learn(self, action: torch.Tensor) -> SimulatorState:
        params = {**POSSIBLE_PARAMS, **decode_params(action, POSSIBLE_PARAMS)}
        mechanics = MveEnsembleMechanics(ModelArgs(*params["model_args"]))
        data, simulation_duration = mechanics.get_dataset(
            params["dataset_args"], params["trajectory_args"]
        )
        _, stats = pinn_train(args, data, model=self.pinn)
        return SimulatorState(
            params=params,
            stats=stats,
            simulation_duration=simulation_duration,
        )

    def reward(self, old_state: SimulatorState, new_state: SimulatorState) -> float:
        """
        Reward based on:
        - Variable Objective loss
        - Canonical loss
        - Computational cost
        """
        vol_diff = old_state.stats.min_train_loss - new_state.stats.min_train_loss
        canonical_loss_diff = 0  # TODO
        sim_duration_diff = (
            old_state.simulation_duration - new_state.simulation_duration
        ) * 100

        return vol_diff + canonical_loss_diff + sim_duration_diff


def simulator_train(args: dict, data: dict, plot_loss_callback: Callable | None = None):
    """
    Training loop for the RL model
    """
    total_params = len(flatten_dict(POSSIBLE_PARAMS).keys())
    sbn = SimulatorModel(state_dim=total_params, action_dim=total_params)
    env = SimulatorEnv()

    optimizer = torch.optim.Adam(sbn.parameters(), lr=args.learn_rate)
    y0, masses = get_initial_conditions(args.n_bodies, args.n_dims)

    for experiment in range(args.num_experiments):
        state = env.reset()
        experiment_reward = 0

        for step in range(args.max_simulator_steps):
            action = sbn(state)
            next_state, reward, is_done = env.step(action)

            # Compute the loss and update the agent
            loss = -reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            experiment_reward += reward
            state = next_state

            if is_done:
                break

        print(f"experiment {experiment + 1}: Reward = {experiment_reward:.2f}")
