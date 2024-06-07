from typing import Callable
import torch
from pydantic import BaseModel
import statistics as math

from dynnn.layers.simulator import SimulatorModel
from dynnn.layers.pinn import PINN
from dynnn.layers.encoder import encode_params, flatten_dict, unflatten_params
from dynnn.simulation.mve_ensemble import MveEnsembleMechanics, get_initial_conditions
from dynnn.train.pinn_train import pinn_train
from dynnn.types import GeneratorType, ModelArgs, OdeSolver, PinnStats, SimulatorParams


def get_initial_params(args: dict) -> SimulatorParams:
    y0, masses = get_initial_conditions(args.n_bodies, args.n_dims)
    return SimulatorParams(
        dataset_args={
            "num_samples": 2,
            "test_split": 0.8,
        },
        trajectory_args={
            # "y0": y0,
            # "masses": masses,
            "n_bodies": args.n_bodies,
            "time_scale": 3,
            "t_span_min": 0,
            "t_span_max": 4,
            "generator_type": GeneratorType.HAMILTONIAN,
            # "odeint_rtol": 1e-10,
            # "odeint_atol": 1e-6,
            # "odeint_solver": OdeSolver.TSIT5,
        },
        model_args={
            "domain_min": 0,
            "domain_max": 10,
        },
    )


class SimulatorState(BaseModel):
    params: SimulatorParams
    stats: PinnStats = PinnStats()
    sim_duration: float = 0.0

    def encode(self):
        attributes = {
            "params": self.params.encode(),
            "stats": self.stats.encode(),
            "sim_duration": self.sim_duration,
        }

        return encode_params(flatten_dict(attributes))

    @classmethod
    def load(cls, encoded: dict, template_params: dict) -> "SimulatorParams":
        tensor_dict = unflatten_params(encoded, template_params)
        return cls(
            params=SimulatorParams.load(tensor_dict["params"]),
            stats=tensor_dict.get("stats", {}),
            sim_duration=tensor_dict.get("sim_duration", 0.0),
        )


class SimulatorEnv:
    def __init__(
        self,
        args: dict,
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
        self.args = args

    def reset(self) -> SimulatorState:
        self.current_state = self.initial_state
        self.current_step = 0
        return self.current_state

    def validation_penalty(params):
        # Define the compatible range for each parameter
        min_vals = torch.tensor([...])
        max_vals = torch.tensor([...])

        # Clamp the parameters within the compatible range
        clamped_params = torch.clamp(params, min_vals, max_vals)

        # Calculate the deviation from the compatible range
        deviation = torch.abs(params - clamped_params)

        # Apply a smooth penalty based on the deviation
        penalty = torch.exp(deviation.sum())

        return penalty

    def step(self, action: SimulatorState) -> tuple[SimulatorState, float, bool]:
        # Apply the action (parameter configuration) to the simulator
        new_state = self.simulate(action)

        # Compute the reward based on the state transition
        reward = self.reward(self.current_state, new_state)

        # Update the current state and step counter
        self.current_state = new_state
        self.current_step += 1

        # Check if the simulation has reached the maximum number of steps
        is_done = self.current_step >= self.max_steps

        return new_state, reward, is_done

    def simulate(self, action: SimulatorState) -> SimulatorState:
        mechanics = MveEnsembleMechanics(action.params.model_args)

        # generate EOM dataset
        data, sim_duration = mechanics.get_dataset(
            action.params.dataset_args, action.params.trajectory_args
        )
        # train model
        _, stats = pinn_train(self.args, data, model=self.pinn)
        return SimulatorState(
            params=params,
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

        var_loss_reduction = old_stats.min_train_loss - new_stats.min_train_loss
        runtime_penalty = (new_state.sim_duration - old_state.sim_duration) * 100
        # canonical_loss_reduction = 0  # TODO

        return var_loss_reduction + sim_reduction  # + canonical_loss_reduction


def simulator_train(
    args: dict, canonical_data: dict = {}, plot_loss_callback: Callable | None = None
):
    """
    Training loop for the RL model
    """
    initial_state = SimulatorState(params=get_initial_params(args))
    total_params = len(initial_state.encode())
    sbn = SimulatorModel(state_dim=total_params, action_dim=total_params)
    pinn = PINN(
        (args.n_bodies, 2, args.n_dims), args.hidden_dim, field_type=args.field_type
    )

    env = SimulatorEnv(args, initial_state, pinn)
    optimizer = torch.optim.Adam(sbn.parameters(), lr=args.learn_rate)

    for experiment in range(args.num_experiments):
        state = env.reset()
        experiment_reward = 0

        for step in range(args.max_simulator_steps):
            action = sbn(state.encode())

            valid_action = SimulatorState.load(action, initial_state.model_dump())
            next_state, reward, is_done = env.step(valid_action)

            loss = -reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            experiment_reward += reward
            state = next_state

            if is_done:
                break

        print(f"Experiment {experiment + 1}: Reward = {experiment_reward:.2f}")
