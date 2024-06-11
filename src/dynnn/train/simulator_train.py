from typing import Any, Callable
import torch
from pydantic import BaseModel
import statistics as math

from dynnn.layers.simulator import SimulatorModel
from dynnn.layers.pinn import PINN
from dynnn.layers.encoder import encode_params, flatten_dict, unflatten_params
from dynnn.simulation.mve_ensemble import MveEnsembleMechanics, get_initial_conditions
from dynnn.train.pinn_train import pinn_train
from dynnn.types import (
    GeneratorType,
    ModelArgs,
    OdeSolver,
    ParameterLossError,
    PinnStats,
    SimulatorParams,
)


class SimulatorState(BaseModel):
    params: SimulatorParams
    stats: PinnStats = PinnStats()
    sim_duration: float = 0.0

    @property
    def rl_param_sizes(self) -> dict[str, dict[str, Any]]:
        return {
            "params": self.params.rl_param_sizes,
            "stats": 2,
            "sim_duration": 2,
        }

    @property
    def rl_param_sizes_flat(self) -> dict[str, tuple[int, int]]:
        return flatten_dict(self.rl_param_sizes)

    @property
    def num_rl_params(self) -> int:
        return len(self.rl_param_sizes_flat)

    def encode_rl_params(self) -> tuple[torch.Tensor, dict]:
        attributes = {
            "params": self.params.encode_rl_params(),
            "stats": self.stats.encode(),
            "sim_duration": self.sim_duration,
        }

        return encode_params(flatten_dict(attributes)), attributes

    @classmethod
    def load_rl_params(cls, encoded: dict, template: dict) -> "SimulatorParams":
        scalar_dict = unflatten_params(encoded, template, decode_tensors=True)
        return cls(
            params=SimulatorParams.load_rl_params(scalar_dict["params"]),
            stats=scalar_dict.get("stats", {}),
            sim_duration=scalar_dict.get("sim_duration", 0.0),
        )


def get_initial_params(args: dict) -> SimulatorParams:
    y0, masses = get_initial_conditions(args.n_bodies, args.n_dims)
    return SimulatorParams(
        dataset_args={
            "n_samples": 2,
        },
        trajectory_args={
            # "y0": y0,
            # "masses": masses,
            "n_bodies": args.n_bodies,
            "time_scale": 3,
            "t_span_min": 0,
            "t_span_max": 4,
            "odeint_rtol": 1e-10,
            "odeint_atol": 1e-6,
        },
        model_args={
            "domain_min": 0,
            "domain_max": 10,
        },
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

    def step(self, action: SimulatorState) -> tuple[SimulatorState, float, bool]:
        # Apply the action (parameter configuration) to the simulator
        new_state = self.simulate(action)

        # Compute the reward based on the state transition
        reward = self.reward(self.current_state, new_state)

        self.current_state = new_state
        self.current_step += 1

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
) -> tuple[torch.nn.Module, dict]:
    """
    Training loop for the RL model
    """
    initial_state = SimulatorState(params=get_initial_params(args))
    sbn = SimulatorModel(
        state_dim=initial_state.num_rl_params,
        output_ranges=initial_state.rl_param_sizes_flat,
    )
    pinn = PINN(
        (args.n_bodies, 2, args.n_dims), args.hidden_dim, field_type=args.field_type
    )

    env = SimulatorEnv(args, initial_state, pinn)
    optimizer = torch.optim.Adam(
        sbn.parameters(), lr=args.learn_rate, weight_decay=args.weight_decay
    )

    for experiment in range(args.num_experiments):
        state = env.reset()
        experiment_reward = []

        for step in range(args.max_simulator_steps):
            state_tensor, state_dict = state.encode_rl_params()
            action, distribution = sbn(state_tensor)

            valid_action = SimulatorState.load_rl_params(action, state_dict)
            next_state, reward, is_done = env.step(valid_action)

            loss = -reward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sbn.parameters(), max_norm=1.0)
            optimizer.step()

            experiment_reward.append(reward.item())
            state = next_state

            if is_done:
                break

        print(
            f"Experiment {experiment + 1}: Reward = {math.mean(experiment_reward):.2f}"
        )

    return None, None
