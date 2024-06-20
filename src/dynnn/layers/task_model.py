import time
import torch
from torch import nn
from typing import Callable, Literal

from dynnn.layers import ParameterSearchModel, PINN
from dynnn.train import train_pinn, train_simulator, train_task_model
from dynnn.types import (
    Dataset,
    PinnModelArgs,
    ModelStats,
    PinnTrainingArgs,
    SaveableModel,
    SimulatorArgs,
    SimulatorState,
    SimulatorTrainingArgs,
    TrainingArgs,
    TransformY,
)


class TaskModel(SaveableModel):
    """
    Task model with inner PINN and an outer canonical task
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        initial_sim_args: SimulatorArgs | None = None,
        pinn_args: PinnModelArgs = PinnModelArgs(),
    ):
        self.run_id = time.time()
        super(TaskModel, self).__init__("task", self.run_id)

        initial_state = SimulatorState(params=initial_sim_args or SimulatorArgs())
        self.initial_state = initial_state

        self.pinn = PINN(self.run_id, pinn_args)
        self.param_model = ParameterSearchModel(
            self.run_id,
            state_dim=initial_state.num_rl_params,
            output_ranges=initial_state.rl_param_sizes_flat,
        )
        self.task_input_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.task_output_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass for the task model.

        x size: batch_size, n_timepoints x n_bodies x len([q, p]) x n_dims
        """
        inputs = self.task_input_model(x)
        pinn_output = self.pinn(inputs, t)
        return self.task_output_model(pinn_output)

    def _get_training_loop(
        self, canonical_dataset: Dataset, transform_y: TransformY | None = None
    ) -> Callable[[PinnTrainingArgs, Dataset], ModelStats]:
        """
        Training loop for the task (trains inner pinn and outer model)
        """

        def training_loop(args: PinnTrainingArgs, data: Dataset):
            stats = train_pinn(self.pinn, args, data)
            return train_task_model(self, args, canonical_dataset, transform_y)

        return training_loop

    def train(
        self,
        args: SimulatorTrainingArgs,
        canonical_dataset: Dataset,
        transform_y: TransformY | None = None,
    ):
        """
        Training loop for the simulator (RL parameter search + PINN)
        """
        training_loop = self._get_training_loop(canonical_dataset, transform_y)
        return train_simulator(
            self.param_model, args, self.initial_state, train_loop=training_loop
        )
