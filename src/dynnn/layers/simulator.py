import torch


class SimulatorModel(torch.nn.Module):
    """
    RL model for exploring simulation parameter space
    """

    def __init__(self, state_dim: int, action_dim: int, possible_params: dict):
        super(SimulatorModel, self).__init__()
        self.possible_params = possible_params
        # Define the architecture of the RL model
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)
