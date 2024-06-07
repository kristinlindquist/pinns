import torch


class SimulatorModel(torch.nn.Module):
    """
    RL model for exploring simulation parameter space
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(SimulatorModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)
