import torch


class RLModel(torch.nn.Module):
    """
    RL model for exploring simulation parameter space
    """

    def __init__(self, state_dim: int, action_dim: int):
        super(RLModel, self).__init__()
        # Define the architecture of the RL model
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)
