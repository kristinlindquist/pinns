from torch import nn
import torch


class DynamicallySizedNetwork(nn.Module):
    """
    An MLP with input/output layers handling the range of potential sizes (`dynamic_range`)
    of a given dimension (`dynamic_dim`).

    E.g. for PINNs learning over a range of n_bodies values
    """

    def __init__(
        self,
        canonical_input_dim: int,
        hidden_dim: int,
        canonical_output_dim: int,
        dynamic_dim: int,
        dynamic_range: tuple[int, int, int],  # (min, max, step)
        dynamic_multiplier: int,
    ):
        super(DynamicallySizedNetwork, self).__init__()
        self.dynamic_range = dynamic_range
        self.dynamic_dim = dynamic_dim

        # input/output layers for each permitted size of the dynamic dimension
        self.input_layers = nn.ModuleList(
            [
                nn.Linear(dynamic_input_size * dynamic_multiplier, canonical_input_dim)
                for dynamic_input_size in range(*dynamic_range)
            ]
        )

        # core canonical model
        self.canonical_model = nn.Sequential(
            nn.Linear(canonical_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, canonical_output_dim),
        )

        # output layers for each permitted size of the dynamic dimension
        self.output_layers = nn.ModuleList(
            [
                nn.Linear(
                    canonical_output_dim, dynamic_output_size * dynamic_multiplier
                )
                for dynamic_output_size in range(*dynamic_range)
            ]
        )

    def get_dynamic_index(self, x: torch.Tensor) -> int:
        """
        Get the index for the dynamic layer
        (size of the dynamic layer minus the minimum value)
        """
        dynamic_dim_size = x.shape[self.dynamic_dim]
        return dynamic_dim_size - self.dynamic_range[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dynamically sized network.

        - feed x through the dynamic input layer corresponding to the size of the dynamic dimension
            (e.g. 10 if n_bodies=10)
        - send the output through the canonical model
        - feed the output through the dynamic output layer corresponding to the size of the dynamic dimension

        Args:
            x (torch.Tensor): input tensor (batch_size, (time_scale*t_span_max) x n_bodies x len([q, p]) x n_dims)
        """
        # get the index for the dynamic layer
        dynamic_index = self.get_dynamic_index(x)

        # find the input/output layers for that index
        input_layer = self.input_layers[dynamic_index]
        output_layer = self.output_layers[dynamic_index]

        input = x.reshape(x.shape[0], x.shape[1], -1)

        canonical_input = input_layer(input)
        canonical_output = self.canonical_model(canonical_input)
        dynamic_output = output_layer(canonical_output)

        return dynamic_output.reshape(x.shape)
