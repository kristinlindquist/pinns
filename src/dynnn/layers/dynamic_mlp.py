import math
from torch import nn
import torch
from typing import Literal


class DynamicallySizedNetwork(nn.Module):
    """
    An MLP with input/output layers created for each input supplied, based on the size of a given dimension (`dynamic_dim`).

    E.g. for PINNs learning over a range of n_bodies values
    """

    def __init__(
        self,
        canonical_input_dim: int,
        hidden_dim: int,
        canonical_output_dim: int,
        dynamic_dim: int,
        dynamic_multiplier: int,
        extra_canonical_output_layers: list[torch.Tensor] = [],
    ):
        super(DynamicallySizedNetwork, self).__init__()
        self.dynamic_dim = dynamic_dim
        self.dynamic_multiplier = dynamic_multiplier
        self.canonical_input_dim = canonical_input_dim

        # input/output layers for each permitted size of the dynamic dimension
        self.input_layers = nn.ModuleDict({})
        self.output_layers = nn.ModuleDict({})

        # core canonical model
        self.canonical_model = nn.Sequential(
            nn.Linear(canonical_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            *extra_canonical_output_layers,
            nn.Linear(hidden_dim, canonical_output_dim),
        )

    def get_or_create_layer(
        self, dynamic_index: int, layer_type: Literal["input", "output"]
    ):
        """
        Get the dynamically sized layer for the given input tensor
        """
        layer_map = self.input_layers if layer_type == "input" else self.output_layers

        if dynamic_index in layer_map:
            return layer_map[str(dynamic_index)]

        # specify dimensions for input/output layers
        dims = [dynamic_index * self.dynamic_multiplier, self.canonical_input_dim]
        if layer_type == "output":
            dims = sorted(dims, reverse=True)

        layer = nn.Linear(*dims)
        layer_map[str(dynamic_index)] = layer

        return layer

    def forward(
        self, x: torch.Tensor, skip_dynamic_layers: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the dynamically sized network.

        - feed x through the dynamic input layer corresponding to the size of the dynamic dimension
            (e.g. 10 if n_bodies=10)
        - send the output through the canonical model
        - feed the output through the dynamic output layer corresponding to the size of the dynamic dimension

        Args:
            x (torch.Tensor): input tensor (batch_size, (time_scale*t_span_max) x n_bodies x len([q, p]) x n_dims)
            skip_dynamic_layers (bool): whether to skip the dynamic layers and just use the canonical model
                                        (used for outer problem / task model training)
        """
        if skip_dynamic_layers:
            return self.canonical_model(x)

        # get the index for the dynamic layer
        dynamic_index = x.shape[self.dynamic_dim]

        # find the input/output layers for that index
        input_layer = self.get_or_create_layer(dynamic_index, "input")
        output_layer = self.get_or_create_layer(dynamic_index, "output")

        inputs = x.reshape(*x.shape[0:2], -1)
        canonical_input = input_layer(inputs)
        canonical_output = self.canonical_model(canonical_input)

        return output_layer(canonical_output).reshape(x.shape)
