from typing import Literal
import torch
import torch.autograd.functional as AF
from itertools import permutations
import math

from dynnn.layers import RotationallyInvariantLayer, TranslationallyInvariantLayer
from dynnn.utils import permutation_tensor


class MLP(torch.nn.Module):
    """
    MLP to learn the hamiltonian
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim)
        self.nonlinearity = torch.nn.Tanh()

        for layer in [self.linear1, self.linear2, self.linear3]:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

        self.module = torch.nn.Sequential(
            self.linear1,
            self.nonlinearity,
            self.linear2,
            self.nonlinearity,
            self.linear3,
        )

    def forward(self, x) -> torch.Tensor:
        return self.module(x)


class DynNN(torch.nn.Module):
    """
    Learn arbitrary vector fields that are sums of conservative and solenoidal fields

    TODO:
    - dimensionality of permutation_tensor
    """

    def __init__(
        self,
        input_dims: tuple[int, int, int],
        hidden_dim: int,
        field_type: Literal["conservative", "solenoidal", "both"] = "both",
    ):
        super(DynNN, self).__init__()
        self.input_dim = math.prod(input_dims)
        self.P = permutation_tensor()  # Levi-Civita permutation tensor
        self.M = torch.nn.Parameter(torch.randn(self.input_dim, self.input_dim))
        self.input_dims = input_dims
        self.field_type = field_type
        self.invariant_layer = torch.nn.Sequential(
            TranslationallyInvariantLayer(),
            RotationallyInvariantLayer(),
        )
        self.use_invariant_layer = False

        self.model = MLP(
            (
                RotationallyInvariantLayer.get_output_dim(input_dims[0])
                if self.use_invariant_layer
                else self.input_dim
            ),
            hidden_dim,
            self.input_dim,
        )

        # a smooth, rapidly decaying 3d vector field can be decomposed into a conservative and solenoidal field
        # https://en.wikipedia.org/wiki/Helmholtz_decomposition
        if field_type != "both":
            print(
                f"Warning: a field_type of {field_type} might not capture the full dynamics of the system."
            )

    # impose the system matrix to be skew symmetric
    def skew(self):
        return 0.5 * (self.M - self.M.T)

    def forward(self, x: torch.Tensor, t=None) -> torch.Tensor:
        """
        Neural Hamiltonian-style vector field

        x size: batch_size, (time_scale*t_span[1]) x n_bodies x len([q, p]) x n_dims
        """
        if self.use_invariant_layer:
            invariant_features = self.invariant_layer(x)
            potentials = self.model(invariant_features).reshape(*x.shape)
        else:
            potentials = self.model(x.reshape(*x.shape[:-2], -1)).reshape(*x.shape)

        scalar_potential, vector_potential = torch.split(potentials, 1, dim=-2)

        if self.field_type == "none":
            return potentials

        if self.field_type == "port":
            # skew invariant
            d_potential = torch.autograd.grad(potentials.sum(), x, create_graph=True)[0]
            return torch.einsum(
                "bti,ij->btj",
                d_potential.reshape(*d_potential.shape[0:2], -1),
                self.skew(),
            ).reshape(*d_potential.shape)

        # start out with both components set to 0
        conservative_field = torch.zeros_like(x)
        solenoidal_field = torch.zeros_like(x)

        if self.field_type in ["both", "conservative"]:
            """
            conservative: models energy-conserving physical systems; irrotational (vanishing curl)
            """
            # batch_size, (time_scale*t_span[1]) x n_bodies x (len([r, v]) * n_dims)
            d_scalar_potential = torch.autograd.grad(
                scalar_potential.sum(),
                x,
                create_graph=True,
            )[0]
            conservative_field = d_scalar_potential

        if self.field_type in ["both", "solenoidal"]:
            """
            solenoidal: a vector field with zero divergence (aka no sources or sinks)
            """
            d_vector_potential = torch.autograd.grad(
                vector_potential.sum(),
                x,
                create_graph=True,
            )[0]
            solenoidal_field = torch.einsum(
                "ijk,...lj->...li", self.P, d_vector_potential
            )

        return conservative_field + solenoidal_field
