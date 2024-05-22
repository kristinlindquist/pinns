from typing import Literal
import torch
import torch.autograd.functional as AF
from itertools import permutations


class MLP(torch.nn.Module):
    """
    Just a salt-of-the-earth MLP
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)
        self.nonlinearity = torch.nn.Tanh()

        self.module = torch.nn.Sequential(
            self.linear1,
            self.nonlinearity,
            self.linear2,
            self.nonlinearity,
            self.linear3,
        )

    def forward(self, x) -> torch.Tensor:
        return self.module(x)


class HNN(torch.nn.Module):
    """
    Learn arbitrary vector fields that are sums of conservative and solenoidal fields
    """

    def __init__(
        self,
        input_dim: int,
        differentiable_model,
        field_type: Literal["conservative", "solenoidal", "both"] = "both",
    ):
        super(HNN, self).__init__()
        self.differentiable_model = differentiable_model
        self.M = self.permutation_tensor()  # Levi-Civita permutation tensor
        self.input_dim = input_dim
        self.field_type = field_type

        # a smooth, rapidly decaying 3d vector field can be decomposed into a conservative and solenoidal field
        # https://en.wikipedia.org/wiki/Helmholtz_decomposition
        if field_type != "both":
            print(
                f"Warning: a field_type of {field_type} might not capture the full dynamics of the system."
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # batch_size, (timescale*t_span[1]) x n_bodies x (len([r, v]) * num_dim)
        _x = x.reshape(*x.shape[0:3], -1)
        y = self.differentiable_model(_x).reshape(*x.shape)
        scalar_potential, vector_potential = torch.split(y, 1, dim=-2)
        return scalar_potential, vector_potential

    def time_derivative(self, x: torch.Tensor, t=None) -> torch.Tensor:
        """
        Neural Hamiltonian-style vector field
        """
        # batch_size, (timescale*t_span[1]) x n_bodies x len([q, p]) x num_dim
        batch_size, timepoints, n_bodies, coord_dim, dim = x.shape
        scalar_potential, vector_potential = self.forward(x)

        # start out with both components set to 0
        conservative_field = torch.zeros_like(x)
        solenoidal_field = torch.zeros_like(x)

        if self.field_type in ["both", "conservative"]:
            """
            conservative: models energy-conserving physical systems; irrotational (vanishing curl)
            """
            d_scalar_potential = torch.autograd.grad(
                scalar_potential.sum(), x, create_graph=True
            )[0]
            conservative_field = d_scalar_potential

        if self.field_type in ["both", "solenoidal"]:
            """
            solenoidal: a vector field with zero divergence (aka no sources or sinks)
            """
            d_vector_potential = torch.autograd.grad(
                vector_potential.sum(), x, create_graph=True
            )[0]
            solenoidal_field = torch.einsum(
                "ijk,...lj->...li", self.M, d_vector_potential
            )

        return conservative_field + solenoidal_field

    def permutation_tensor(self) -> torch.Tensor:
        """
        Constructs the Levi-Civita permutation tensor for 3 dimensions.
        """
        M = torch.zeros((3, 3, 3))
        M[0, 1, 2] = 1
        M[1, 2, 0] = 1
        M[2, 0, 1] = 1
        M[2, 1, 0] = -1
        M[1, 0, 2] = -1
        M[0, 2, 1] = -1
        return M
