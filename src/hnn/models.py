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
        field_type: Literal["conservative", "solenoidal", "both"] = "solenoidal",
    ):
        super(HNN, self).__init__()
        self.differentiable_model = differentiable_model
        self.M = self.permutation_tensor()  # Levi-Civita permutation tensor
        self.field_type = field_type

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        # batch_size, (timescale*t_span[1]) x n_bodies x (len([r, v]) * num_dim)
        _x = x.reshape(*x.shape[0:3], -1)
        y = self.differentiable_model(_x).reshape(*x.shape)
        F1, F2 = torch.split(y, 1, dim=-2)  # split r & v
        return F1, F2

    def time_derivative(self, x, t=None) -> torch.Tensor:
        """
        Neural Hamiltonian-style vector field
        """
        if len(x.shape) != 5:
            raise ValueError(
                "Input tensor must be of shape (batch_size, timepoints, n_bodies, coord_dim, dim)"
            )

        # batch_size, (timescale*t_span[1]) x n_bodies x len([q, p]) x num_dim
        batch_size, timepoints, n_bodies, coord_dim, dim = x.shape
        F1, F2 = self.forward(x)

        # start out with both components set to 0
        conservative_field = torch.zeros_like(x)
        solenoidal_field = torch.zeros_like(x)

        if self.field_type in ["both", "conservative"]:
            """
            conservative: "vector fields representing forces of physical systems in which energy is conserved"
                (line integral is path independent) https://en.wikipedia.org/wiki/Conservative_vector_field
            """
            dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True)[0]
            eye_tensor = torch.eye(coord_dim, dim).repeat(
                batch_size, timepoints, n_bodies, 1, 1
            )
            conservative_field = torch.einsum("ijklm,ijkln->ijkln", eye_tensor, dF1)

        if self.field_type in ["both", "solenoidal"]:
            """
            solenoidal: "a vector field v with divergence zero at all points in the field"
                (aka with no sources or sinks) https://en.wikipedia.org/wiki/Solenoidal_vector_field
            """
            # gradients for solenoidal field
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0]

            # curl of dF2 -> solenoidal field
            solenoidal_field = torch.einsum("ijk,...lj->...li", self.M, dF2)

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
