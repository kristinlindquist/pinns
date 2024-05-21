from typing import Literal
import torch
import torch.autograd.functional as AF


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
        assume_canonical_coords: bool = False,
    ):
        super(HNN, self).__init__()
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim)  # Levi-Civita permutation tensor
        self.field_type = field_type

        if self.assume_canonical_coords:
            print(
                "Warning: assume_canonical_coords is True, which probably won't work."
            )

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        # batch_size, (timescale*t_span[1]) x n_bodies x (len([r, v]) * num_dim)
        _x = x.reshape(*x.shape[0:3], -1)
        y = self.differentiable_model(_x).reshape(*x.shape)
        F1, F2 = torch.split(y, 1, dim=-2)  # split r & v
        return F1, F2

    def time_derivative(
        self,
        x,
        t=None,
        separate_fields: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
            # gradients for conservative field
            dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True)[0]
            eye_tensor = (
                torch.eye(coord_dim, dim)
                .to(dF1.device)
                .repeat(batch_size, timepoints, n_bodies, 1, 1)
            )

            conservative_field = torch.einsum(
                "ijklm,ijkln->ijkln", eye_tensor, dF1
            ).squeeze(-1)

        if self.field_type in ["both", "solenoidal"]:
            # gradients for solenoidal field
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0]
            # TODO: shape is probably wrong
            M_tensor = (
                self.M.t().to(dF2.device).repeat(batch_size, timepoints, n_bodies, 1, 1)
            )
            solenoidal_field = torch.einsum("ijkl,ijkm->ijkm", M_tensor, dF2).squeeze(
                -1
            )

        if separate_fields:
            return [conservative_field, solenoidal_field]

        return conservative_field + solenoidal_field

    def permutation_tensor(self, n: int) -> torch.Tensor:
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n)  # diagonal matrix
            M = torch.cat([M[n // 2 :], -M[: n // 2]])
        else:
            """
            Constructs the Levi-Civita permutation tensor
            """
            M = torch.ones(n, n)  # matrix of ones
            M *= 1 - torch.eye(n)  # clear diagonals
            M[::2] *= -1  # pattern of signs
            M[:, ::2] *= -1

            for i in range(n):  # make asymmetric
                for j in range(i + 1, n):
                    M[i, j] *= -1
        return M
