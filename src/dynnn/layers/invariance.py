"""
This module contains layers that enforce invariance properties on the input data.
Inspired by https://www.sciencedirect.com/science/article/pii/S0021999123003297
"""

import torch
from torch import nn


class TranslationallyInvariantLayer(nn.Module):
    """
    Compute translationally invariant features from a set of particle positions.
    """

    def __init__(self):
        super(TranslationallyInvariantLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        in/out shape: (batch_size, timepoints, n_bodies, num_vectors, n_dims)
        """

        # get mean for each batch, so it is a constant.
        x_mean = x.mean(dim=(1, 2, 3, 4), keepdim=True)

        # Subtract the mean from each vector to ensure translation invariance
        x = x - x_mean
        return x


class SkewInvariantLayer(nn.Module):
    """
    Compute skew invariant features from a set of particle positions.
    """

    def __init__(self, input_dim: int):
        super(SkewInvariantLayer, self).__init__()

        # Skew-symmetric matrix
        self.S = nn.Parameter(torch.randn(input_dim, input_dim))

    @property
    def skew(self):
        """
        Skew-symmetric matrix
        """
        return 0.5 * (self.S - self.S.T)

    def forward(
        self,
        input: torch.Tensor,
        potentials: torch.Tensor,
    ) -> torch.Tensor:
        d_potential = torch.autograd.grad(
            [potentials.sum()], [input], create_graph=True
        )[0]

        assert d_potential is not None
        return torch.einsum(
            "bti,ij->btj",
            d_potential.reshape(*d_potential.shape[0:2], -1),
            self.skew,
        ).reshape(d_potential.shape)


class RotationallyInvariantLayer(nn.Module):
    """
    Compute rotationally invariant features from a set of particle positions.

    NOTE: this might be buggy, perhaps mishandling dimensions.
    Its use results in very bad learning performance.

    self.invariant_layer = nn.Sequential(
        RotationallyInvariantLayer(),
        TranslationallyInvariantLayer()
    )
    in_dim = (
        RotationallyInvariantLayer.get_output_dim(input_dims[0])
        if self.use_invariant_layer and False
        else self.input_dim
    )
    """

    def __init__(self):
        super(RotationallyInvariantLayer, self).__init__()

    @staticmethod
    def get_output_dim(n_bodies: int) -> int:
        return int(n_bodies * 2 + ((n_bodies * (n_bodies - 1)) / 2)) * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate rotationally invariant features from a set of particle positions.
        """
        batch_size, timepoints, n_bodies, num_vectors, n_dims = x.shape
        x = x.view(batch_size * timepoints, n_bodies, num_vectors, n_dims)

        # Compute pairwise dot products for each body
        # batch_size * timepoints, n_bodies, 1, num_vectors, n_dims
        x_i = x.unsqueeze(2)
        # batch_size * timepoints, 1, n_bodies, num_vectors, n_dims
        x_j = x.unsqueeze(1)

        # batch_size * timepoints, n_bodies, n_bodies, num_vectors
        # captures the relative orientations between the vectors
        dot_products = torch.sum(x_i * x_j, dim=-1)

        # retain only the upper diagonal (to avoid redundant computations)
        mask = torch.ones(n_bodies, n_bodies, dtype=torch.bool).triu()

        dot_products = dot_products[:, mask]

        # the magnitudes of the individual vectors
        norms = torch.norm(x, dim=-1)

        invariant_features = torch.cat([dot_products, norms], dim=-2)

        return invariant_features.reshape(batch_size, timepoints, -1)
