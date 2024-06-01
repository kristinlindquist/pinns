"""
This module contains layers that enforce invariance properties on the input data.
Inspired by https://www.sciencedirect.com/science/article/pii/S0021999123003297
"""

import torch


class TranslationallyInvariantLayer(torch.nn.Module):
    def __init__(self):
        super(TranslationallyInvariantLayer, self).__init__()

    def forward(self, x):
        """
        Calculate translationally invariant features from a set of particle positions.
        """
        # batch_size, timepoints, 1, num_vectors, n_dims
        x_mean = x.mean(dim=2, keepdim=True)

        # Subtract the mean from each vector to ensure translation invariance
        x = x - x_mean
        return x


class RotationallyInvariantLayer(torch.nn.Module):
    def __init__(self):
        super(RotationallyInvariantLayer, self).__init__()

    @staticmethod
    def get_output_dim(n_bodies: int) -> int:
        return int(n_bodies * 2 + ((n_bodies * (n_bodies - 1)) / 2)) * 2

    def forward(self, x):
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
        dot_products = torch.sum(x_i * x_j, dim=-1)

        # retain only the upper diagonal (to avoid redundant computations)
        mask = torch.ones(n_bodies, n_bodies, dtype=torch.bool).triu()
        dot_products = dot_products[:, mask]

        norms = torch.norm(x, dim=-1)
        invariant_features = torch.cat([dot_products, norms], dim=-2)

        return invariant_features.reshape(batch_size, timepoints, -1)