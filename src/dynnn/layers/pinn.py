from typing import Literal
import torch
from torch import nn
import torch.autograd.functional as AF
from itertools import permutations
import math

from dynnn.layers import DynamicallySizedNetwork, TranslationallyInvariantLayer
from dynnn.types import MIN_N_BODIES, MAX_N_BODIES
from dynnn.utils import permutation_tensor


class PINN(nn.Module):
    """
    Learn arbitrary vector fields that are sums of conservative and solenoidal fields

    TODO:
    - dimensionality of permutation_tensor
    """

    def __init__(
        self,
        input_dims: tuple[int, int, int],
        hidden_dim: int,
        field_type: Literal["conservative", "solenoidal", "both", "port"] = "both",
    ):
        super(PINN, self).__init__()
        self.input_dim = math.prod(input_dims)
        self.P = permutation_tensor()  # Levi-Civita permutation tensor
        self.M = nn.Parameter(torch.randn(self.input_dim, self.input_dim))
        self.field_type = field_type
        self.invariant_layer = TranslationallyInvariantLayer()
        self.use_invariant_layer = True

        self.model = DynamicallySizedNetwork(
            self.input_dim,
            hidden_dim,
            self.input_dim,
            dynamic_dim=2,
            dynamic_range=(MIN_N_BODIES, MAX_N_BODIES),
            dynamic_multiplier=math.prod(input_dims[1:]),
        )

    def skew(self):
        """
        Skew-symmetric matrix
        """
        return 0.5 * (self.M - self.M.T)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Neural vector field

        x size: batch_size, (time_scale*t_span_max) x n_bodies x len([q, p]) x n_dims
        """
        # if self.use_invariant_layer:
        #     invariant_features = self.invariant_layer(x)
        #     potentials = self.model(invariant_features).reshape(x.shape)

        # get the potentials from the MLP
        potentials = self.model(x)

        # split the potentials into scalar and vector components
        scalar_potential, vector_potential = torch.split(potentials, 1, dim=-2)

        if self.field_type == "none":
            return potentials

        if self.field_type == "port":
            # learn skew invariance
            d_potential = torch.autograd.grad(
                [potentials.sum()], [x], create_graph=True
            )[0]

            assert d_potential is not None
            return torch.einsum(
                "bti,ij->btj",
                d_potential.reshape(d_potential.shape[0], d_potential.shape[1], -1),
                self.skew(),
            ).reshape(d_potential.shape)

        # start out with both components set to 0
        conservative_field = torch.zeros_like(x)
        solenoidal_field = torch.zeros_like(x)

        if self.field_type in ["both", "conservative"]:
            """
            Conservative: models energy-conserving physical systems; irrotational (vanishing curl).

            If F is a conservative vector field, ∃ a scalar function φ such that F = ∇φ
            (so the MLP learns φ and we take the gradient to get F)

            Vector field F is conservative IFF there exists this scalar function,
            i.e. a conservative vector field is completely described by its scalar potential function
            (which is why we're looking at only scalar_potential here)
            """
            # batch_size, (time_scale*t_span_max) x n_bodies x (len([r, v]) * n_dims)
            d_scalar_potential = torch.autograd.grad(
                [scalar_potential.sum()],
                [x],
                create_graph=True,
            )[0]
            assert d_scalar_potential is not None
            conservative_field = d_scalar_potential

        if self.field_type in ["both", "solenoidal"]:
            """
            Solenoidal: a vector field with zero divergence (aka no sources or sinks).
            """
            d_vector_potential = torch.autograd.grad(
                [vector_potential.sum()],
                [x],
                create_graph=True,
            )[0]
            assert d_vector_potential is not None

            # Levi-Civita tensor ensures that the curl operation is performed correctly regardless of the chosen coordinates.
            solenoidal_field = torch.einsum(
                "ijk,...lj->...li", self.P, d_vector_potential
            )

        return conservative_field + solenoidal_field
