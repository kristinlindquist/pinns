from itertools import permutations
import torch
from torch import nn
import torch.autograd.functional as AF
from typing import Literal

from dynnn.layers import (
    DynamicallySizedNetwork,
    SkewInvariantLayer,
    TranslationallyInvariantLayer,
)
from dynnn.types import MIN_N_BODIES, MAX_N_BODIES, PinnModelArgs, VectorField

from .utils import permutation_tensor


class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for learning arbitrary vector fields.

    TODO:
    - dimensionality of permutation_tensor
    """

    def __init__(
        self,
        args: PinnModelArgs,
        n_dims: int,
    ):
        super(PINN, self).__init__()

        # canonical_input_dim * len([q, p]) * n_dims
        input_dim = args.canonical_input_dim * 2 * n_dims

        # Levi-Civita permutation tensor
        self.P = permutation_tensor()

        self.invariant_layer = TranslationallyInvariantLayer()
        self.skew_invariant_layer = SkewInvariantLayer(args.canonical_hidden_dim)
        self.use_invariant_layer = args.use_invariant_layer
        self.field_type = args.vector_field_type

        self.model = DynamicallySizedNetwork(
            input_dim,
            args.canonical_hidden_dim,
            input_dim,
            dynamic_dim=2,
            dynamic_range=(MIN_N_BODIES, MAX_N_BODIES),
            dynamic_multiplier=2 * n_dims,  # len([q, p]) * n_dims
            extra_canonical_output_layers=(
                [self.skew_invariant_layer]
                if args.vector_field_type == VectorField.PORT
                else []
            ),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Neural vector field

        x size: batch_size, (time_scale*t_span_max) x n_bodies x len([q, p]) x n_dims
        """
        if self.use_invariant_layer:
            # run input through invariant layer(s) first to improve learning rate
            x = self.invariant_layer(x)

        # get the potentials from the MLP
        potentials = self.model(x)

        # split the potentials into scalar and vector components
        scalar_potential, vector_potential = torch.split(potentials, 1, dim=-2)

        if self.field_type == VectorField.NONE:
            return potentials

        if self.field_type == VectorField.PORT:
            # Port-Hamiltonian systems
            # (skew-symmetric matrix - already handled in MLP)
            return potentials

        # start out with both components set to 0
        conservative_field = torch.zeros_like(x)
        solenoidal_field = torch.zeros_like(x)

        if self.field_type in [VectorField.HELMHOLTZ, VectorField.CONSERVATIVE]:
            """
            Conservative: models energy-conserving physical systems; irrotational (vanishing curl).

            If F is a conservative vector field, ∃ a scalar function φ such that F = ∇φ
            (so the MLP learns φ and we take the gradient to get F)

            Vector field F is conservative IFF there exists this scalar function,
            i.e. a conservative vector field is completely described by its scalar potential function
            (which is why we're looking at only scalar_potential here)
            """
            # batch_size, (time_scale*t_span_max) x n_bodies x (len([q, p]) * n_dims)
            d_scalar_potential = torch.autograd.grad(
                [scalar_potential.sum()],
                [x],
                create_graph=True,
            )[0]
            assert d_scalar_potential is not None
            conservative_field = d_scalar_potential

        if self.field_type in [VectorField.HELMHOLTZ, VectorField.SOLENOIDAL]:
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
