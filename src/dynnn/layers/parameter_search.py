import torch
from torch import nn
from torch.distributions import Categorical, Distribution, RelaxedOneHotCategorical

from dynnn.types import SaveableModel

OutputRanges = dict[str, tuple[int, int] | tuple[float, float]]


class SampledRangeOutputLayer(nn.Module):
    """
    An output layer that
    1) scales outputs according to their specified ranges
    2) samples from a distribution to provide a policy gradient

    Args:
        input_size: size of input tensor
        output_ranges: dictionary of output ranges

    Model outputs:
        tuple[torch.Tensor, Distribution]: scaled outputs and distribution
    """

    def __init__(
        self,
        input_size: int,
        output_ranges: OutputRanges,
    ):
        super().__init__()
        self.output_ranges = output_ranges
        self.linear = nn.Linear(input_size, len(output_ranges))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Distribution]:
        """
        Forward pass that:

        1) Computes logits from the linear layer
        2) Gets the sigmoid outputs
        3) Samples from the distribution (for policy gradients)
        4) Scales the outputs according to the specified ranges

        Returns:
            tuple[torch.Tensor, torch.Tensor, Distribution]:
                scaled outputs (used for `step`)
                sampled_outputs (used for loss)
                distribution (used for loss)
        """

        logits = self.linear(x)
        sigmoid_outputs = torch.sigmoid(logits)
        distribution = RelaxedOneHotCategorical(
            torch.tensor(1.0), logits=sigmoid_outputs
        )
        sampled_outputs = distribution.rsample()

        def scale_output(i: int, o_range: tuple) -> float:
            start, end = o_range
            output_span = end - start
            return sampled_outputs[i] * output_span + start

        scaled_outputs = [
            scale_output(i, o_range)
            for i, o_range in enumerate(self.output_ranges.values())
            if isinstance(o_range, tuple)
        ]

        return torch.stack(scaled_outputs), sampled_outputs, distribution


class ParameterSearchModel(SaveableModel):
    """
    Simple feedforward model for RL parameter search.

    - Uses a sampled output layer to provide policy gradients.
    - Scales outputs according to specified ranges.

    Args:
        state_dim: size of input tensor
        output_ranges: dictionary of output ranges
        hidden_dim: size of hidden layer
    """

    def __init__(
        self,
        run_id: float | str,
        state_dim: int,
        output_ranges: OutputRanges,
        hidden_dim: int = 128,
        rnn_hidden_dim: int = 64,
        model_name="parameter_search",
    ):
        super(ParameterSearchModel, self).__init__(model_name, run_id)
        action_dim = len(output_ranges)
        input_dim = state_dim + rnn_hidden_dim

        self.rnn = nn.LSTM(state_dim, rnn_hidden_dim, batch_first=True)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            SampledRangeOutputLayer(action_dim, output_ranges),
        )

    def forward(
        self, state: torch.Tensor, state_history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Distribution]:
        """
        Forward pass through the model.

        Args:
            state: input tensor (current state)
            state_history: input tensor (all historical states)

        Returns:
            tuple[torch.Tensor, torch.Tensor, Distribution]:
                scaled outputs (used for `step`)
                sampled_outputs (used for loss)
                distribution (used for loss)
        """
        _, (hidden, _) = self.rnn(state_history)
        distilled_state_history = hidden.squeeze(0)

        return self.layers(torch.concatenate([state, distilled_state_history]))
