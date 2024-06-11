import torch
from torch.distributions import Categorical, Distribution, RelaxedOneHotCategorical

OutputRanges = dict[str, tuple[int, int] | tuple[float, float]]


class SampledRangeOutputLayer(torch.nn.Module):
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
        self.linear = torch.nn.Linear(input_size, len(output_ranges))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Distribution]:
        logits = self.linear(x)
        sigmoid_outputs = torch.sigmoid(logits)
        distribution = RelaxedOneHotCategorical(1.0, logits=sigmoid_outputs)
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

        return torch.stack(scaled_outputs), distribution


class SimulatorModel(torch.nn.Module):
    """
    RL model for exploring simulation parameter space
    """

    def __init__(
        self,
        state_dim: int,
        output_ranges: OutputRanges,
        hidden_dim: int = 64,
    ):
        super(SimulatorModel, self).__init__()
        action_dim = len(output_ranges)

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
            SampledRangeOutputLayer(action_dim, output_ranges),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, Distribution]:
        return self.layers(state)
