import torch
from typing import Any, Callable

from .args import TrainingArgs
from .stats import ModelStats
from .types import Dataset

GeneratorFunction = Callable[[torch.Tensor], torch.Tensor]
TrainLoop = Callable[[TrainingArgs, Dataset], ModelStats]

# specific to MVE
TransformY = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
