from functools import partial
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from pydantic.functional_validators import BeforeValidator
import torch
import torch.nn.functional as F
from typing import Annotated, Any, Callable, Literal

from dynnn.utils import coerce_int


ForcedInt = Annotated[int, BeforeValidator(coerce_int)]
ForcedIntOrNone = Annotated[int, BeforeValidator(partial(coerce_int, allow_none=True))]

GeneratorFunction = Callable[[torch.Tensor], torch.Tensor]


class Trajectory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    q: torch.Tensor
    p: torch.Tensor
    dq: torch.Tensor
    dp: torch.Tensor
    t: torch.Tensor
    masses: torch.Tensor
