from functools import partial
from pydantic import BaseModel, ConfigDict
from pydantic.functional_validators import BeforeValidator
import torch
from typing import Annotated, Any, Literal

from dynnn.utils import coerce_int, load_model, save_model


ForcedInt = Annotated[int, BeforeValidator(coerce_int)]
ForcedIntOrNone = Annotated[int, BeforeValidator(partial(coerce_int, allow_none=True))]


class Trajectory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    q: torch.Tensor
    p: torch.Tensor
    dq: torch.Tensor
    dp: torch.Tensor
    t: torch.Tensor
    masses: torch.Tensor


class Dataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    x: torch.Tensor
    dx: torch.Tensor
    test_x: torch.Tensor
    test_dx: torch.Tensor
    masses: torch.Tensor
    time: torch.Tensor


class SaveableModel(torch.nn.Module):
    def __init__(self, model_name: str, run_id: float | str):
        super(SaveableModel, self).__init__()
        self.model_name = model_name
        self.run_id = run_id

    def save(self):
        return save_model(self, run_id=self.run_id, model_name=self.model_name)

    def load(self):
        return self(load_model(self.run_id, self.model_name))


class DatasetGenerationFailure(Exception):
    pass
