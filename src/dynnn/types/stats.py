from pydantic import BaseModel, ConfigDict
import torch


class ModelStats(BaseModel):
    """
    Object to hold statistics for a PINN model
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    train_loss: list[torch.Tensor] = []
    test_loss: list[torch.Tensor] = []
    train_additional_loss: list[torch.Tensor] = []
    test_additional_loss: list[torch.Tensor] = []

    def get_as_float(self, key: str):
        values = getattr(self, key)
        return [v.item() for v in values]

    @staticmethod
    def _calc_mean(values: list[torch.Tensor]) -> torch.Tensor:
        if len(values) == 0:
            return torch.tensor([0.0])
        return torch.stack(values).mean()

    @staticmethod
    def _calc_min(values: list[torch.Tensor]) -> torch.Tensor:
        if len(values) == 0:
            return torch.tensor([0.0])
        return torch.stack(values).min()

    @property
    def min_train_loss(self) -> torch.Tensor:
        return self._calc_min(self.train_loss)

    @property
    def min_test_loss(self) -> torch.Tensor:
        return self._calc_min(self.test_loss)

    def encode(self) -> tuple[float, float]:
        return (
            self.min_train_loss,
            self.min_test_loss,
        )
