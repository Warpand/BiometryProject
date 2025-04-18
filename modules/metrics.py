import torch
from torchmetrics import Metric

from data import IMPOSTOR_ID


class ImpostorAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: torch.Tensor, target: torch.Tensor) -> None:
        impostor_mask = target == IMPOSTOR_ID
        predictions = predictions[impostor_mask]
        self.correct += torch.sum(predictions == IMPOSTOR_ID)
        self.total += len(predictions)

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total


class MemberAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: torch.Tensor, target: torch.Tensor) -> None:
        member_mask = target != IMPOSTOR_ID
        predictions = predictions[member_mask]
        target = target[member_mask]
        self.correct += torch.sum(predictions == target)
        self.total += len(target)

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total
