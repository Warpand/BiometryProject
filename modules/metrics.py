from abc import ABC, abstractmethod

import torch
from torchmetrics import Metric

from data import IMPOSTOR_ID


class BaseAccuracy(Metric, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total

    @abstractmethod
    def update(self, predictions: torch.Tensor, target: torch.Tensor) -> None:
        pass


class ImpostorAccuracy(BaseAccuracy):
    def update(self, predictions: torch.Tensor, target: torch.Tensor) -> None:
        impostor_mask = target == IMPOSTOR_ID
        predictions = predictions[impostor_mask]
        self.correct += torch.sum(predictions == IMPOSTOR_ID)
        self.total += len(predictions)


class MemberAccuracy(BaseAccuracy):
    def update(self, predictions: torch.Tensor, target: torch.Tensor) -> None:
        member_mask = target != IMPOSTOR_ID
        predictions = predictions[member_mask]
        target = target[member_mask]
        self.correct += torch.sum(predictions == target)
        self.total += len(predictions)


class AnyoneAccuracy(BaseAccuracy):
    def update(self, predictions: torch.Tensor, target: torch.Tensor) -> None:
        member_mask = target != IMPOSTOR_ID
        predictions = predictions[member_mask]
        self.correct += torch.sum(predictions != IMPOSTOR_ID)
        self.total += len(predictions)
