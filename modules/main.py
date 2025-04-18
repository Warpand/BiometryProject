from typing import Any, List, Optional, Tuple, Union

import lightning
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import Metric

from data import IMPOSTOR_ID

from .loss import ArcFaceLoss
from .metrics import ImpostorAccuracy, MemberAccuracy
from .resnet import ResnetAdapter


class ArcFaceModule(lightning.LightningModule):
    def __init__(
        self,
        resnet: ResnetAdapter,
        num_classes: int,
        arc_face_margin: float = 0.5,
        arc_face_scale: float = 64.0,
        threshold: float | List[float] = 0.4,
    ):
        super().__init__()
        self.resnet = resnet
        self.arc_face_loss = ArcFaceLoss(
            resnet.embedding_dim,
            num_classes,
            scale=arc_face_scale,
            margin=arc_face_margin,
        )
        if isinstance(threshold, float):
            self.thresholds = [threshold]
        else:
            self.thresholds = threshold
        self.impostor_accuracy: List[Metric] = []
        self.member_accuracy: List[Metric] = []
        self.knowledge: Optional[Tuple[torch.Tensor, torch.LongTensor]] = None
        self.save_hyperparameters(ignore=["resnet"])

    def setup(self, stage: str) -> None:
        self.impostor_accuracy = [
            ImpostorAccuracy().to(self.device) for _ in range(len(self.thresholds))
        ]
        self.member_accuracy = [
            MemberAccuracy().to(self.device) for _ in range(len(self.thresholds))
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

    def embedd(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.resnet(x))

    def set_knowledge(self, knowledge: Tuple[torch.Tensor, torch.LongTensor]):
        self.knowledge = (self.embedd(knowledge[0]), knowledge[1])

    def find_identities(
        self, x: torch.Tensor, thresholds: List[float]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        database, identities = self.knowledge
        embeddings = self.embedd(x)
        similarities = F.linear(embeddings, database)
        max_index = torch.argmax(similarities, dim=1)
        max_val = similarities[torch.arange(0, len(database)), max_index]
        return [
            torch.where(max_val > t, identities[max_index], IMPOSTOR_ID)
            for t in thresholds
        ]

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.LongTensor], *args
    ) -> STEP_OUTPUT:
        x, y = batch
        embeddings = self(x)
        loss = self.arc_face_loss(embeddings, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def evaluation_step(self, batch: Tuple[torch.Tensor, torch.LongTensor]) -> None:
        x, y = batch
        predicted_identities = self.find_identities(x, self.thresholds)
        for impostor_accuracy, member_accuracy, y_hat in zip(
            self.impostor_accuracy, self.member_accuracy, predicted_identities
        ):
            impostor_accuracy(y_hat, y)
            member_accuracy(y_hat, y)

    def evaluation_epoch_end(self, stage: str) -> None:
        for impostor_accuracy, member_accuracy, t in zip(
            self.impostor_accuracy, self.member_accuracy, self.thresholds
        ):
            self.log(f"{stage}/{impostor_accuracy}_{t}", impostor_accuracy.compute())
            self.log(f"{stage}/{member_accuracy}_{t}", member_accuracy.compute())
            impostor_accuracy.restart()
            member_accuracy.restart()

    def on_validation_epoch_start(self) -> None:
        self.set_knowledge(self.trainer.datamodule.get_knowledge("validation"))  # type: ignore

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.LongTensor]
    ) -> STEP_OUTPUT:
        self.evaluation_step(batch)
        return None

    def on_validation_epoch_end(self) -> None:
        self.evaluation_epoch_end("validation")

    def on_test_epoch_start(self) -> None:
        self.set_knowledge(self.trainer.datamodule.get_knowledge("test"))  # type: ignore

    def test_step(self, batch: Tuple[torch.Tensor, torch.LongTensor]) -> STEP_OUTPUT:
        self.evaluation_step(batch)
        return None

    def on_test_epoch_end(self) -> None:
        self.evaluation_epoch_end("test")
