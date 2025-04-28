import itertools
from typing import Any, List, Optional, Tuple, Union

import pytorch_lightning
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import (
    STEP_OUTPUT,
    LRSchedulerTypeUnion,
    OptimizerLRScheduler,
)
from torchmetrics import Metric

from cfg import OptimizerConfig
from data import IMPOSTOR_ID, Knowledge

from .loss import ArcFaceLoss
from .metrics import AdmissionAccuracy, ImpostorAccuracy, MemberAccuracy
from .resnet import ResnetAdapter


class ArcFaceModule(pytorch_lightning.LightningModule):
    def __init__(
        self,
        resnet: ResnetAdapter,
        num_classes: int,
        arc_face_margin: float,
        arc_face_scale: float,
        threshold: float | List[float],
        epochs: int,
        optimizer_config: OptimizerConfig,
        batch_size: Optional[int] = None,
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

        self.epochs = epochs
        self.optimizer_config = optimizer_config
        self.batch_size = batch_size

        self.knowledge: Optional[Knowledge] = None
        self.impostor_accuracy: List[Metric] = []
        self.member_accuracy: List[Metric] = []
        self.admission_accuracy: List[Metric] = []
        self.save_hyperparameters(ignore=["resnet"], logger=False)

    def setup(self, stage: str) -> None:
        self.impostor_accuracy = [
            ImpostorAccuracy().to(self.device) for _ in range(len(self.thresholds))
        ]
        self.member_accuracy = [
            MemberAccuracy().to(self.device) for _ in range(len(self.thresholds))
        ]
        self.admission_accuracy = [
            AdmissionAccuracy().to(self.device) for _ in range(len(self.thresholds))
        ]
        if stage == "fit":
            assert self.batch_size is not None
            self.resnet = torch.compile(self.resnet)
            self.arc_face_loss = torch.compile(self.arc_face_loss)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

    def embedd(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.resnet(x))

    def set_knowledge(self, knowledge: Knowledge):
        self.knowledge = knowledge.to(self.device)
        self.knowledge.data = self.embedd(self.knowledge.data)

    def find_identities(
        self, x: torch.Tensor, thresholds: List[float]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        database, identities = self.knowledge.data, self.knowledge.ids
        if self.batch_size is not None and len(x) != self.batch_size:
            padding = torch.empty(
                self.batch_size - len(x), *x.shape[1:], device=x.device
            )
            embeddings = self.embedd(torch.vstack([x, padding]))
            embeddings = embeddings[: len(x)]
        else:
            embeddings = self.embedd(x)
        similarities = F.linear(embeddings, database)
        max_index = torch.argmax(similarities, dim=1)
        max_val = similarities[torch.arange(len(similarities)), max_index]
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

    def on_train_epoch_end(self) -> None:
        lrs = self.lr_schedulers().get_last_lr()
        self.log("train/lr_backbone", lrs[0])
        self.log("train/lr_head", lrs[1])

    def evaluation_step(self, batch: Tuple[torch.Tensor, torch.LongTensor]) -> None:
        x, y = batch
        predicted_identities = self.find_identities(x, self.thresholds)
        for impostor_accuracy, member_accuracy, admission_accuracy, y_hat in zip(
            self.impostor_accuracy,
            self.member_accuracy,
            self.admission_accuracy,
            predicted_identities,
        ):
            impostor_accuracy(y_hat, y)
            member_accuracy(y_hat, y)
            admission_accuracy(y_hat, y)

    def evaluation_epoch_end(self, stage: str) -> None:
        for impostor_accuracy, member_accuracy, admission_accuracy, t in zip(
            self.impostor_accuracy,
            self.member_accuracy,
            self.admission_accuracy,
            self.thresholds,
        ):
            self.log(f"{stage}/impostor_accuracy_{t}", impostor_accuracy.compute())
            self.log(f"{stage}/member_accuracy_{t}", member_accuracy.compute())
            self.log(f"{stage}/admission_accuracy_{t}", admission_accuracy.compute())
            impostor_accuracy.reset()
            member_accuracy.reset()
            admission_accuracy.reset()

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

    def configure_optimizers(self) -> OptimizerLRScheduler:
        cfg = self.optimizer_config
        optimizer = torch.optim.SGD(
            [
                {"params": self.resnet.backbone.parameters(), "lr": cfg.lr_backbone},
                {
                    "params": itertools.chain(
                        self.resnet.head.parameters(), self.arc_face_loss.parameters()
                    ),
                    "lr": cfg.lr_head,
                },
            ],
            weight_decay=cfg.weight_decay,
        )

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: (epoch + 1) / 5
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs - cfg.warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[cfg.warmup_epochs],
        )
        return [optimizer], [scheduler]

    def lr_scheduler_step(
        self, scheduler: LRSchedulerTypeUnion, metric: Optional[Any]
    ) -> None:
        scheduler.step()
