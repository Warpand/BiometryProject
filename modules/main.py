import itertools
from typing import List, Optional, Tuple, Union

import lightning
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchmetrics import Metric

from cfg import OptimizerConfig
from data import IMPOSTOR_ID, Knowledge

from .loss import ArcFaceLoss
from .metrics import ImpostorAccuracy, MemberAccuracy
from .resnet import ResnetAdapter


class ArcFaceModule(lightning.LightningModule):
    def __init__(
        self,
        resnet: ResnetAdapter,
        num_classes: int,
        arc_face_margin: float,
        arc_face_scale: float,
        threshold: float | List[float],
        epochs: int,
        optimizer_config: OptimizerConfig,
        compile_submodules: bool,
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
        self.compile_submodules = compile_submodules

        self.knowledge: Optional[Knowledge] = None
        self.impostor_accuracy: List[Metric] = []
        self.member_accuracy: List[Metric] = []
        self.save_hyperparameters(ignore=["resnet"], logger=False)

    def setup(self, stage: str) -> None:
        self.impostor_accuracy = [
            ImpostorAccuracy().to(self.device) for _ in range(len(self.thresholds))
        ]
        self.member_accuracy = [
            MemberAccuracy().to(self.device) for _ in range(len(self.thresholds))
        ]
        if self.compile_submodules:
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

    def on_train_epoch_end(self) -> None:
        lrs = self.lr_schedulers()[0].get_last_lr()
        self.log("train/lr_backbone", lrs[0])
        self.log("train/lr_head", lrs[1])

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
