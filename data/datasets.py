from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import lightning
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from PIL import Image

from . import pandas_utils

_BASE_PATH = Path(__file__).parent.parent / "dataset"
IMPOSTOR_ID = -1


def _get_full_path(path: str) -> Path:
    return _BASE_PATH / path


class WebFaceDataset(torch.utils.data.Dataset):
    NUM_IDENTITIES = 10571

    def __init__(
        self, metadata: pd.DataFrame, transform: Callable[[Image], torch.Tensor]
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> (torch.Tensor, np.int64):
        data = self.metadata.iloc[index]
        image = Image.open(_get_full_path(data["file"]))
        return self.transform(image), data["id"]


@dataclass
class Knowledge:
    data: torch.Tensor
    ids: torch.Tensor

    def to(self, device: str | torch.device) -> Self:
        return Knowledge(self.data.to(device), self.ids.to(device))


class WebFaceDatamodule(lightning.LightningDataModule):
    TEST_BATCH_SIZE = 1024

    def __init__(
        self,
        transform: Callable[[Image], torch.Tensor],
        batch_size: int,
        num_workers: int = 8,
        num_validation_members: int = 50,
        num_validation_impostors: int = 50,
        num_members: int = 100,
        num_impostors: int = 100,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_validation_members = num_validation_members
        self.num_validation_impostors = num_validation_impostors
        self.num_members = num_members
        self.num_impostors = num_impostors

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.knowledge: Dict[str, Knowledge] = {}

    def _get_evaluation_data(
        self, df: pd.DataFrame, min_impostor_id
    ) -> (WebFaceDataset, Knowledge):
        members_metadata = pandas_utils.l_than(df, "id", min_impostor_id)
        members_test_metadata = members_metadata.drop_duplicates(subset="id")
        impostor_metadata = pandas_utils.ge_than(df, "id", min_impostor_id)
        impostor_metadata["id"] = IMPOSTOR_ID
        metadata = pd.concat((members_test_metadata, impostor_metadata))
        dataset = WebFaceDataset(metadata, self.transform)
        knowledge_metadata = members_metadata.duplicated(subset="id")
        knowledge = Knowledge(
            torch.stack(
                [
                    self.transform(Image.open(_get_full_path(row[1])))
                    for row in knowledge_metadata.iterrows()
                ]
            ),
            torch.LongTensor(knowledge_metadata["id"]),
        )
        return dataset, knowledge

    def setup(self, stage: str) -> None:
        df = pd.read_csv(_get_full_path("metadata.csv"))
        max_id = df["id"].iloc[-1]
        min_impostor_id = max_id - self.num_impostors
        min_member_id = min_impostor_id - self.num_members
        min_validation_impostor_id = min_member_id - self.num_validation_impostors
        min_validation_member_id = min_member_id - self.num_validation_members

        if stage == "fit":
            train_metadata = pandas_utils.l_than(df, "id", min_validation_member_id)
            val_metadata = pandas_utils.between(
                df, "id", min_validation_member_id, min_member_id
            )
            self.train_dataset = WebFaceDataset(train_metadata, self.transform)
            self.validation_dataset, knowledge = self._get_evaluation_data(
                val_metadata, min_validation_impostor_id
            )
            self.knowledge["validation"] = knowledge
        else:
            self.test_dataset, knowledge = self._get_evaluation_data(
                pandas_utils.ge_than(df, "id", min_member_id), min_impostor_id
            )
            self.knowledge["test"] = knowledge

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        assert self.train_dataset is not None
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        assert self.validation_dataset is not None
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        assert self.test_dataset is not None
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.TEST_BATCH_SIZE
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        assert self.test_dataset is not None
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.TEST_BATCH_SIZE
        )

    def get_knowledge(self, phase: Literal["validation", "test"]) -> Knowledge:
        return self.knowledge[phase]

    @property
    def num_train_classes(self) -> int:
        return WebFaceDataset.NUM_IDENTITIES - (
            self.num_validation_members
            + self.num_validation_impostors
            + self.num_members
            + self.num_impostors
        )
