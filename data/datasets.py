from functools import cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image

_BASE_PATH = Path(__file__).parent.parent / "dataset"


def _get_full_path(path: str) -> Path:
    return _BASE_PATH / path


class TrainDataset(torch.utils.data.Dataset):
    def __init__(
        self, metadata: pd.DataFrame, transform: Callable[[Image], torch.Tensor]
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.transform = transform
        self._num_classes = metadata["id"].max()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> (torch.Tensor, np.int64):
        data = self.metadata.iloc[index]
        image = Image.open(_get_full_path(data["file"]))
        return self.transform(image), data["id"]

    @property
    def num_classes(self) -> int:
        return self._num_classes


class TestDataset(torch.utils.data.Dataset):
    def __init__(
        self, metadata: pd.DataFrame, transform: Callable[[Image], torch.Tensor]
    ) -> None:
        super().__init__()
        self.data = torch.stack(
            [
                transform(Image.open(_get_full_path(row[1])))
                for row in metadata.itertuples()
            ]
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]

    @property
    def all(self) -> torch.Tensor:
        return self.data


class DatasetFactory:
    def __init__(
        self,
        num_members: int,
        num_impostors: int,
        transform: Callable[[Image], torch.Tensor],
    ):
        self.df = pd.read_csv(_get_full_path("metadata.csv"))
        max_id = self.df["id"].iloc[-1]
        self.min_impostor_id = max_id - num_impostors
        self.min_member_id = self.min_impostor_id - num_members
        self.transform = transform

    @cache
    def _get_member_metadata(self) -> pd.DataFrame:
        return self.df[
            (self.df["id"] >= self.min_member_id)
            & (self.df["id"] < self.min_impostor_id)
        ]

    def get_train_dataset(self) -> TrainDataset:
        train_metadata = self.df[self.df["id"] < self.min_member_id]
        return TrainDataset(train_metadata, self.transform)

    def get_knowledge_dataset(self) -> TestDataset:
        member_metadata = self._get_member_metadata()
        knowledge_metadata = member_metadata.drop_duplicates(subset="id")
        return TestDataset(knowledge_metadata, self.transform)

    def get_member_dataset(self) -> TestDataset:
        member_metadata = self._get_member_metadata()
        member_metadata = member_metadata[member_metadata.duplicated(subset="id")]
        return TestDataset(member_metadata, self.transform)

    def get_impostor_dataset(self) -> TestDataset:
        return TestDataset(
            self.df[self.df["id"] >= self.min_impostor_id], self.transform
        )
