from collections.abc import Callable
from typing import Union

import torch
import torchvision.models
from PIL import Image

RESNET_VERSIONS = [18, 34, 50, 101, 152]

RESNET_FACTORY_TYPE = Union[
    torchvision.models.resnet18,
    torchvision.models.resnet34,
    torchvision.models.resnet50,
    torchvision.models.resnet101,
    torchvision.models.resnet152,
]

RESNET_WEIGHTS_TYPE = Union[
    torchvision.models.ResNet18_Weights,
    torchvision.models.ResNet34_Weights,
    torchvision.models.ResNet50_Weights,
    torchvision.models.ResNet101_Weights,
    torchvision.models.ResNet152_Weights,
]


class ResnetAdapter(torch.nn.Module):
    def __init__(self, resnet: torchvision.models.ResNet, embedding_dim: int) -> None:
        super().__init__()
        feature_dim = resnet.fc.in_features
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.head = torch.nn.Linear(feature_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.flatten(self.backbone(x)))

    @property
    def embedding_dim(self) -> int:
        return len(self.head.weight)


class ResNetBuilder:
    def __init__(
        self,
        embedding_dim: int,
        factory: RESNET_FACTORY_TYPE,
        weights: RESNET_WEIGHTS_TYPE,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.factory = factory
        self.weights = weights

    def build(self, pretrained: bool = True) -> ResnetAdapter:
        weights = self.weights if pretrained else None
        resnet = self.factory(weights=weights)
        return ResnetAdapter(resnet, self.embedding_dim)

    @property
    def transforms(self) -> Callable[[Image], torch.Tensor]:
        return self.weights.transforms


def get_resnet_builder(embedding_dim: int, version: int = 18):
    assert version in RESNET_VERSIONS, "Unknown ResNet version!"
    match version:
        case 18:
            return ResNetBuilder(
                embedding_dim,
                torchvision.models.resnet18,
                torchvision.models.ResNet18_Weights.DEFAULT,
            )
        case 34:
            return ResNetBuilder(
                embedding_dim,
                torchvision.models.resnet34,
                torchvision.models.ResNet34_Weights.DEFAULT,
            )
        case 50:
            return ResNetBuilder(
                embedding_dim,
                torchvision.models.resnet50,
                torchvision.models.ResNet50_Weights.DEFAULT,
            )
        case 101:
            return ResNetBuilder(
                embedding_dim,
                torchvision.models.resnet101,
                torchvision.models.ResNet101_Weights.DEFAULT,
            )
        case 152:
            return ResNetBuilder(
                embedding_dim,
                torchvision.models.resnet152,
                torchvision.models.ResNet152_Weights.DEFAULT,
            )
        case _:
            raise ValueError()
