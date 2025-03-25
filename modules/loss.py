import math

import torch
import torch.nn.functional as F


class ArcFaceLoss(torch.nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_features = feature_dim
        self.out_features = num_classes
        self.scale = scale

        self.weight = torch.nn.Parameter(
            torch.empty(size=(self.out_features, self.in_features))
        )
        torch.nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, x: torch.Tensor, y: torch.LongTensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine * cosine).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, y.view(-1, 1), 1)
        logit = (one_hot * phi) + ((1 - one_hot) * cosine)
        return F.cross_entropy(logit * self.scale, y)
