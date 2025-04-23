from unittest import TestCase

import torch

from modules.loss import ArcFaceLoss


class TestArcFaceLoss(TestCase):
    def test_forward(self):
        feature_dim = 512
        num_classes = 1000
        batch_size = 32
        loss = ArcFaceLoss(feature_dim, num_classes)
        features = torch.randn(batch_size, feature_dim)
        labels = torch.randint(0, num_classes, (32,))

        loss_value = loss(features, labels)
        self.assertEqual(0, loss_value.ndim)
