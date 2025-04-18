from unittest import TestCase

import torch

from data import IMPOSTOR_ID
from modules.metrics import ImpostorAccuracy, MemberAccuracy


class TestMetrics(TestCase):
    x1 = torch.LongTensor([15, IMPOSTOR_ID, IMPOSTOR_ID, 23, 4, IMPOSTOR_ID])
    y1 = torch.LongTensor([15, 2, IMPOSTOR_ID, 23, IMPOSTOR_ID, IMPOSTOR_ID])
    x2 = torch.LongTensor([IMPOSTOR_ID, 2, IMPOSTOR_ID])
    y2 = torch.LongTensor([IMPOSTOR_ID, 2, 3])

    def test_impostor_accuracy(self):
        metric = ImpostorAccuracy()
        metric(self.x1, self.y1)
        metric(self.x2, self.y2)
        self.assertEqual(torch.tensor(3 / 4), metric.compute())

    def test_member_accuracy(self):
        metric = MemberAccuracy()
        metric(self.x1, self.y1)
        metric(self.x2, self.y2)
        self.assertEqual(torch.tensor(3 / 5), metric.compute())
