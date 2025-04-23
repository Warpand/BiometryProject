from unittest import TestCase

import numpy as np
import pandas as pd

import data.pandas_utils as utils


class TestPandasUtils(TestCase):
    DF = pd.DataFrame({"sorted": [0, 0, 1, 1, 4, 5, 7, 8, 8, 9]})

    def test_l_than(self):
        result = utils.l_than(self.DF, "sorted", 7)["sorted"]
        self.assertTrue(np.array_equal(np.array([0, 0, 1, 1, 4, 5]), result))

    def test_ge_than(self):
        result = utils.ge_than(self.DF, "sorted", 1)["sorted"]
        self.assertTrue(np.array_equal(np.array([1, 1, 4, 5, 7, 8, 8, 9]), result))

    def test_between(self):
        result = utils.between(self.DF, "sorted", 1, 8)["sorted"]
        self.assertTrue(np.array_equal(np.array([1, 1, 4, 5, 7]), result))
