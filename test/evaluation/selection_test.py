import unittest
import pytest
import numpy as np
from numpy.random import default_rng
import numpy.random

from evaluation import selection

class SelectionTest(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        rng_mock = mocker.patch("numpy.random.default_rng", return_value=default_rng(42))

    def test_hold_out_with_train_size_05(self):
        #given
        shape = (4, 3)
        is_rated = np.array([True for x in range(12)]).reshape(shape)
        #when
        return_indices = selection.select_indices_with_hold_out(
            shape,
            is_rated,
            train_size=0.5
        )
        #then
        expected_indices = np.array([
            [0, 0],
            [2, 1],
            [2, 0],
            [3, 0],
            [3, 2],
            [1, 0],
          ])

        assert return_indices.shape == expected_indices.shape
        assert (return_indices == expected_indices).all()

    def test_hold_out_with_is_rated_matrix(self):
        #given
        shape = (4, 3)
        is_rated = np.array([
            [True, True, False],
            [True, False, True],
            [False, True, False],
            [True, True, False]
        ])

        #then
        return_indices = selection.select_indices_with_hold_out(
            shape,
            is_rated,
            train_size=0.5
        )
        #then
        expected_indices = np.array([
            (1, 2),
            (1, 0),
            (3, 1),
            (2, 1)
        ])

        assert return_indices.shape == expected_indices.shape
        assert (return_indices == expected_indices).all()

