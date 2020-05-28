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
        return_train_indices, return_test_indices = selection.select_indices_with_hold_out(
            shape,
            is_rated,
            train_size=0.5
        )
        #then
        expected_train_indices = np.array([
            [0, 0],
            [2, 1],
            [2, 0],
            [3, 0],
            [3, 2],
            [1, 0],
          ])

        expected_test_indices = np.array([
            [1, 2],
            [0, 2],
            [1, 1],
            [3, 1],
            [0, 1],
            [2, 2]
        ])

        assert return_train_indices.shape == expected_train_indices.shape
        assert (return_train_indices == expected_train_indices).all()
        assert (return_test_indices == expected_test_indices).all()

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
        return_train_indices, return_test_indices = selection.select_indices_with_hold_out(
            shape,
            is_rated,
            train_size=0.5
        )
        #then
        expected_train_indices = np.array([
            (1, 2),
            (1, 0),
            (3, 1),
            (2, 1)
        ])

        expected_test_indices = np.array([
            [0, 1],
            [3, 0],
            [0, 0]
        ])

        assert return_train_indices.shape == expected_train_indices.shape
        assert (return_train_indices == expected_train_indices).all()
        assert (return_test_indices == expected_test_indices).all()

    def test_keep_elements_by_index(self):
        #given
        ratings_matrix = np.array([
            [3, 1, 2],
            [5, 2, 4],
            [1, 2, 4]
        ])

        indices = np.array([
            [0, 0], [0, 1],
            [1, 1], [1, 2],
            [2, 0]
        ])

        #when
        filtered_matrix = selection.keep_elements_by_index(ratings_matrix, indices, 0)

        #then
        expected_matrix = np.array([
            [3, 1, 0],
            [0, 2, 4],
            [1, 0, 0]
        ])

        assert (filtered_matrix == expected_matrix).all()

    def test_cross_validation_with_train_size_08(self):
        #given
        shape = (1, 5)
        is_rated = np.array([True for x in range(5)]).reshape(shape)

        #when
        train_test_sets = selection.select_indices_with_cross_validation(shape, is_rated, 0.8)

        #then
        expected_train_indices = np.array([
            [(0, 4), (0, 2), (0, 3), (0, 1)],
            [(0, 4), (0, 2), (0, 3), (0, 0)],
            [(0, 4), (0, 2), (0, 1), (0, 0)],
            [(0, 4), (0, 3), (0, 1), (0, 0)],
            [(0, 2), (0, 3), (0, 1), (0, 0)]
        ])

        expected_test_indices = np.array([
            [(0, 0)],
            [(0, 1)],
            [(0, 3)],
            [(0, 2)],
            [(0, 4)]
        ])

        counter_splits = 0
        for train_return, test_return in train_test_sets:
            assert list(train_return) in expected_train_indices
            assert list(test_return) in expected_test_indices
            counter_splits += 1

        assert counter_splits == 5

    def test_cross_validation_with_is_rated_matrix(self):
        #given
        shape = (1, 6)
        is_rated = np.array([True, True, True, True, True, False])

        #when
        train_test_sets = selection.select_indices_with_cross_validation(shape, is_rated, 0.8)

        #then
        expected_train_indices = np.array([
            [(0, 4), (0, 2), (0, 3), (0, 1)],
            [(0, 4), (0, 2), (0, 3), (0, 0)],
            [(0, 4), (0, 2), (0, 1), (0, 0)],
            [(0, 4), (0, 3), (0, 1), (0, 0)],
            [(0, 2), (0, 3), (0, 1), (0, 0)]
        ])

        expected_test_indices = np.array([
            [(0, 0)],
            [(0, 1)],
            [(0, 3)],
            [(0, 2)],
            [(0, 4)]
        ])

        counter_splits = 0
        for train_return, test_return in train_test_sets:
            assert list(train_return) in expected_train_indices
            assert list(test_return) in expected_test_indices
            counter_splits += 1

        assert counter_splits == 5

    def test_cross_validation_with_undevidable_shape(self):
        # segments = [
        #     [(0, 4), (0, 1],
        #     [(0, 2), (0, 0)],
        #     [(0, 3)]
        # ]
        #given
        shape = (1, 5)
        is_rated = np.array([True for x in range(5)]).reshape(shape)

        #when
        train_test_sets = selection.select_indices_with_cross_validation(shape, is_rated, 0.7)

        #then
        expected_train_indices = [
            [[0, 4], [0, 1], [0, 2], [0, 0]],
            [[0, 4], [0, 1], [0, 3]],
            [[0, 2], [0, 0], [0, 3]],
        ]

        expected_test_indices = [
            [[0, 3]],
            [[0, 4], [0, 1]],
            [[0, 2], [0, 0]]
        ]

        counter_splits = 0
        for train_return, test_return in train_test_sets:
            train_return = map(
                lambda pair: pair.tolist(),
                train_return
            )
            test_return = map(
                lambda pair: pair.tolist(),
                test_return
            )
            assert list(train_return) in expected_train_indices
            assert list(test_return) in expected_test_indices
            counter_splits += 1

        assert counter_splits == 3
