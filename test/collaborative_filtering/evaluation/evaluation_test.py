import unittest
import pytest
import numpy as np
import math

import evaluation
from evaluation import EvaluationPropertiesBuilder

class EvaluationPropertiesBuilderTest(unittest.TestCase):

    dummy_function = lambda x, y, z: x

    def test_builder_should_throw_error_without_is_rated_matrix(self):
        with pytest.raises(ValueError):
            EvaluationPropertiesBuilder() \
                .with_ratings_matrix(np.array([])) \
                .with_similarity("test") \
                .with_selection_strategy(self.dummy_function) \
                .build()

    def test_builder_should_throw_error_without_ratings_matrix(self):
        with pytest.raises(ValueError):
            EvaluationPropertiesBuilder() \
                .with_is_rated_matrix(np.array([])) \
                .with_similarity("test") \
                .with_selection_strategy(self.dummy_function) \
                .build()

    def test_builder_should_throw_error_without_similarity(self):
        with pytest.raises(ValueError):
            EvaluationPropertiesBuilder() \
                .with_is_rated_matrix(np.array([])) \
                .with_ratings_matrix(np.array([])) \
                .with_selection_strategy(self.dummy_function) \
                .build()

    def test_build_should_trow_error_without_selection_strategy(self):
        with pytest.raises(ValueError):
            EvaluationPropertiesBuilder() \
                .with_is_rated_matrix(np.array([])) \
                .with_ratings_matrix(np.array([])) \
                .with_similarity("test") \
                .build()

    def test_builder_should_throw_error_for_negative_train_size(self):
        with pytest.raises(ValueError):
            EvaluationPropertiesBuilder() \
                .with_train_size(-1)

    def test_builder_should_throw_error_for_train_size_greater_one(self):
        with pytest.raises(ValueError):
            EvaluationPropertiesBuilder() \
                .with_train_size(2)

    def test_build(self):
        #given
        ratings_matrix = np.array([
            [3, 1, 2],
            [5, 2, 4],
            [1, 2, 4]
        ])
        is_rated = np.array([
            [True, True, False],
            [True, False, True],
            [False, True, False]
        ])
        #when
        evaluation_properties = EvaluationPropertiesBuilder() \
            .with_ratings_matrix(ratings_matrix) \
            .with_is_rated_matrix(is_rated) \
            .with_similarity("testSimilarity") \
            .with_selection_strategy(self.dummy_function) \
            .with_train_size(0.5) \
            .build()

        #then
        assert (evaluation_properties.ratings_matrix == ratings_matrix).all()
        assert (evaluation_properties.is_rated_matrix == is_rated).all()
        assert (evaluation_properties.similarity == "testSimilarity")
        assert (evaluation_properties.selection_strategy == self.dummy_function)
        assert (evaluation_properties.train_size == 0.5)
        assert evaluation_properties.approach == ITEM_BASED

    def test_builder_with_ratings_matrix_and_user_axis_0(self):
        #given
        ratings_matrix = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])

        #when
        evaluation_properties_builder = EvaluationPropertiesBuilder() \
            .with_ratings_matrix(ratings_matrix, 0)

        #then
        transposed_ratings = np.array([
            [1, 4],
            [2, 5],
            [3, 6]
        ])
        assert evaluation_properties_builder.ratings_matrix.shape == transposed_ratings.shape
        assert (evaluation_properties_builder.ratings_matrix == transposed_ratings).all()

    def test_builder_with_ratings_matrix_and_user_axis_1(self):
        #given
        ratings_matrix = np.array([
            [1, 2],
            [3, 4]
        ])

        #when
        evaluation_properties_builder = EvaluationPropertiesBuilder() \
            .with_ratings_matrix(ratings_matrix, 1)

        #then
        assert evaluation_properties_builder.ratings_matrix.shape == ratings_matrix.shape
        assert (evaluation_properties_builder.ratings_matrix == ratings_matrix).all()

    def test_builder_with_is_rated_matrix_and_user_axis_0(self):
        #given
        is_rated_matrix = np.array([
            [True, True],
            [False, False]
        ])

        #when
        evaluation_properties_builder = EvaluationPropertiesBuilder() \
            .with_is_rated_matrix(is_rated_matrix, 0)

        #then
        transposed_is_rated = np.array([
            [True, False],
            [True, False]
        ])

        assert (evaluation_properties_builder.is_rated_matrix == transposed_is_rated).all()

    def test_builder_with_is_rated_matrix_and_user_axis_1(self):
        #given
        is_rated_matrix = np.array([
            [True, True],
            [False, False]
        ])

        #when
        evaluation_properties_builder = EvaluationPropertiesBuilder() \
            .with_is_rated_matrix(is_rated_matrix, 1)

        #then
        assert (evaluation_properties_builder.is_rated_matrix == is_rated_matrix).all()


def test_analyse_data_set():
    #given
    ratings_matrix = np.array([
        [3, 4, 0, 0],
        [5, 0, 1, 2],
        [0, 0, 3, 2]
    ])

    is_rated_matrix = ratings_matrix != 0
    #when
    analysis = evaluation.analyse_data_set(ratings_matrix, is_rated_matrix, 1)
    #then
    assert analysis["num_of_movies"] == 3
    assert analysis["num_of_users"] == 4
    assert analysis["num_of_ratings"] == 7
    assert math.isclose(analysis["avg_num_ratings_per_item"], (7 / 3))
    assert analysis["avg_num_ratings_per_user"] == 1.75
    assert math.isclose(analysis["avg_rating"], (20 / 7))
