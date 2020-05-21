import unittest
import pytest
import numpy as np

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
