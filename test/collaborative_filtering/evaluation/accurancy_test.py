import unittest
import numpy as np
import math
import pytest

from evaluation import accurancy as ac
from evaluation.accurancy import AccurancyEvaluationPropertiesBuilder

class ErrorTest(unittest.TestCase):

    def test_error_of_prediction_for_4_and_3(self):
        prediction = 4
        rating = 3

        assert 1 == ac.error(prediction, rating)

    def test_error_of_predicition_for_1_and_2(self):
        prediction = 1
        rating = 2

        assert -1 == ac.error(prediction, rating)

    def test_error_of_prediction_for_correct_prediction(self):
        prediction = 3
        rating = 3

        assert 0 == ac.error(prediction, rating)

class RootMeanSquaredErrorTest(unittest.TestCase):

    def test_root_mean_squared_error_for_4_values(self):
        #given
        predictions = np.array([
            4, 5, 1
        ])
        ratings = np.array([
            4, 2, 3
        ])

        #when
        error = ac.root_mean_squared_error(predictions, ratings)
        expected_error = 2.081665999

        #then
        assert math.isclose(error, expected_error)

class AccurancyEvaluationPropertiesBuilderTest(unittest.TestCase):

    dummy_function = lambda x, y, z: x

    def test_builder_should_throw_error_without_error_measurement(self):
        with pytest.raises(ValueError):
            AccurancyEvaluationPropertiesBuilder() \
                .with_is_rated_matrix(np.array([])) \
                .with_ratings_matrix(np.array([])) \
                .with_similarity("test") \
                .with_selection_strategy(self.dummy_function) \
                .build()

    def test_builder(self):
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
        evaluation_properties = AccurancyEvaluationPropertiesBuilder() \
            .with_ratings_matrix(ratings_matrix) \
            .with_is_rated_matrix(is_rated) \
            .with_similarity("testSimilarity") \
            .with_selection_strategy(self.dummy_function) \
            .with_error_measurement(self.dummy_function) \
            .build()

        #then
        assert (evaluation_properties.ratings_matrix == ratings_matrix).all()
        assert (evaluation_properties.is_rated_matrix == is_rated).all()
        assert (evaluation_properties.similarity == "testSimilarity")
        assert (evaluation_properties.selection_strategy == self.dummy_function)
        assert (evaluation_properties.error_measurement == self.dummy_function)
