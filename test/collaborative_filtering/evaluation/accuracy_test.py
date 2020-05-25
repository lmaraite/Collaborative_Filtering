import unittest
import numpy as np
import math
import pytest

from evaluation import accuracy as ac
from evaluation import selection
from evaluation.accuracy import SinglePredictionAccuracyEvaluationPropertiesBuilder, SinglePredictionAccuracyEvaluationProperties
from similarity import PEARSON, COSINE
import prediction.prediction as prediction

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

class SinglePredictionAccuracyEvaluationPropertiesBuilderTest(unittest.TestCase):

    dummy_function = lambda x, y, z: x

    def test_builder_should_throw_error_without_error_measurement(self):
        with pytest.raises(ValueError):
            SinglePredictionAccuracyEvaluationPropertiesBuilder() \
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
        evaluation_properties = SinglePredictionAccuracyEvaluationPropertiesBuilder() \
            .with_ratings_matrix(ratings_matrix) \
            .with_is_rated_matrix(is_rated) \
            .with_similarity(COSINE) \
            .with_selection_strategy(self.dummy_function) \
            .with_error_measurement(self.dummy_function) \
            .build()

        #then
        assert (evaluation_properties.ratings_matrix == ratings_matrix).all()
        assert (evaluation_properties.is_rated_matrix == is_rated).all()
        assert evaluation_properties.similarity == COSINE
        assert evaluation_properties.selection_strategy == self.dummy_function
        assert evaluation_properties.error_measurement == self.dummy_function
        assert evaluation_properties.prediction_function == prediction.predicition_cosine_similarity

    def test_with_pearson_similarity(self):
        #when
        builder = SinglePredictionAccuracyEvaluationPropertiesBuilder() \
            .with_similarity(PEARSON)
        #then
        assert builder.prediction_function == prediction.predicition_pearson_correlation

    def test_with_cosine_similarity(self):
        #when
        builder = SinglePredictionAccuracyEvaluationPropertiesBuilder() \
            .with_similarity(COSINE)

        assert builder.prediction_function == prediction.predicition_cosine_similarity


class AccuracyEvaluationTest(unittest.TestCase):

    #given
    train_size = 5/6
    ratings_matrix = np.array([
        [3, 1, 4],
        [4, 4, 3]
    ])
    is_rated_matrix = np.array([
        [True, True, True],
        [True, True, True]
    ])

    #returned values
    selected_train_indices = np.array([(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)])
    selected_test_indices = np.array([(0, 2)])

    filtered_is_rated = np.array([
        [True, True, False],
        [True, True, True]
    ])

    similarity_matrix = np.array([
        [1, 0.8944271910],
        [0.8944271910, 1]
    ])

    prediction_return = 3
    error_return = 1

    #callbacks with assertions
    def callback_selection_strategy_mock(self, shape, is_rated, train_size):
        assert shape == (2, 3)
        assert (is_rated == self.is_rated_matrix).all()
        assert train_size == self.train_size

        return (self.selected_train_indices, self.selected_test_indices)

    def callback_filter_mock(self, matrix, indices, baseValue):
        assert (indices == self.selected_train_indices).all()
        assert (matrix == self.is_rated_matrix).all()
        assert baseValue == False

        return self.filtered_is_rated

    def callback_similarity_creation_mock(self, all_ratings, is_rated, mode):
        assert (all_ratings == self.ratings_matrix).all()
        assert (is_rated == self.filtered_is_rated).all()
        assert mode == COSINE

        return self.similarity_matrix

    def callback_prediction_creation_mock(self, key_id, element_id, data):
        assert key_id == 0
        assert element_id == 2
        assert (data.similarity_matrix == self.similarity_matrix).all()
        assert (data.rating_matrix == self.ratings_matrix).all()
        assert (data.is_rated_matrix == self.filtered_is_rated).all()

        return self.prediction_return

    def callback_error_measurement_mock(self, predictions, ratings):
        assert (predictions == np.array([self.prediction_return])).all()
        assert (ratings == np.array([self.ratings_matrix[0, 2]])).all()

        return self.error_return

    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        self.selection_strategy_mock = mocker.patch(
            "evaluation.selection.select_indices_with_hold_out"
        )
        self.selection_strategy_mock.side_effect = self.callback_selection_strategy_mock

        self.filter_mock = mocker.patch(
            "evaluation.selection.keep_elements_by_index"
        )
        self.filter_mock.side_effect = self.callback_filter_mock

        self.similarity_creation_mock = mocker.patch(
            "similarity.create_similarity_matrix"
        )
        self.similarity_creation_mock.side_effect = self.callback_similarity_creation_mock

        self.prediction_creation_mock = mocker.patch(
            "prediction.prediction.predicition_cosine_similarity",
        )
        self.prediction_creation_mock.side_effect = self.callback_prediction_creation_mock

        self.error_measurement_mock = mocker.patch(
            "evaluation.accuracy.root_mean_squared_error",
        )
        self.error_measurement_mock.side_effect = self.callback_error_measurement_mock

    def test_run_accuracy_evaluation(self):
        #given
        eval_props = SinglePredictionAccuracyEvaluationProperties(
            self.ratings_matrix,
            self.is_rated_matrix,
            COSINE,
            selection.select_indices_with_hold_out,
            self.train_size,
            ac.root_mean_squared_error,
            prediction.predicition_cosine_similarity
        )

        #when
        error = ac.run_accuracy_evaluation(eval_props)
        #then
        assert error == self.error_return
