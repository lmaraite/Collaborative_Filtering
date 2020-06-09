import unittest
import numpy as np
import math
import pytest

from evaluation import accuracy as ac
from evaluation import selection
from evaluation.accuracy import SinglePredictionAccuracyEvaluationPropertiesBuilder, SinglePredictionAccuracyEvaluationProperties
from similarity.similarity import PEARSON, COSINE, ITEM_BASED, USER_BASED
import prediction.prediction as prediction
from prediction.data import dataset

def assert_are_same_matrices(first_matrix, second_matrix):
    assert first_matrix.shape == second_matrix.shape
    assert (first_matrix == second_matrix).all()

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

    def test_root_mean_squared_error_for_3_values(self):
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

class MeanAbsoluteErrorTest(unittest.TestCase):

    def test_mean_absolute_error_for_4_values(self):
        #given
        predictions = [4, 1, 2]
        ratings = [2, 3, 3]

        #when
        error = ac.mean_absolute_error(predictions, ratings)

        #then
        assert math.isclose(error, (5 / 3))

class SinglePredictionAccuracyEvaluationPropertiesBuilderTest(unittest.TestCase):

    dummy_function = lambda x, y, z: x

    def test_builder_should_throw_error_without_error_measurement(self):
        with pytest.raises(ValueError):
            SinglePredictionAccuracyEvaluationPropertiesBuilder() \
                .with_is_rated_matrix(np.array([]), 0) \
                .with_ratings_matrix(np.array([]), 0) \
                .with_similarity("test") \
                .with_selection_strategy(self.dummy_function) \
                .with_approach("") \
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
            .with_ratings_matrix(ratings_matrix, 1) \
            .with_is_rated_matrix(is_rated, 1) \
            .with_similarity(COSINE) \
            .with_approach(ITEM_BASED) \
            .with_selection_strategy(self.dummy_function) \
            .with_error_measurement(self.dummy_function) \
            .build()

        #then
        assert (evaluation_properties.ratings_matrix == ratings_matrix).all()
        assert (evaluation_properties.is_rated_matrix == is_rated).all()
        assert evaluation_properties.similarity == COSINE
        assert evaluation_properties.approach == ITEM_BASED
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

def test_builder_prediction_function_with_item_based(mocker):
    #given
    mock_pred_func = mocker.patch("prediction.prediction.predicition_cosine_similarity")

    evaluation_properties_builder = SinglePredictionAccuracyEvaluationPropertiesBuilder() \
        .with_similarity(COSINE) \
        .with_approach(ITEM_BASED)

    #when
    prediction_function = evaluation_properties_builder.prediction_function

    #then
    data = dataset(None, None, None)
    prediction_function(1, 2, data)
    mock_pred_func.assert_called_once_with(1, 2, data)



def test_builder_prediction_function_with_user_based(mocker):
    #given
    mock_pred_func = mocker.patch("prediction.prediction.predicition_cosine_similarity")

    evaluation_properties_builder = SinglePredictionAccuracyEvaluationPropertiesBuilder() \
        .with_similarity(COSINE) \
        .with_approach(USER_BASED)

    #when
    prediction_function = evaluation_properties_builder.prediction_function

    #then
    data = dataset(None, None, None)
    prediction_function(1, 2, data)
    mock_pred_func.assert_called_once_with(2, 1, data)


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

    prediction_return = 3, 1
    error_return = 1

    #callbacks with assertions
    def callback_selection_strategy_mock(self, shape, is_rated, train_size):
        assert shape == (2, 3)
        assert_are_same_matrices(is_rated, self.is_rated_matrix)
        assert train_size == self.train_size

        self.selected_train_indices_iter = iter(self.selected_train_indices)

        yield (self.selected_train_indices_iter, iter(self.selected_test_indices))

    def callback_filter_mock(self, matrix, indices, baseValue):
        for first, second in zip(indices, iter(self.selected_train_indices)):
            assert (first == second).all()
        assert_are_same_matrices(matrix, self.is_rated_matrix)
        assert baseValue == False

        return self.filtered_is_rated

    def callback_similarity_creation_mock(self, approach, algorithm, all_ratings, is_rated):
        assert (all_ratings == self.ratings_matrix).all()
        assert (is_rated == self.filtered_is_rated).all()
        assert algorithm == COSINE
        assert approach == ITEM_BASED

        return self.similarity_matrix

    def callback_prediction_creation_mock(self, key_id, element_id, data):
        assert key_id == 0
        assert element_id == 2
        assert_are_same_matrices(data.similarity_matrix, self.similarity_matrix)
        assert_are_same_matrices(data.rating_matrix, self.ratings_matrix)
        assert_are_same_matrices(data.is_rated_matrix, self.filtered_is_rated)

        return self.prediction_return

    def callback_error_measurement_mock(self, predictions, ratings):
        assert predictions == [self.prediction_return[0]]
        assert ratings == [self.ratings_matrix[0, 2]]

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
            "similarity.similarity.create_similarity_matrix"
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
            ITEM_BASED,
            ac.root_mean_squared_error,
            prediction.predicition_cosine_similarity
        )

        #when
        error = ac.run_accuracy_evaluation(eval_props)
        #then
        assert error == self.error_return

@pytest.fixture
def mock_empty(mocker):
    mocker.patch("evaluation.selection.select_indices_with_hold_out")
    mocker.patch("evaluation.selection.keep_elements_by_index")
    mocker.patch("similarity.similarity.create_similarity_matrix")
    mocker.patch("prediction.prediction.predicition_cosine_similarity")
    mocker.patch("evaluation.accuracy.root_mean_squared_error")

def cross_validation_mock_callback(shape, is_rated_matrix, train_size):
    yield iter([(1, 0), (2, 0)]), iter([(0, 0)])
    yield iter([(0, 0)]), iter([(1, 0), (2, 0)])

def prediction_function_mock_callback(key_id, element_id, dataset):
    assert key_id in [0, 1, 2]
    assert element_id == 0

    if key_id == 0:
        return 2, -1
    if key_id == 1:
        return 5, -1
    if key_id == 2:
        return 1, -1

def error_function_mock_callback(predictions, actual_ratings):
    assert predictions in [[2], [5, 1]]
    assert actual_ratings in [[4], [1, 4]]

    if predictions == [2] and actual_ratings == [4]:
        return 2
    elif predictions == [5, 1] and actual_ratings == [1, 4]:
        return 3.5

def test_run_accuracy_evaluation_error_with_multiple_splits(mock_empty, mocker):
    #given
    select_strategy_mock = mocker.patch("evaluation.selection.select_indices_with_cross_validation")
    select_strategy_mock.side_effect = cross_validation_mock_callback

    prediction_function_mock = mocker.patch("prediction.prediction.predicition_cosine_similarity")
    prediction_function_mock.side_effect = prediction_function_mock_callback

    error_function_mock = mocker.patch("evaluation.accuracy.root_mean_squared_error")
    error_function_mock.side_effect = error_function_mock_callback

    ratings_matrix = np.array([
        [4],
        [1],
        [4]
    ])
    is_rated_matrix = ratings_matrix != 0

    eval_props = SinglePredictionAccuracyEvaluationProperties(
        ratings_matrix,
        is_rated_matrix,
        COSINE,
        selection.select_indices_with_cross_validation,
        0.8,
        ITEM_BASED,
        ac.root_mean_squared_error,
        prediction.predicition_cosine_similarity
    )

    #when
    error = ac.run_accuracy_evaluation(eval_props)
    #then
    assert error == 2.75
