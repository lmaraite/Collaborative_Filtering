import numpy as np
import math
import textwrap
import statistics
import itertools
import multiprocessing
import _pickle

from evaluation import EvaluationProperties, EvaluationPropertiesBuilder, selection
from similarity import similarity
import prediction.data as data
import prediction.prediction as prediction



class SinglePredictionAccuracyEvaluationProperties(EvaluationProperties):

    def __init__(
        self,
        ratings_matrix: np.ndarray,
        is_rated_matrix: np.ndarray,
        similarity: str,
        selection_strategy,
        train_size,
        approach,
        error_measurement,
        prediction
    ):
        super().__init__(
            ratings_matrix,
            is_rated_matrix,
            similarity,
            selection_strategy,
            train_size,
            approach
        )
        self.error_measurement = error_measurement
        self.prediction_function = prediction

    def __str__(self):
        return super().__str__() + \
        textwrap.dedent("""\
        error measurement: {}
        """.format(_names[self.error_measurement]))

class SinglePredictionAccuracyEvaluationPropertiesBuilder(EvaluationPropertiesBuilder):

    def __init__(self):
        super().__init__()
        self.error_measurement = None

    def with_error_measurement(self, error_measurement):
        self.error_measurement = error_measurement
        return self

    @property
    def prediction_function(self):
        if self.similarity == similarity.PEARSON:
            prediction_function = prediction.predicition_pearson_correlation
        elif self.similarity == similarity.COSINE:
            prediction_function = prediction.predicition_cosine_similarity
        elif self.similarity == similarity.ADJUSTED_COSINE:
            raise Error("Adjusted cosine is not yet implemented")
        else:
            return None

        if self.approach == similarity.USER_BASED:
            return lambda key_id, element_id, data: prediction_function(
                element_id,
                key_id,
                data
            )

        return prediction_function

    def build(self):
        if not self._are_properties_complete():
            raise ValueError("Initialization not complete")

        return SinglePredictionAccuracyEvaluationProperties(
            self.ratings_matrix,
            self.is_rated_matrix,
            self.similarity,
            self.selection_strategy,
            self.train_size,
            self.approach,
            self.error_measurement,
            self.prediction_function
        )

    def _are_properties_complete(self):
        return super()._are_properties_complete() and not self.error_measurement is None

def error(prediction, rating):
    return prediction - rating

def root_mean_squared_error(predictions: np.array, ratings: np.array) -> float:
    return math.sqrt(
        sum(
            map(
                lambda difference: difference ** 2,
                map(
                    lambda prediction_rating: error(prediction_rating[0], prediction_rating[1]),
                    zip(predictions, ratings)
                )
            )
        ) / len(predictions)
     )

def mean_absolute_error(predictions, ratings):
    return sum(
        map(
            lambda e: abs(e),
            map(
                lambda prediction_rating: error(prediction_rating[0], prediction_rating[1]),
                zip(predictions, ratings)
            )
        )
    ) / len(predictions)

def _run_single_test_case(train_indices, test_indices, eval_props: SinglePredictionAccuracyEvaluationProperties):
    kept_is_rated_matrix = selection.keep_elements_by_index(
        eval_props.is_rated_matrix,
        train_indices,
        False
    )

    similarity_matrix = similarity.create_similarity_matrix(
        eval_props.approach,
        eval_props.similarity,
        eval_props.ratings_matrix,
        kept_is_rated_matrix
    )

    dataset = data.dataset(
        similarity_matrix,
        eval_props.ratings_matrix,
        kept_is_rated_matrix
    )
    predictions = []
    actual_ratings = []

    for test_index in test_indices:
        prediction, _ = eval_props.prediction_function(
            test_index[0],
            test_index[1],
            dataset
        )
        predictions.append(prediction)
        actual_ratings.append(eval_props.ratings_matrix[test_index[0], test_index[1]])

    return eval_props.error_measurement(predictions, actual_ratings)

def run_accuracy_evaluation(eval_props: SinglePredictionAccuracyEvaluationProperties):
    train_test_data_sets = eval_props.selection_strategy(
        eval_props.ratings_matrix.shape,
        eval_props.is_rated_matrix,
        eval_props.train_size
    )

    all_errors = []

    train_test_data_sets = list(map(
            lambda indices_tuple: (list(indices_tuple[0]), list(indices_tuple[1])),
            train_test_data_sets
        ))

    try:
        # The following two lines can't be easily tested due to limitations of Pickling of Mocks
        with multiprocessing.Pool() as pool:
            all_errors = pool.starmap(_run_single_test_case,
                map(
                    lambda tuple: (*tuple[0], tuple[1]),
                    zip(
                        train_test_data_sets,
                        itertools.repeat(eval_props)
                    )
                )
            )
    except _pickle.PicklingError as e:
        for train_indices, test_indices, props in map(
            lambda tuple: (*tuple[0], tuple[1]),
            zip(
                train_test_data_sets,
                itertools.repeat(eval_props)
            )
        ):
            all_errors.append(_run_single_test_case(train_indices, test_indices, props))

    return statistics.mean(all_errors)

_names = {
    root_mean_squared_error: "Root Mean Squared Error",
    mean_absolute_error: "Mean Absolute Error"
}
