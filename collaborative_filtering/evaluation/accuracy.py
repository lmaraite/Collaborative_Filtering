import numpy as np
import math
import textwrap

from evaluation import EvaluationProperties, EvaluationPropertiesBuilder, selection
import similarity
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
        error_measurement,
        prediction
    ):
        super().__init__(
            ratings_matrix,
            is_rated_matrix,
            similarity,
            selection_strategy,
            train_size
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
        self.prediction_function = None

    def with_error_measurement(self, error_measurement):
        self.error_measurement = error_measurement
        return self

    def with_similarity(self, similarity_mode):
        super().with_similarity(similarity_mode)

        if similarity_mode == similarity.PEARSON:
            self.prediction_function = prediction.predicition_pearson_correlation
        elif similarity_mode == similarity.COSINE:
            self.prediction_function = prediction.predicition_cosine_similarity
        elif similarity_mode == similarity.ADJUSTED_COSINE:
            raise Error("Adjusted cosine is not yet implemented")

        return self

    def build(self):
        if not self._are_properties_complete():
            raise ValueError("Initialization not complete")

        return SinglePredictionAccuracyEvaluationProperties(
            self.ratings_matrix,
            self.is_rated_matrix,
            self.similarity,
            self.selection_strategy,
            self.train_size,
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


def run_accuracy_evaluation(eval_props: SinglePredictionAccuracyEvaluationProperties):
    train_indices, test_indices = eval_props.selection_strategy(
        eval_props.ratings_matrix.shape,
        eval_props.is_rated_matrix,
        eval_props.train_size
    )

    kept_is_rated_matrix = selection.keep_elements_by_index(
        eval_props.is_rated_matrix,
        train_indices,
        False
    )

    similarity_matrix = similarity.create_similarity_matrix(
        eval_props.ratings_matrix,
        kept_is_rated_matrix,
        eval_props.similarity
    )

    dataset = data.dataset(
        similarity_matrix,
        eval_props.ratings_matrix,
        kept_is_rated_matrix
    )
    predictions = np.empty(test_indices.shape[0])
    actual_ratings = np.empty(test_indices.shape[0])

    for i in range(test_indices.shape[0]):
        test_index = test_indices[i]
        predictions[i], _ = eval_props.prediction_function(
            test_index[0],
            test_index[1],
            dataset
        )
        actual_ratings[i] = eval_props.ratings_matrix[test_index[0], test_index[1]]

    return eval_props.error_measurement(
        predictions,
        actual_ratings
    )

_names = {
    root_mean_squared_error: "Root Mean Squared Error"
}
