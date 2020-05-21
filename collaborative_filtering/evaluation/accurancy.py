import numpy as np
import math

from evaluation import EvaluationProperties, EvaluationPropertiesBuilder
import similarity
import prediction.prediction as prediction

class AccurancyEvaluationProperties(EvaluationProperties):

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

class AccurancyEvaluationPropertiesBuilder(EvaluationPropertiesBuilder):

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

        return AccurancyEvaluationProperties(
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
