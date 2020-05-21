import numpy as np
import math

from evaluation import EvaluationProperties, EvaluationPropertiesBuilder

class AccurancyEvaluationProperties(EvaluationProperties):

    def __init__(
        self,
        ratings_matrix: np.ndarray,
        is_rated_matrix: np.ndarray,
        similarity: str,
        selection_strategy,
        error_measurement
    ):
        super().__init__(
            ratings_matrix,
            is_rated_matrix,
            similarity,
            selection_strategy
        )
        self.error_measurement = error_measurement

class AccurancyEvaluationPropertiesBuilder(EvaluationPropertiesBuilder):

    def __init__(self):
        super().__init__()
        self.error_measurement = None

    def with_error_measurement(self, error_measurement):
        self.error_measurement = error_measurement
        return self

    def build(self):
        if not self._are_properties_complete():
            raise ValueError("Initialization not complete")

        return AccurancyEvaluationProperties(
            self.ratings_matrix,
            self.is_rated_matrix,
            self.similarity,
            self.selection_strategy,
            self.error_measurement
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
