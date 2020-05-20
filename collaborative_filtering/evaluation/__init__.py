import numpy as np
import functools
import similarity as similarity

class EvaluationProperties(object):

    def __init__(
            self,
            ratings_matrix: np.ndarray,
            is_rated_matrix: np.ndarray,
            similarity: str,
            selection_strategy
        ):
        self.ratings_matrix = ratings_matrix
        self.is_rated_matrix = is_rated_matrix
        self.similarity = similarity
        self.selection_strategy = selection_strategy

class EvaluationPropertiesBuilder(object):

    def __init__(self):
        self.ratings_matrix = None
        self.is_rated_matrix = None
        self.similarity = None
        self.selection_strategy = None

    def with_ratings_matrix(self, ratings_matrix):
        self.ratings_matrix = ratings_matrix
        return self

    def with_is_rated_matrix(self, is_rated_matrix):
        self.is_rated_matrix = is_rated_matrix
        return self

    def with_similarity(self, similarity):
        self.similarity = similarity
        return self

    def with_selection_strategy(self, selection_strategy):
        self.selection_strategy = selection_strategy
        return self

    def build(self):
        if not self._are_properties_complete():
            raise ValueError("Initialization not complete")
        return EvaluationProperties(
            self.ratings_matrix,
            self.is_rated_matrix,
            self.similarity,
            self.selection_strategy
        )


    def _are_properties_complete(self):
        if self.ratings_matrix is None:
            return False
        if self.is_rated_matrix is None:
            return False
        if self.similarity is None:
            return False
        if self.selection_strategy is None:
            return False
        return True
