import numpy as np
import functools
import textwrap

from similarity import similarity
from evaluation import selection

class EvaluationProperties(object):

    def __init__(
            self,
            ratings_matrix: np.ndarray,
            is_rated_matrix: np.ndarray,
            similarity: str,
            selection_strategy,
            train_size: float
        ):
        self.ratings_matrix = ratings_matrix
        self.is_rated_matrix = is_rated_matrix
        self.similarity = similarity
        self.selection_strategy = selection_strategy
        self.train_size = train_size

    def __str__(self):
        return textwrap.dedent("""\
        similarity: {}
        selection strategy: {}
        train size: {}
        """.format(self.similarity, selection._names[self.selection_strategy], self.train_size))

class EvaluationPropertiesBuilder(object):

    def __init__(self):
        self.ratings_matrix = None
        self.is_rated_matrix = None
        self.similarity = None
        self.selection_strategy = None
        self.train_size = 0.8

    def with_ratings_matrix(self, ratings_matrix, user_axis):
        if user_axis == 0:
            self.ratings_matrix = ratings_matrix.T
        else:
            self.ratings_matrix = ratings_matrix

        return self

    def with_is_rated_matrix(self, is_rated_matrix, user_axis):
        if user_axis == 0:
            self.is_rated_matrix = is_rated_matrix.T
        else:
            self.is_rated_matrix = is_rated_matrix

        return self

    def with_similarity(self, similarity):
        self.similarity = similarity
        return self

    def with_selection_strategy(self, selection_strategy):
        self.selection_strategy = selection_strategy
        return self

    def with_train_size(self, train_size):
        if train_size < 0:
            raise ValueError("A negative train size is not allowed")
        if train_size > 1:
            raise ValueError("A train size greater 1 is not allowed")
        self.train_size = train_size
        return self

    def build(self):
        if not self._are_properties_complete():
            raise ValueError("Initialization not complete")
        return EvaluationProperties(
            self.ratings_matrix,
            self.is_rated_matrix,
            self.similarity,
            self.selection_strategy,
            self.train_size
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
