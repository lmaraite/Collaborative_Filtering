import numpy as np
import similarity.similarity as similarity

class DatasetBuilder:
    def __init__(self):
        self.rating_matrix = None
        self.is_rated_matrix = None
        self.similarity_matrix = None
        self.approach = None

    def with_rating_matrix(self, rating_matrix, user_axis):
        self.rating_matrix = rating_matrix

        if user_axis != 1:
            self.rating_matrix = self.rating_matrix.T

        return self

    def with_is_rated_matrix(self, is_rated_matrix, user_axis):
        self.is_rated_matrix = is_rated_matrix

        if user_axis != 1:
            self.is_rated_matrix = self.is_rated_matrix.T

        return self

    def with_similarity_matrix(self, similarity_matrix):
        self.similarity_matrix = similarity_matrix
        return self

    def with_approach(self, approach):
        if not approach in [similarity.USER_BASED, similarity.ITEM_BASED]:
            raise ValueError("approach is not valid")
        self.approach = approach

        return self

    def build(self):
        self._check_if_complete()
        rating_matrix = self.rating_matrix
        is_rated_matrix = self.is_rated_matrix
        if self.approach == similarity.USER_BASED:
            rating_matrix = rating_matrix.T
            is_rated_matrix = is_rated_matrix.T


        return dataset(
            self.similarity_matrix,
            rating_matrix,
            is_rated_matrix
        )

    def _check_if_complete(self):
        if self.rating_matrix is None:
            raise ValueError("Rating Matrix is not initialized")
        if self.is_rated_matrix is None:
            raise ValueError("Is Rated Matrix is not initialized")
        if self.similarity_matrix is None:
            raise ValueError("Similarity Matrix is not initialized")
        if self.approach is None:
            raise ValueError("Approach is not initialized")

class dataset:
    def __init__(self, similarity_matrix: np.ndarray, rating_matrix: np.ndarray, is_rated_matrix: np.ndarray):
        self.similarity_matrix = similarity_matrix
        self.rating_matrix = rating_matrix
        self.is_rated_matrix = is_rated_matrix
