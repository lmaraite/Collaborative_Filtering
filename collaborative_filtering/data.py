import numpy as np

class dataset:
    def __init__(self, similarity_matrix: np.ndarray, rating_matrix: np.ndarray, is_rated_matrix: np.ndarray):
        self.similarity_matrix = similarity_matrix
        self.rating_matrix = rating_matrix
        self.is_rated_matrix = is_rated_matrix
