#!/usr/bin/env python3
import numpy as np

similarity_matrix = np.genfromtxt("small-dataset/cosine-similarity-matrix.csv", delimiter=",")
rating_matrix = np.genfromtxt("small-dataset/R.csv", delimiter=",")
is_rated_matrix = np.genfromtxt("small-dataset/Y.csv", delimiter=",")

"""
Item based:
    key_id     = movie_id
    element_id = user_id

User based:
    key_id     = user_id
    element_id = movie_id
"""

class neighbor:
    def __init__(self, rating: int, key_id: int, similarity: float):
        self.rating = rating
        self.key_id = key_id
        self.similarity = similarity

    def __lt__(self, neighbor2):
        return self.similarity < neighbor2.similarity

class dataset:
    def __init__(self, similarity_matrix: np.ndarray, rating_matrix: np.ndarray, is_rated_matrix: np.ndarray):
        self.similarity_matrix = similarity_matrix
        self.rating_matrix = rating_matrix
        self.is_rated_matrix = is_rated_matrix

def has_rated(key_id: int, element_id: int, is_rated_matrix: np.ndarray) -> bool:
    return bool(is_rated_matrix[key_id][element_id])


def get_neighbors(key_id: int, element_id: int, data: dataset) -> list:
    neighbors = []
    for it_key_id in range(0, len(data.similarity_matrix)-1):  # it = iterator
        if has_rated(it_key_id, element_id, data.is_rated_matrix):
            neighbors.append(neighbor(data.rating_matrix[it_key_id][element_id]
                            , it_key_id
                            , data.similarity_matrix[key_id][it_key_id]))
    return neighbors


def get_nearest_neighbors(max_nearest_neighbors: int , key_id: int, element_id: int, data: dataset) -> list:
    neighbors = sorted(get_neighbors(key_id, element_id, data), reverse=True)
    if len(neighbors) <= max_nearest_neighbors:
        return neighbors
    else:
        return neighbors[:max_nearest_neighbors]




test_data = dataset(similarity_matrix, rating_matrix, is_rated_matrix)
for e in get_nearest_neighbors(max_nearest_neighbors=5, key_id=4, element_id=0, data=test_data):
    print("----------------")
    print(e.rating)
    print(e.key_id)
    print(e.similarity)
