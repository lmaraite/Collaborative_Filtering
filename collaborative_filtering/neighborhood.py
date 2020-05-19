#!/usr/bin/env python3
import numpy as np
from data import dataset

"""
Item based:
    key_id     = movie_id
    element_id = user_id

User based:
    key_id     = user_id
    element_id = movie_id
"""


class Neighbor:
    def __init__(self, rating: int, key_id: int, similarity: float, pearson_average: float):
        self.rating = rating
        self.key_id = key_id
        self.similarity = similarity
        self.pearson_average = pearson_average

    def __lt__(self, neighbor2):
        return self.similarity < neighbor2.similarity


def has_rated(key_id: int, element_id: int, is_rated_matrix: np.ndarray) -> bool:
    return bool(is_rated_matrix[key_id][element_id])


def get_neighbors(key_id: int, element_id: int, data: dataset) -> list:
    neighbors = []
    for it_key_id in range(0, len(data.similarity_matrix)-1):  # it = iterator
        if has_rated(it_key_id, element_id, data.is_rated_matrix):
            neighbors.append(Neighbor(data.rating_matrix[it_key_id][element_id]
                                      , it_key_id
                                      , data.similarity_matrix[key_id][it_key_id], 0))  # pearson_average wird standartmäßig auf
    return neighbors                                                                    # 0 gesetzt und später in der prediction initialisiert


def get_nearest_neighbors(max_nearest_neighbors: int , key_id: int, element_id: int, data: dataset) -> list:
    neighbors = sorted(get_neighbors(key_id, element_id, data), reverse=True)
    return neighbors[:max_nearest_neighbors]
