#!/usr/bin/env python3
from data import dataset
from neighborhood import get_nearest_neighbors, has_rated
import numpy as np
import math

"""
Item based:
    key_id     = movie_id
    element_id = user_id

User based:
    key_id     = user_id
    element_id = movie_id
"""

MAX_NEAREST_NEIGHBORS = 10


similarity_matrix = np.genfromtxt("dataset/cosine-similarity-matrix.csv", delimiter=",")
rating_matrix = np.genfromtxt("dataset/R.csv", delimiter=",")
is_rated_matrix = np.genfromtxt("dataset/Y.csv", delimiter=",").astype(bool)


def predicition_cosine_similarity(key_id: int, element_id: int, data: dataset) -> float:
    nearest_neighbors = get_nearest_neighbors(MAX_NEAREST_NEIGHBORS, key_id, element_id, data)
    counter = 0
    denominator = 0
    for it in nearest_neighbors:
        counter += it.similarity * it.rating
        denominator += it.similarity

        """
        print("sim: "+str(it.similarity))
        print(math.isnan(it.similarity))
        print("id:  "+str(it.key_id))
        print("r:   "+str(it.rating))
        print("---")
        """

    prediction = counter / denominator
    if denominator == 0:
        prediction = 0
    return prediction


def add_pearson_average(key_id: int, nearest_neighbors: list, data: dataset):
    for it in nearest_neighbors:
        rating_vector = data.rating_matrix[it.key_id]
        intersection = np.logical_and(data.is_rated_matrix[key_id], data.is_rated_matrix[it.key_id])
        intersected_rating_vector = rating_vector[intersection]
        it.pearson_average = np.average(intersected_rating_vector)


def predicition_pearson_correlation(key_id: int, element_id: int, data: dataset) -> float:
    nearest_neighbors = get_nearest_neighbors(MAX_NEAREST_NEIGHBORS, key_id, element_id, data)
    add_pearson_average(key_id, nearest_neighbors, data)
    counter = 0
    denominator = 0
    for it in nearest_neighbors:
        counter += it.similarity * (it.rating - it.pearson_average)
        denominator += it.similarity
    return counter / denominator


def get_top_n_list(n: int, user_id, data: dataset) -> list:
    top_n = {}
    for it in range(0, len(data.rating_matrix)):
        if not has_rated(it, user_id, data.is_rated_matrix):
            top_n[it] = predicition_cosine_similarity(it, user_id, data)
    top_n = sorted(top_n.items(), key=lambda item: item[1], reverse=True)
    return top_n[:n]


test_data = dataset(similarity_matrix, rating_matrix, is_rated_matrix)
<<<<<<< HEAD:prediction.py

nan_prediction = 295
not_nan_prediction = 287

#print(predicition_cosine_similarity(nan_prediction,0,test_data))
#for n in range(0,30):
#    print(has_rated(0,n,test_data.is_rated_matrix))


#"""
top_n_list = get_top_n_list(20, 0, test_data)
=======
print(predicition_cosine_similarity(4, 0, test_data))
"""
top_n_list = get_top_n_list(5, 0, test_data)
>>>>>>> 45674a75badb26cd16ff388aa3bbcd965d7d6017:collaborative_filtering/prediction.py
for it in top_n_list:
    print("--------------------")
    print("movie_id:          "+str(it[0]))
    print("rating_prediction: "+str(it[1]))
    print("--------------------")
<<<<<<< HEAD:prediction.py






#"""
=======
"""
>>>>>>> 45674a75badb26cd16ff388aa3bbcd965d7d6017:collaborative_filtering/prediction.py
