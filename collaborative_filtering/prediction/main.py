#!
from prediction import get_top_n_list, predicition_cosine_similarity, predicition_pearson_correlation
from data import dataset
import numpy as np
import sys


similarity_matrix = np.genfromtxt("../../output/small_user_based_pearson_similarity_matrix.csv", delimiter=",")
rating_matrix = np.genfromtxt("../../small-dataset/R.csv", delimiter=",").T
is_rated_matrix = np.genfromtxt("../../small-dataset/Y.csv", delimiter=",").astype(bool).T
data = dataset(similarity_matrix, rating_matrix, is_rated_matrix)


print(predicition_pearson_correlation(0, 4, data))
"""
top_n_list = get_top_n_list(10, 0, data)
for it in top_n_list:
    print("--------------------")
    print("movie_id:          "+str(it[0]))
    print("rating_prediction: "+str(it[1]))
    print("--------------------")
"""
