from prediction import get_top_n_list, predicition_cosine_similarity, predicition_pearson_correlation
from data import dataset
import numpy as np
import sys


similarity_matrix = np.genfromtxt("../../output/user_based_pearson_similarity_matrix.csv", delimiter=",")
rating_matrix = np.genfromtxt("../../dataset/R.csv", delimiter=",")
is_rated_matrix = np.genfromtxt("../../dataset/Y.csv", delimiter=",").astype(bool)
data = dataset(similarity_matrix, rating_matrix, is_rated_matrix)

top_n_list = get_top_n_list(0, 0, data, "cosine")
for it in top_n_list:
    print("--------------------")
    print("movie_id:          "+str(it[0]))
    print("rating_prediction: "+str(it[1]))
    print("--------------------")

