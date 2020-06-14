from prediction import get_top_n_list, predicition_cosine_similarity, predicition_pearson_correlation
from data import dataset
import numpy as np
import sys


similarity_matrix = np.genfromtxt("../../output/item_based_pearson_similarity_matrix.csv", delimiter=",")
rating_matrix = np.genfromtxt("../../dataset/R.csv", delimiter=",")
is_rated_matrix = np.genfromtxt("../../dataset/Y.csv", delimiter=",").astype(bool)
data_item = dataset(similarity_matrix, rating_matrix, is_rated_matrix)



similarity_matrix = np.genfromtxt("../../output/user_based_pearson_similarity_matrix.csv", delimiter=",")
rating_matrix = np.genfromtxt("../../dataset/R.csv", delimiter=",")
is_rated_matrix = np.genfromtxt("../../dataset/Y.csv", delimiter=",").astype(bool)
data_user = dataset(similarity_matrix, rating_matrix.transpose(), is_rated_matrix.transpose())

user_id = 155
n = 10
algorithm = "pearson"
top_n_list_item = get_top_n_list(n, user_id, data_item, algorithm, "item")
top_n_list_user = get_top_n_list(n, user_id, data_user, algorithm, "user")

for it in range(0, len(top_n_list_user)):
    print("--------------------")
    print("item-based: "+str(top_n_list_item[it][0])+" | "+str(top_n_list_item[it][1]))
    print("user-based: "+str(top_n_list_item[it][0])+" | "+str(top_n_list_item[it][1]))
    print("--------------------")

