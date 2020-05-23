from collaborative_filtering.prediction.prediction import get_top_n_list, predicition_cosine_similarity
from collaborative_filtering.prediction.data import dataset
import numpy as np

similarity_matrix = np.genfromtxt("../../output/item_based_adjusted_cosine_similarity_matrix.csv", delimiter=",")
rating_matrix = np.genfromtxt("../../dataset/R.csv", delimiter=",")
is_rated_matrix = np.genfromtxt("../../dataset/Y.csv", delimiter=",").astype(bool)
data = dataset(similarity_matrix, rating_matrix, is_rated_matrix)

print(predicition_cosine_similarity(0, 0, data))
top_n_list = get_top_n_list(10, 0, data)
for it in top_n_list:
    print("--------------------")
    print("movie_id:          "+str(it[0]))
    print("rating_prediction: "+str(it[1]))
    print("--------------------")

