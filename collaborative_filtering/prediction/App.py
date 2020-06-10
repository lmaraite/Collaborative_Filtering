import numpy as np
from prediction import get_top_n_list
from data import dataset

similarity_matrix = np.genfromtxt("../../output/item_based_adjusted_cosine_similarity_matrix.csv", delimiter=",")
rating_matrix = np.genfromtxt("../../dataset/R.csv", delimiter=",")
is_rated_matrix = np.genfromtxt("../../dataset/Y.csv", delimiter=",").astype(bool)
data = dataset(similarity_matrix, rating_matrix, is_rated_matrix)
movies = []
with open("../../dataset/movie_ids.txt") as file:
    for line in file:
        movies.append(line.split(";")[1].replace("\n", ""))
movies = np.array(movies)


def has_rated(user_id: int, item_id: int) -> bool:
    return is_rated_matrix[item_id, user_id]


def get_rating(user_id:int, item_id: int) -> bool:
    return rating_matrix[item_id, user_id]


def user_ratings(user_id: int) -> list:
    ratings = []
    for i in range(0, len(movies)):
        if has_rated(user_id, i):
            ratings.append((movies[i], get_rating(user_id, i)))
    return ratings


def user_best_ratings(user_ratings: list, count: int) -> list:
    sorted_list = sorted(user_ratings, key=lambda item: item[1], reverse=True)
    return sorted_list[:count]


def get_top_n(user_id: int, n: int) -> list:
    top_n = get_top_n_list(n, user_id, data, "cosine")
    formatted_top_n = []
    for movie in top_n:
        formatted_top_n.append((movies[movie[0]], movie[1][0]))
    return formatted_top_n


while True:
    alice = int(input("Please enter your user id: "))
    alice_ratings = user_ratings(alice)
    print("Your favourite movies are: ")
    for movie in user_best_ratings(alice_ratings, 15):
        print(movie)
    print("\nWe would recommend you: ")
    for movie in get_top_n(alice, 30):
        print(movie)
    print()