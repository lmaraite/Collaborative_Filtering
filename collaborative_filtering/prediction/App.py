#!/usr/bin/env python3
import numpy as np
from prediction import get_top_n_list
from data import dataset


movies = []
with open("../../dataset/movie_ids.txt") as file:
    for line in file:
        movies.append(line.split(";")[1].replace("\n", ""))
movies = np.array(movies)


def has_rated(user_id: int, item_id: int) -> bool:
    return is_rated_matrix[item_id, user_id]


def get_rating(user_id: int, item_id: int) -> bool:
    return rating_matrix[item_id, user_id]


def user_ratings(user_id: int) -> list:
    ratings = []
    for i in range(0, len(movies)):
        if has_rated(user_id, i):
            ratings.append((movies[i], get_rating(user_id, i)))
    return ratings


def user_best_ratings(user_ratings: list, count: int) -> list:
    sorted_list = sorted(user_ratings, key=lambda item: item[1], reverse=True)
    for item in sorted_list:
        if item[1] < 3:
            del item
    return sorted_list[:count]


def get_top_n(user_id: int, n: int, algorithm: str, method: str) -> list:
    top_n = get_top_n_list(n, user_id, data, algorithm, method)
    formatted_top_n = []
    for movie in top_n:
        rating = movie[1][0]
        if rating > 5:
            rating = 5
        formatted_top_n.append((movies[movie[0]], rating))
    return formatted_top_n


alg = input("Algorithm (adjusted_cosine / cosine / pearson): ")
base = input("Method (item / user): ")

similarity_matrix = np.genfromtxt("../../output/modified_" + base + "_based_" + alg + "_similarity_matrix.csv",
                                  delimiter=",")
rating_matrix = np.genfromtxt("../../output/modified_ratings_matrix.csv", delimiter=",")
is_rated_matrix = np.genfromtxt("../../output/modified_is_rated_matrix.csv", delimiter=",").astype(bool)
data = dataset(similarity_matrix, rating_matrix, is_rated_matrix)

if alg == "adjusted_cosine":
    alg = "cosine"

if base == "user":
    data.similarity_matrix = similarity_matrix.T

while True:
    alice = int(input("Please enter your user id: "))
    alice_ratings = user_ratings(alice)
    print("Your favourite movies are: ")
    for movie in user_best_ratings(alice_ratings, 10):
        print(movie)
    print("\nWe would recommend you: ")
    for movie in get_top_n(alice, 10, alg, base):
        print(movie)
    print()
