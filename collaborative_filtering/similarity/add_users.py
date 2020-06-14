import sys, site, os

site.addsitedir(os.path.join(os.path.abspath(sys.path[0]), ".."))

import numpy as np
from input.filesystem import read_ratings_matrix, read_is_rated_matrix

# The ratings of the new users are saved in a lists of tuples.
raw_ratings_new_user1 = [(221, 5),      # Star Trek: First Contact
                         (226, 5),      # Star Trek VI
                         (227, 5),      # Star Trek: Khan
                         (110, 1),      # Truth about cats and dogs
                         (120, 1),      # Independence Day
                         (85, 1)]       # Remains of the Day

raw_ratings_new_user2 = [(0, 5),        # Toy Story
                         (16, 1),       # From Dusk Till Dawn
                         (55, 1),       # Pulp Fiction
                         (70, 5),       # Lion King
                         (94, 5),       # Aladdin
                         (143, 1),      # Die Hard
                         (216, 1),      # Bram Stokers Dracula
                         (218, 1),      # Nightmare on Elm Street
                         (225, 1),      # Die Hard 2
                         (230, 3),      # Batman Returns
                         (312, 2),      # Titanic
                         (402, 3),      # Batman
                         (464, 5),      # Jungle Book
                         (589, 1),      # Hellraiser: Bloodline
                         (901, 2),      # Big Lebowski
                         (1016, 1),     # Con Air
                         (1126, 2)]     # Truman Show

raw_ratings_new_user3 = [(0, 1),        # Toy Story
                         (16, 5),       # From Dusk Till Dawn
                         (55, 5),       # Pulp Fiction
                         (70, 1),       # Lion King
                         (94, 1),       # Aladdin
                         (143, 5),      # Die Hard
                         (216, 3),      # Bram Stokers Dracula
                         (218, 3),      # Nightmare on Elm Street
                         (225, 5),      # Die Hard 2
                         (230, 4),      # Batman Returns
                         (312, 1),      # Titanic
                         (402, 3),      # Batman
                         (464, 1),      # Jungle Book
                         (589, 2),      # Hellraiser: Bloodline
                         (901, 2),      # Big Lebowski
                         (1016, 5),     # Con Air
                         (1126, 1)]     # Truman Show

new_users = [raw_ratings_new_user1, raw_ratings_new_user2, raw_ratings_new_user3]

# Loading the original matrices.
ratings_matrix = read_ratings_matrix()
is_rated_matrix = read_is_rated_matrix()

for raw_ratings_new_user in new_users:
    # Preparing the raw ratings.
    number_of_movies = ratings_matrix.shape[0]
    ratings_new_user = np.zeros(number_of_movies)
    is_rated_new_user = np.full(number_of_movies, False)
    for rating in raw_ratings_new_user:
        ratings_new_user[rating[0]] = rating[1]
        is_rated_new_user[rating[0]] = True

    # Adding the new user to the original matrices.
    ratings_matrix = np.hstack((ratings_matrix, np.vstack(ratings_new_user)))
    is_rated_matrix = np.hstack((is_rated_matrix, np.vstack(is_rated_new_user)))
    print(ratings_matrix.shape)

# Saving the modified matrices to files.
# Attempts to save these files directly to ../output/ have not been successful. They need to be moved manually.
np.savetxt("modified_ratings_matrix.csv", ratings_matrix, delimiter=",", fmt="%d")
np.savetxt("modified_is_rated_matrix.csv", is_rated_matrix, delimiter=",", fmt="%d")
