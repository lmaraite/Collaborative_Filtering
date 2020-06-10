import sys, site, os

site.addsitedir(os.path.join(os.path.abspath(sys.path[0]), ".."))

import numpy as np
from input.filesystem import read_ratings_matrix, read_is_rated_matrix


# The ratings of the new user is saved in a list of tuples.
raw_ratings_new_user = [(0, 5),     # Toy Story
                        (16, 4),    # From Dusk Till Dawn
                        (55, 4),    # Pulp Fiction
                        (70, 5),    # Lion King
                        (94, 5),    # Aladdin
                        (143, 4),   # Die Hard
                        (216, 2),   # Bram Stokers Dracula
                        (218, 1),   # Nightmare on Elm Street
                        (225, 4),   # Die Hard 2
                        (230, 5),   # Batman Returns
                        (312, 3),   # Titanic
                        (402, 5),   # Batman
                        (464, 5),   # Jungle Book
                        (589, 1),   # Hellraiser: Bloodline
                        (901, 5),   # Big Lebowski
                        (1016, 4),  # Con Air
                        (1126, 3)]  # Truman Show

# Preparing the ratings of the new user.
ratings_new_user = np.zeros(1682)
is_rated_new_user = np.full(1682, False)
for rating in raw_ratings_new_user:
    ratings_new_user[rating[0]] = rating[1]
    is_rated_new_user[rating[0]] = True

# Loading the original matrices.
original_ratings_matrix = read_ratings_matrix()
original_is_rated_matrix = read_is_rated_matrix()

# Adding the new user to the original matrices.
modified_ratings_matrix = np.hstack((original_ratings_matrix, np.vstack(ratings_new_user)))
modified_is_rated_matrix = np.hstack((original_is_rated_matrix, np.vstack(is_rated_new_user)))

# Saving the modified matrices to files.
# Attempts to save these files directly to ../output/ have not been successful. They need to be moved manually.
np.savetxt("modified_ratings_matrix.csv", modified_ratings_matrix, delimiter=",", fmt="%d")
np.savetxt("modified_is_rated_matrix.csv", modified_is_rated_matrix, delimiter=",", fmt="%d")
