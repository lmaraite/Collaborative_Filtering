import sys, site, os

site.addsitedir(os.path.join(os.path.abspath(sys.path[0]), ".."))

import numpy as np
from input.filesystem import read_ratings_matrix, read_is_rated_matrix

# The ratings of the new users are saved in a lists of tuples.
raw_ratings_new_user1 = [(0, 5),    # Toy Story
                         (49, 5),   # Star Wars
                         (55, 2),   # Pulp Fiction
                         (61, 3),   # Stargate
                         (68, 4),   # Forrest Gump
                         (77, 4),   # Free Willy
                         (81, 2),   # Jurassic Park
                         (82, 2),   # Much Ado About Nothing
                         (93, 4),   # Home Alone
                         (95, 4),   # Terminator 2
                         (116, 5),  # The Rock
                         (143, 4),  # Die Hard
                         (153, 3),  # Life of Brian
                         (154, 5),  # Dirty Dancing
                         (160, 2),  # Top Gun
                         (171, 5),  # Empire Strikes Back
                         (173, 4),  # Raiders of the lost Ark
                         (180, 5),  # Return of the Jedi
                         (184, 4),  # Psycho
                         (194, 4),  # Terminator 1
                         (203, 5),  # Back to the Future
                         (209, 5),  # Indiana Jones 3
                         (225, 4),  # Die Hard 2
                         (248, 2),  # Austin Powers
                         (271, 2),  # Good Will Hunting
                         (312, 4),  # Titanic
                         (317, 2),  # Schindler's List
                         (392, 3),  # Mrs. Doubtfire
                         (398, 4),  # Three Musketeers
                         (404, 4)]  # Mission Impossible

raw_ratings_new_user2 = [(0, 5),  # Toy Story
                         (16, 1),  # From Dusk Till Dawn
                         (55, 1),  # Pulp Fiction
                         (70, 5),  # Lion King
                         (94, 5),  # Aladdin
                         (143, 1),  # Die Hard
                         (216, 1),  # Bram Stokers Dracula
                         (218, 1),  # Nightmare on Elm Street
                         (225, 1),  # Die Hard 2
                         (230, 3),  # Batman Returns
                         (312, 2),  # Titanic
                         (402, 3),  # Batman
                         (464, 5),  # Jungle Book
                         (589, 1),  # Hellraiser: Bloodline
                         (901, 1),  # Big Lebowski
                         (1015, 1),  # Con Air
                         (1126, 2)]  # Truman Show

raw_ratings_new_user3 = [(0, 1),  # Toy Story
                         (16, 5),  # From Dusk Till Dawn
                         (55, 5),  # Pulp Fiction
                         (70, 1),  # Lion King
                         (94, 1),  # Aladdin
                         (143, 5),  # Die Hard
                         (216, 3),  # Bram Stokers Dracula
                         (218, 3),  # Nightmare on Elm Street
                         (225, 5),  # Die Hard 2
                         (230, 4),  # Batman Returns
                         (312, 1),  # Titanic
                         (402, 4),  # Batman
                         (464, 1),  # Jungle Book
                         (589, 2),  # Hellraiser: Bloodline
                         (901, 4),  # Big Lebowski
                         (1015, 5),  # Con Air
                         (1126, 1)]  # Truman Show

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
