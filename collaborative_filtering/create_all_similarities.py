import numpy as np
import timeit
import similarity
from similarity import create_similarity_matrix, get_similarity
import similarity
from similarity import create_similarity_matrix
from input.filesystem import read_ratings_matrix, read_is_rated_matrix

# TODO: Implement additional functions to tidy up the code.
# TODO: Use pydoc for documentation.
# TODO: Implement tests to ensure the correctness of the computed similarities.

# Tracking the time for later evaluations.
global_start_time = timeit.default_timer()

# Loading R (item_ratings) and Y (is_rated).
item_based_ratings = read_ratings_matrix()
is_rated = read_is_rated_matrix()

global_elapsed_time = timeit.default_timer() - global_start_time
print("The program has taken " + str(global_elapsed_time) + " seconds to load the data.")

algorithms = [similarity.COSINE, similarity.ADJUSTED_COSINE, similarity.PEARSON]

# Computing item-based similarity-matrices and saving them.
for algorithm in algorithms:
    local_start_time = timeit.default_timer()

    similarity_matrix = create_similarity_matrix(item_based_ratings, is_rated, algorithm)
    np.savetxt("../output/item_based_" + algorithm + "_similarity_matrix.csv", similarity_matrix, delimiter=",", fmt="%s")

    local_elapsed_time = timeit.default_timer() - local_start_time
    print("The program has taken " + str(local_elapsed_time) + " seconds to create and save the item-based " + algorithm + "-matrix.")

# Transposing item_based_ratings and is_rated to allow for user-based computations.
user_based_ratings = item_based_ratings.T
has_rated = is_rated.T

# Computing user-based similarity-matrices and saving them.
for algorithm in algorithms:
    local_start_time = timeit.default_timer()

    similarity_matrix = create_similarity_matrix(user_based_ratings, has_rated, algorithm)
    np.savetxt("../output/user_based_" + algorithm + "_similarity_matrix.csv", similarity_matrix, delimiter=",", fmt="%s")

    local_elapsed_time = timeit.default_timer() - local_start_time
    print("The program has taken " + str(local_elapsed_time) + " seconds to create and save the user-based " + algorithm + "-matrix.")

global_elapsed_time = timeit.default_timer() - global_start_time
print("The program has taken " + str(global_elapsed_time) + " seconds to execute completely.")


# For testing purposes only. Will print single similarities between elements.
'''
element_ids = (0, 1)

print("The similarities between the elements " + str(element_ids) + " are:")
print("cosine:\t\t\t\t" + str(get_similarity(element_ids, item_based_ratings, is_rated, "cosine")))
print("adjusted cosine:\t" + str(get_similarity(element_ids, item_based_ratings, is_rated, "adjusted_cosine")))
print("pearson:\t\t\t" + str(get_similarity(element_ids, item_based_ratings, is_rated, "pearson")))
'''