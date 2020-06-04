import sys, site, os
site.addsitedir(os.path.join(os.path.abspath(sys.path[0]), ".."))

import numpy as np
import timeit
import similarity
from similarity import create_similarity_matrix, create_number_of_co_ratings_matrix
from input.filesystem import read_ratings_matrix, read_is_rated_matrix


# Tracking the time for later evaluations.
global_start_time = timeit.default_timer()

# Loading R (item_ratings) and Y (is_rated).
item_based_ratings = read_ratings_matrix()
is_rated = read_is_rated_matrix()

global_elapsed_time = timeit.default_timer() - global_start_time
print("The program has taken " + str(global_elapsed_time) + " seconds to load the data.")

algorithms = [similarity.COSINE, similarity.ADJUSTED_COSINE, similarity.PEARSON]
approaches = [similarity.ITEM_BASED, similarity.USER_BASED]

for approach in approaches:
    local_start_time = timeit.default_timer()

    number_of_co_ratings_matrix = create_number_of_co_ratings_matrix(approach, item_based_ratings, is_rated)
    np.savetxt("../output/" + approach + "_" + "number_of_co_ratings_matrix.csv",
               number_of_co_ratings_matrix, delimiter=",", fmt="%s")

    local_elapsed_time = timeit.default_timer() - local_start_time
    print("The program has taken " + str(local_elapsed_time) + " seconds to create and save the "
          + approach + "_" + "number_of_co_ratings_matrix.")

    for algorithm in algorithms:
        local_start_time = timeit.default_timer()

        similarity_matrix = create_similarity_matrix(approach, algorithm, item_based_ratings, is_rated)
        np.savetxt("../output/" + approach + "_" + algorithm + "_similarity_matrix.csv",
                   similarity_matrix, delimiter=",", fmt="%s")

        local_elapsed_time = timeit.default_timer() - local_start_time
        print("The program has taken " + str(local_elapsed_time) + " seconds to create and save the "
              + approach + "_" + algorithm + "_similarity_matrix.")

global_elapsed_time = timeit.default_timer() - global_start_time
print("The program has taken " + str(global_elapsed_time) + " seconds to execute completely.")
