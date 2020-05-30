import numpy as np
from similarity import create_similarity_matrix

'''
    To validate the implementation of "pearson" in similarity.py a data-set was taken from (Gross, 2016, p.7). 
    The results of this program match the provided similarity matrices for both column-based and item-based on p.8-9. 
    It also shows that the computation of the pearson correlation coefficient is almost independent of 
    column-based and item-based. A simple adjustment of the data-set by using numpy.transpose() is sufficient.
'''

# Data-set for a quick evaluation of "pearson". Taken from (Gross, 2016, p.7).
item_based_ratings = np.array([[0.0, 1.0, 2.0, 1.0, 2.0],
                               [3.0, 4.0, 0.0, 1.0, 3.0],
                               [2.0, 2.0, 4.0, 0.0, 4.0],
                               [0.0, 5.0, 3.0, 1.0, 1.0],
                               [1.0, 0.0, 5.0, 4.0, 0.0],
                               [2.0, 4.0, 0.0, 4.0, 2.0]])

is_rated = np.array([[False, True, True, True, True],
                     [True, True, False, True, True],
                     [True, True, True, False, True],
                     [False, True, True, True, True],
                     [True, False, True, True, False],
                     [True, True, False, True, True]])

# Suppresses the scientific notation of floating-point numbers.
np.set_printoptions(suppress=True)

algorithm = "pearson"

# Computing item-based similarity-matrix and saving it.
print("\nITEM-BASED:\n")

similarity_matrix = create_similarity_matrix(item_based_ratings, is_rated, algorithm)
print(algorithm)
print(similarity_matrix)
print()

# Transposing item_based_ratings and is_rated to allow for column-based computations.
user_based_ratings = item_based_ratings.transpose()
has_rated = is_rated.transpose()

# Computing column-based similarity-matrix and saving it.
print("\nUSER-BASED:\n")

similarity_matrix = create_similarity_matrix(user_based_ratings, has_rated, algorithm)
print(algorithm)
print(similarity_matrix)
print()
