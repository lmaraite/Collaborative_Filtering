import numpy as np
import warnings
from numpy.linalg import norm

# TODO: It's possible to save computation time for adjusted_cosine by computing and saving all element-means globally.
# TODO: Find a faster solution for numpy iteration with two-dimensional indexing.
# TODO: Implement further functions to tidy up the code.
# TODO: Use pydoc for documentation.


'''
    Creates a similarity-matrix by iterating over all_ratings and saving every similarity at both
    destinations at the same time to prevent a second computation of the same similarity.
    The similarity of an element with itself is set to nan.
    Returns the whole matrix.
'''
def create_similarity_matrix(all_ratings, is_rated, mode):
    side_length = all_ratings.shape[0]
    similarity_matrix = np.full((side_length, side_length), np.nan)
    for element1 in range(side_length):
        for element2 in range(element1+1, side_length):
            similarity_matrix[element1, element2] = get_similarity((element1, element2), all_ratings, is_rated, mode)
            similarity_matrix[element2, element1] = similarity_matrix[element1, element2]
    return similarity_matrix


'''
    Manages different steps to gather the co_ratings and preparing them for a final computation of the
    similarity. The adjustment according to the chosen mode will only take place if there are enough
    co_ratings to prevent computation errors.
    Returns the computed similarity or nan if there are too few co_ratings.
'''
def get_similarity(element_ids, all_ratings, is_rated, mode):
    co_ratings = get_co_ratings(element_ids, all_ratings, is_rated)
    if less_than_x_co_ratings(co_ratings, 2):
        return np.nan
    else:
        adjusted_co_ratings = get_adjusted_co_ratings(element_ids, all_ratings, is_rated, co_ratings, mode)
        return compute_similarity(adjusted_co_ratings)


'''
    Collects all co_ratings by masking all_ratings with a boolean is_co_rated array.
    If two users rated the same item the co_rating will be added.
    Returns the compiled array of co_ratings.
'''
# TODO: Find a way to index both rows at the same time to avoid vstack.
def get_co_ratings(element_ids, all_ratings, is_rated):
    is_co_rated = get_is_co_rated(element_ids, is_rated)
    ratings_element1 = all_ratings[element_ids[0], :]
    ratings_element2 = all_ratings[element_ids[1], :]
    co_ratings = np.vstack((ratings_element1[is_co_rated], ratings_element2[is_co_rated]))

    # Transposing to get vertical alignment for further computations.
    return co_ratings.T


'''
    Uses a logical AND to produce a boolean array that can be used as a mask on the original ratings matrix.
    Returns boolean information about whether the given pair of elements share ratings by individual users.
'''
def get_is_co_rated(element_ids, is_rated):
    return np.logical_and(is_rated[element_ids[0], :], is_rated[element_ids[1], :])


'''
    Checks if there are too few co_ratings which would trigger an unwanted behaviour in compute_similarity().
    Returns TRUE if there are indeed less than x co_ratings.
'''
def less_than_x_co_ratings(co_ratings, x):
    return co_ratings.shape[0] < x


'''
    Evaluates mode to choose the correct adjustment of the co_ratings. No adjustments are needed in case of
    cosine. Defaults to cosine if the mode is unknown. 
    Returns the algorithm-adjusted co_ratings.
'''
def get_adjusted_co_ratings(element_ids, all_ratings, is_rated, co_ratings, mode):
    if mode == "adjusted_cosine":
        return convert_to_adjusted_cosine(element_ids, all_ratings, is_rated, co_ratings)
    elif mode == "pearson":
        return convert_to_pearson(co_ratings)
    elif mode == "cosine":
        return co_ratings
    else:
        print("Unknown mode selected. The algorithm defaults to cosine similarity.")
        return co_ratings


'''
    Adjusts the co_ratings to pearson by subtracting the means of the element-specific co_ratings.
    Returns the adjusted co_ratings.
'''
def convert_to_pearson(co_ratings):
    return co_ratings - np.array([np.mean(co_ratings[:, 0]), np.mean(co_ratings[:, 1])])


'''
    Adjusts the co_ratings to adjusted_cosine by subtracting the means of all ratings 
    given to the specific elements. Extra steps are needed to filter only the element-specific 
    ratings from all_ratings considering that some may be redundant.
    Returns the adjusted co_ratings.
'''
def convert_to_adjusted_cosine(element_ids, all_ratings, is_rated, co_ratings):
    actual_ratings_element1 = get_actual_element_ratings(element_ids[0], all_ratings, is_rated)
    actual_ratings_element2 = get_actual_element_ratings(element_ids[1], all_ratings, is_rated)

    # Subtracts the means of all element-specific ratings column-wise.
    return co_ratings - np.array([np.mean(actual_ratings_element1), np.mean(actual_ratings_element2)])


'''
    Filters all 'ratings' which are actually redundant.
    Returns an array with only the element's real ratings.
'''
def get_actual_element_ratings(element_id, all_ratings, is_rated):
    element_ratings = all_ratings[element_id, :]
    return element_ratings[is_rated[element_id, :]]


'''
    The final step to compute the similarity between two elements according to (Jannach et. al, 2011, p.19). 
    Would produce a RuntimeWarning when a division by zero is attempted. This is ignored because in these cases
    nan will be returned which is expected behavior.
    Returns the resulting similarity.
'''
def compute_similarity(ratings):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        similarity = np.dot(ratings[:, 0], ratings[:, 1]) / (norm(ratings[:, 0]) * norm(ratings[:, 1]))
    return similarity