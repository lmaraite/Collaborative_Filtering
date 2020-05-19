import numpy as np
from numpy.linalg import norm
import itertools
import warnings

# TODO: Implement further functions to tidy up the code.
# TODO: Search for the problems causing the RuntimeWarning when computing item-based adjusted cosine.
# TODO: Change comment style to javadoc to document all parameters etc.

'''
    Creates a similarity-matrix by iterating over all_ratings and saving every similarity at both
    destinations at the same time to prevent a second computation of the same similarity. 
    Returns the whole matrix.
'''
def create_similarity_matrix(all_ratings, is_rated, mode):
    side_length = all_ratings.shape[0]
    similarity_matrix = np.full((side_length, side_length), np.nan)
    for element1, element2 in itertools.product(range(side_length), range(side_length)):
        if np.isnan(similarity_matrix[element1, element2]):
            similarity_matrix[element1, element2] = get_similarity((element1, element2), all_ratings, is_rated, mode)
            similarity_matrix[element2, element1] = similarity_matrix[element1, element2]
    return similarity_matrix


'''
    Manages different steps to gather the co_ratings and preparing them for a final computation of the
    similarity. The adjustment according to the chosen mode will only take place if there are enough
    co_ratings to prevent computations errors.
    Returns the computed similarity or nan if there are less than two co_ratings.
'''
def get_similarity(elements_ids, all_ratings, is_rated, mode):
    co_ratings = get_co_ratings(elements_ids, all_ratings, is_rated)
    if less_than_two_co_ratings(co_ratings):
        return np.nan
    else:
        adjusted_co_ratings = get_adjusted_co_ratings(elements_ids, all_ratings, co_ratings, mode)
        return compute_similarity(adjusted_co_ratings)


'''
    Collects all co_ratings by iterating over two rating-arrays and two is_rated-arrays 
    at the same time. If two users rated the same item the co_rating will be added.
    Returns the compiled array of co_ratings.
'''
# TODO: It's possible to omit is_rated by checking all_ratings for zeroes.
# TODO: Replace .append() with a faster solution.
def get_co_ratings(elements_ids, all_ratings, is_rated):
    co_ratings = np.empty(shape=[0, 2])
    for element1_rating, element2_rating, element1_is_rated, element2_is_rated \
            in zip(all_ratings[elements_ids[0], :], all_ratings[elements_ids[1], :], is_rated[elements_ids[0], :], is_rated[elements_ids[1], :]):
        if element1_is_rated and element2_is_rated:
            co_ratings = np.append(co_ratings, [[element1_rating, element2_rating]], axis=0)
    return co_ratings


'''
    Checks if there are less than two co_ratings which would trigger an unwanted behaviour in compute_similarity().
    Returns TRUE if there are indeed less than two ratings.
'''
def less_than_two_co_ratings(co_ratings):
    return co_ratings.shape[0] < 2


'''
    Evaluates mode to choose the correct adjustment of the co_ratings. No adjustments are needed in case of
    cosine. Defaults to cosine if the mode is unknown. 
    Returns the algorithm-adjusted co_ratings.
'''
def get_adjusted_co_ratings(elements_ids, all_ratings, co_ratings, mode):
    if (mode == "adjusted_cosine"):
        return convert_to_adjusted_cosine(elements_ids, all_ratings, co_ratings)
    elif (mode == "pearson"):
        return convert_to_pearson(co_ratings)
    elif (mode == "cosine"):
        return co_ratings
    else:
        print("Unknown mode selected. The algorithm defaults to cosine similarity.")
        return co_ratings


'''
    Adjusts the co_ratings to pearson by subtracting the means of the element-specific co_ratings.
    Returns the adjusted co_ratings.
'''
def convert_to_pearson(co_ratings):
    return co_ratings - get_column_specific_means(co_ratings)


'''
    Adjusts the co_ratings to adjusted_cosine by subtracting the means of all ratings given to the specific elements.
    Extra steps are needed to filter only the element-specific ratings from all_ratings.
    Returns the adjusted co_ratings.
'''
# TODO: Abstract an implementation of get_element_filtered_ratings from this function.
# TODO: It's possible to save a lot computation time by computing and saving all element-specific means in advance.
def convert_to_adjusted_cosine(elements_ids, all_ratings, co_ratings):
    all_ratings[all_ratings == 0] = np.nan
    element_filtered_ratings = np.vstack([all_ratings[elements_ids[0], :], all_ratings[elements_ids[1], :]])

    # Transposing in order to use get_column_specific_means().
    return co_ratings - get_column_specific_means(element_filtered_ratings.transpose())


'''
    Computes the means of all elements column-wise. All nan-values are ignored to prevent wrong means.
    Returns an array with two values: One mean for each column.
'''
def get_column_specific_means(ratings):
    return np.array([np.nanmean(ratings[:, 0]), np.nanmean(ratings[:, 1])])


'''
    The final step to compute the similarity between two elements according to (Jannach et. al, 2011, p.19). 
    Small internal rounding errors unveil if an element is compared to itself which will not produce a
    similarity of 1.0 . An additional rounding helps to correct these instances. 
    Will produce a RuntimeWarning if a division by zero is attempted.
    Returns the resulting similarity.
'''
def compute_similarity(ratings):
    similarity = np.dot(ratings[:, 0], ratings[:, 1]) / (norm(ratings[:, 0]) * norm(ratings[:, 1]))

    # Rounding to accommodate previous rounding errors due to floating-point arithmetic.
    return round(similarity, 14)

