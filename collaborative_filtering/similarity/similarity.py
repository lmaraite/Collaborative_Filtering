import numpy as np
import warnings
from numpy.linalg import norm

# TODO: Implement function to check for invalid input (approach, algorithm).

PEARSON = "pearson"
ADJUSTED_COSINE = "adjusted_cosine"
COSINE = "cosine"
ITEM_BASED = "item_based"
USER_BASED = "user_based"

"""
    In order to save computation time during prediction the numbers of the individual co-ratings get saved in an
    additional matrix. In case of USER_BASED the given matrices get transposed.
    Returns the whole matrix.
"""
def create_number_of_co_ratings_matrix(approach, all_ratings, is_rated):
    if approach == USER_BASED:
        all_ratings = all_ratings.T
        is_rated = is_rated.T

    side_length = all_ratings.shape[0]
    number_of_co_ratings_matrix = np.full((side_length, side_length), np.nan)
    for row1 in range(side_length):
        for row2 in range(row1+1, side_length):
            co_ratings = get_co_ratings((row1, row2), all_ratings, is_rated)
            number_of_co_ratings_matrix[row1, row2] = co_ratings.shape[0]
            number_of_co_ratings_matrix[row2, row1] = number_of_co_ratings_matrix[row1, row2]
    return number_of_co_ratings_matrix


'''
    Creates a similarity-matrix by iterating over all_ratings and saving every similarity at both
    destinations at the same time to prevent a second computation of the same similarity.
    To run "adjusted_cosine" efficiently the order of operations is changed.
    In case of USER_BASED the matrices have to be transposed.
    It is not a clean solution but it will stay like this for the time being.
    Returns the whole matrix.
'''
def create_similarity_matrix(approach, algorithm, all_ratings, is_rated):
    if algorithm == ADJUSTED_COSINE:
        all_ratings = get_adjusted_cosine_ratings(all_ratings, is_rated)
    if approach == USER_BASED:
        all_ratings = all_ratings.T
        is_rated = is_rated.T

    side_length = all_ratings.shape[0]
    similarity_matrix = np.full((side_length, side_length), np.nan)
    for row1 in range(side_length):
        for row2 in range(row1+1, side_length):
            similarity_matrix[row1, row2] = get_similarity((row1, row2), all_ratings, is_rated, algorithm)
            similarity_matrix[row2, row1] = similarity_matrix[row1, row2]
    return similarity_matrix


'''
    To avoid the repeated computation of the same means all_ratings gets adjusted to "adjusted_cosine" in advance.
    Returns an adjusted version of all_ratings in which all column means have already been subtracted column-wise.
'''
def get_adjusted_cosine_ratings(all_ratings, is_rated):
    all_column_means = get_all_column_means(all_ratings, is_rated)
    return all_ratings - all_column_means


'''
    Returns an array with all column means.
'''
def get_all_column_means(all_ratings, is_rated):
    number_of_columns = all_ratings.shape[1]
    all_columns_means = np.empty(number_of_columns)
    for column in range(number_of_columns):
        actual_column_ratings = get_actual_column_ratings(column, all_ratings, is_rated)
        all_columns_means[column] = np.mean(actual_column_ratings)
    return all_columns_means


'''
    Filters all 'ratings' which are actually redundant.
    Returns an array with only the column's real ratings.
'''
def get_actual_column_ratings(column, all_ratings, is_rated):
    column_ratings = all_ratings[:, column]
    return column_ratings[is_rated[:, column]]


'''
    Manages different steps to gather the co_ratings and preparing them for a final computation of the
    similarity. If the algorithm is "pearson" the co_ratings need further adjustment. "cosine" needs no adjustment
    at all and in case of "adjusted_cosine" the ratings will already be adjusted beforehand.
    The adjustment and/or computation will only take place if there are enough co_ratings to prevent computation errors.
    Returns the computed similarity or nan if there are too few co_ratings.
'''
def get_similarity(rows, all_ratings, is_rated, mode):
    co_ratings = get_co_ratings(rows, all_ratings, is_rated)
    if less_than_x_co_ratings(co_ratings, 2):
        return np.nan
    elif mode == PEARSON:
        pearson_adjusted_co_ratings = get_pearson_adjusted_co_ratings(co_ratings)
        return compute_similarity(pearson_adjusted_co_ratings)
    else:
        return compute_similarity(co_ratings)


'''
    Collects all co_ratings by masking all_ratings with a boolean is_co_rated array.
    Returns the compiled array of co_ratings in the format (number_of_co_ratings * 2).
'''
def get_co_ratings(rows, all_ratings, is_rated):
    is_co_rated = get_is_co_rated(rows, is_rated)
    ratings_row1 = all_ratings[rows[0]]
    ratings_row2 = all_ratings[rows[1]]
    co_ratings = np.vstack((ratings_row1[is_co_rated], ratings_row2[is_co_rated]))

    # Transposing to get vertical alignment for further computations.
    return co_ratings.T


'''
    Uses a logical AND to produce a boolean array that can be used as a mask on the original ratings matrix.
    Returns boolean information about whether the given pair of rows share ratings.
'''
def get_is_co_rated(rows, is_rated):
    return np.logical_and(is_rated[rows[0]], is_rated[rows[1]])


'''
    Checks if there are too few co_ratings which would trigger an unwanted behaviour in compute_similarity().
    Returns TRUE if there are indeed less than x co_ratings.
'''
def less_than_x_co_ratings(co_ratings, x):
    return co_ratings.shape[0] < x


'''
    Adjusts the co_ratings to pearson according to (Gross,  2016, p.7-9).
    Returns the adjusted co_ratings.
'''
def get_pearson_adjusted_co_ratings(co_ratings):
    return co_ratings - np.array([np.mean(co_ratings[:, 0]), np.mean(co_ratings[:, 1])])


'''
    The final step to compute the similarity between two elements according to (Jannach et al., 2011, p.19). 
    Would raise a RuntimeWarning when a division by zero is attempted. This is ignored because in these cases
    nan will be returned which is expected behavior.
    Returns the resulting similarity.
'''
def compute_similarity(ratings):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        similarity = np.dot(ratings[:, 0], ratings[:, 1]) / (norm(ratings[:, 0]) * norm(ratings[:, 1]))
    return similarity
