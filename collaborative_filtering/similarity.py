import numpy as np
import warnings
from numpy.linalg import norm

# TODO: Implement option for user-based.
# TODO: Correct all issues with adjusted cosine und Pearson
# TODO: Fix comments (check all comments).
# TODO: Clean up and improve the implementation of matrix_mode in adjusted_cosine.
# TODO: Find a faster solution for numpy iteration with two-dimensional indexing.
# TODO: Implement further functions to tidy up the code.
# TODO: Use pydoc for documentation.

'''
    Creates a similarity-matrix by iterating over all_ratings and saving every similarity at both
    destinations at the same time to prevent a second computation of the same similarity.
    To run "adjusted_cosine" efficiently the order of operations has to be changed. 
    It is a hacky solution but it will stay like this for the time being.
    Returns the whole matrix.
'''
def create_similarity_matrix(all_ratings, is_rated, mode):
    if mode == "adjusted_cosine":
        all_ratings = get_adjusted_cosine_ratings(all_ratings, is_rated)
    side_length = all_ratings.shape[0]
    similarity_matrix = np.full((side_length, side_length), np.nan)
    for element1 in range(side_length):
        for element2 in range(element1+1, side_length):
            similarity_matrix[element1, element2] = get_similarity((element1, element2), all_ratings, is_rated, mode)
            similarity_matrix[element2, element1] = similarity_matrix[element1, element2]
    return similarity_matrix


'''
    To avoid the repeated computation of the same means all_ratings gets adjusted to "adjusted_cosine" in advance.
    Returns an adjusted version of all_ratings in which all user means have already been subtracted.
'''
def get_adjusted_cosine_ratings(all_ratings, is_rated):
    all_user_means = get_all_user_means(all_ratings, is_rated)
    return all_ratings - all_user_means.transpose()


'''
    Returns an array with all user means.
'''
def get_all_user_means(all_ratings, is_rated):
    number_of_users = all_ratings.shape[1]
    all_user_means = np.empty(number_of_users)
    for user in range(number_of_users):
        actual_user_ratings = get_actual_user_ratings(user, all_ratings, is_rated)
        all_user_means[user] = np.mean(actual_user_ratings)
    return all_user_means


'''
    Filters all 'ratings' which are actually redundant.
    Returns an array with only the user's real ratings.
'''
def get_actual_user_ratings(user, all_ratings, is_rated):
    user_ratings = all_ratings[:, user]
    return user_ratings[is_rated[:, user]]


'''
    Manages different steps to gather the co_ratings and preparing them for a final computation of the
    similarity. If the mode is "pearson" the co_ratings need further adjustment. "cosine" needs no adjustment
    at all and in case of "adjusted_cosine" the ratings will already be adjusted beforehand.
    The adjustment and/or computation will only take place if there are enough co_ratings to prevent computation errors.
    Returns the computed similarity or nan if there are too few co_ratings.
'''
def get_similarity(element_ids, all_ratings, is_rated, mode):
    co_ratings = get_co_ratings(element_ids, all_ratings, is_rated)
    if less_than_x_co_ratings(co_ratings, 2):
        return np.nan
    elif mode == "pearson":
        pearson_adjusted_co_ratings = get_pearson_adjusted_co_ratings(co_ratings)
        return compute_similarity(pearson_adjusted_co_ratings)
    else:
        return compute_similarity(co_ratings)


'''
    Collects all co_ratings by masking all_ratings with a boolean is_co_rated array.
    If two users rated the same item the co_rating will be added.
    Returns the compiled array of co_ratings.
'''
# TODO: Find a way to index both rows at the same time to avoid vstack.
def get_co_ratings(element_ids, all_ratings, is_rated):
    is_co_rated = get_is_co_rated(element_ids, is_rated)
    ratings_element1 = all_ratings[element_ids[0]]
    ratings_element2 = all_ratings[element_ids[1]]
    co_ratings = np.vstack((ratings_element1[is_co_rated], ratings_element2[is_co_rated]))

    # Transposing to get vertical alignment for further computations.
    return co_ratings.T


'''
    Uses a logical AND to produce a boolean array that can be used as a mask on the original ratings matrix.
    Returns boolean information about whether the given pair of elements share ratings by individual users.
'''
def get_is_co_rated(element_ids, is_rated):
    return np.logical_and(is_rated[element_ids[0]], is_rated[element_ids[1]])


'''
    Checks if there are too few co_ratings which would trigger an unwanted behaviour in compute_similarity().
    Returns TRUE if there are indeed less than x co_ratings.
'''
def less_than_x_co_ratings(co_ratings, x):
    return co_ratings.shape[0] < x


'''
    Adjusts the co_ratings to pearson by subtracting the means of the element-specific co_ratings.
    Returns the adjusted co_ratings.
'''
# TODO: This function seems to have some issues. Needs more work.
def get_pearson_adjusted_co_ratings(co_ratings):
    pearson_adjusted_co_ratings = np.empty(co_ratings.shape)
    for row, index in zip(co_ratings, range(co_ratings.shape[0])):
        pearson_adjusted_co_ratings[index] = row - np.mean(row)
    return pearson_adjusted_co_ratings


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
