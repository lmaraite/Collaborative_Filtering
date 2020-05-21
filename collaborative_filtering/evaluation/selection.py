import numpy as np

import numpy.random

def select_indices_with_hold_out(shape, is_rated, train_size=0.8) -> np.array:
    indices = np.array(
        [(x, y) for x in range(shape[0])
                    for y in range(shape[1])]
    )
    indices = indices[is_rated.flat]
    random_generator = numpy.random.default_rng()

    size = indices.shape[0]
    max_number_elements = round(size * train_size)
    permutated_indices = random_generator.permutation(indices)
    return (permutated_indices[:max_number_elements], permutated_indices[max_number_elements:])

def keep_elements_by_index(matrix: np.ndarray, indices: np.array, baseValue: object) -> np.ndarray:
    kept_matrix = np.full(matrix.shape, baseValue)
    for index_x, index_y in indices:
        kept_matrix[index_x, index_y] = matrix[index_x, index_y]
    return kept_matrix
