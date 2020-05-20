import numpy as np

import numpy.random

def select_indices_with_hold_out(shape, train_size=0.8) -> np.array:
    indices = np.array(
        [(x, y) for x in range(shape[0])
                    for y in range(shape[1])]
    )
    size = shape[0] * shape[1]

    random_generator = numpy.random.default_rng()

    max_number_elements = round(size * train_size)
    return random_generator.permutation(indices)[:max_number_elements]

def filter_by_index(matrix: np.ndarray, indices: np.array) -> np.ndarray:
    pass
