import numpy as np
import numpy.random
import math

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
    yield (permutated_indices[:max_number_elements], permutated_indices[max_number_elements:])

def select_indices_with_cross_validation(shape, is_rated, train_size=0.9):
    indices = np.array(
        [(x, y) for x in range(shape[0])
                    for y in range(shape[1])]
    )
    random_generator = numpy.random.default_rng()

    indices = indices[is_rated.flat]
    permutated_indices = random_generator.permutation(indices)

    segment_size = round(1 - train_size, 8)
    segment_len = indices.shape[0] * segment_size
    segment_num = math.floor(indices.shape[0] / segment_len)
    segment_len = math.floor(segment_len)

    remnant_len = indices.shape[0] - segment_num * segment_len
    if remnant_len != 0:
        remnant = permutated_indices[-remnant_len:]
        permutated_indices = permutated_indices[:-remnant_len]
    else:
        remnant = np.empty((0,0))

    segments = np.vsplit(permutated_indices, segment_num)
    for i in range(remnant_len):
        segments[i % segment_num] = np.vstack((segments[i % segment_num], [remnant[i]]))

    for i in range(segment_num):
        yield _concat_train_set(segments, test_index=i), _concat_test_set(segments, test_index=i)

def _concat_train_set(segments, test_index):
    train_set = np.empty((0, 0))
    for i in range(len(segments)):
        if i == test_index:
            continue
        for index in segments[i]:
            yield index

def _concat_test_set(segments, test_index):
    segment = segments[test_index]

    for index in segment:
        yield index

def keep_elements_by_index(matrix: np.ndarray, indices: np.array, baseValue: object) -> np.ndarray:
    kept_matrix = np.full(matrix.shape, baseValue)
    for index_x, index_y in indices:
        kept_matrix[index_x, index_y] = matrix[index_x, index_y]
    return kept_matrix

_names = {
    select_indices_with_hold_out: "Hold Out"
}
