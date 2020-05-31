import itertools
import numpy as np


'''
    Tests all elements of a matrix for a certain value. 
    If activated the positions of the value will be printed on screen.
    Returns True if the value is found in any position.
'''
def test_matrix_for_value(matrix, value, display_output_activated=None):
    if display_output_activated is None:
        display_output_activated = False

    value_found = False
    for element1, element2 in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        if (matrix[element1, element2] == value) or (np.isnan(value) and np.isnan(matrix[element1, element2])):
            value_found = True
            if display_output_activated:
                print(str(value) + " at : ( " + str(element1) + " / " + str(element2) + " )")
    return value_found
