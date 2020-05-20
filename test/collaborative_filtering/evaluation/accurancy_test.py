import unittest
import numpy as np
import math

from collaborative_filtering.evaluation import accurancy as ac

class ErrorTest(unittest.TestCase):

    def test_error_of_prediction_for_4_and_3(self):
        prediction = 4
        rating = 3

        assert 1 == ac.error(prediction, rating)

    def test_error_of_predicition_for_1_and_2(self):
        prediction = 1
        rating = 2

        assert -1 == ac.error(prediction, rating)

    def test_error_of_prediction_for_correct_prediction(self):
        prediction = 3
        rating = 3

        assert 0 == ac.error(prediction, rating)

class RootMeanSquaredErrorTest(unittest.TestCase):

    def test_root_mean_squared_error_for_4_values(self):
        #given
        predictions = np.array([
            4, 5, 1
        ])
        ratings = np.array([
            4, 2, 3
        ])

        #when
        error = ac.root_mean_squared_error(predictions, ratings)
        expected_error = 2.081665999

        #then
        assert math.isclose(error, expected_error)
