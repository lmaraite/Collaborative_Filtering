import sys
import site
import os

cwd = sys.path[0]
cwd = os.path.abspath(cwd)
collaborative_filtering_dir = os.path.join(cwd, "..")
site.addsitedir(collaborative_filtering_dir)

import numpy as np
import threading

from evaluation import accuracy as ac
from evaluation import selection
from input import filesystem
from similarity import similarity

ratings_matrix = filesystem.read_ratings_matrix() # movie x user matrix
is_rated_matrix = filesystem.read_is_rated_matrix()

class EvaluationThread(threading.Thread):
    def __init__(self, evaluation_function, evaluation_props):
        super().__init__()
        self.result = None
        self.evaluation_function = evaluation_function
        self.evaluation_props = evaluation_props

    def run(self):
        self.result = self.evaluation_function(self.evaluation_props)

def print_result(evaluation: EvaluationThread):
    print(evaluation.evaluation_props)
    print("Result: " + str(evaluation.result))

#Pearson correlation and item-based
pearson_item_based_prop = ac.SinglePredictionAccuracyEvaluationPropertiesBuilder() \
    .with_ratings_matrix(ratings_matrix, 1) \
    .with_is_rated_matrix(is_rated_matrix, 1) \
    .with_similarity(similarity.PEARSON) \
    .with_approach(similarity.ITEM_BASED) \
    .with_selection_strategy(selection.select_indices_with_cross_validation) \
    .with_train_size(0.95) \
    .with_error_measurement(ac.root_mean_squared_error).build()

pearson_item_based = EvaluationThread(ac.run_accuracy_evaluation, pearson_item_based_prop)

#Raw cosine similarity and item-based
cosine_item_based_prop = ac.SinglePredictionAccuracyEvaluationPropertiesBuilder() \
    .with_ratings_matrix(ratings_matrix, 1) \
    .with_is_rated_matrix(is_rated_matrix, 1) \
    .with_similarity(similarity.COSINE) \

    .with_approach(similarity.ITEM_BASED) \
    .with_selection_strategy(selection.select_indices_with_cross_validation) \
    .with_train_size(0.95) \
    .with_error_measurement(ac.root_mean_squared_error).build()

cosine_item_based = EvaluationThread(ac.run_accuracy_evaluation, cosine_item_based_prop)

pearson_item_based.start()
cosine_item_based.start()
pearson_item_based.join()
cosine_item_based.join()

print_result(pearson_item_based)
print("=========")
print_result(cosine_item_based)
