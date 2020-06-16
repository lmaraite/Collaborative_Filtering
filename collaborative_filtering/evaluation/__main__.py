import sys
import site
import os

cwd = sys.path[0]
cwd = os.path.abspath(cwd)
collaborative_filtering_dir = os.path.join(cwd, "..")
site.addsitedir(collaborative_filtering_dir)

import numpy as np
import threading

import evaluation
from evaluation import accuracy as ac
from evaluation import selection
from input import filesystem
from similarity import similarity

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
    print("============")

def print_analysis(analysis):
    for name, result in analysis.items():
        print("{}: {}".format(name, result))

#dat set from ilias
data_set_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
ratings_matrix = filesystem.read_ratings_matrix(
    os.path.join(data_set_dir, "movielens_100k_converted.csv")
) # movie x user matrix
is_rated_matrix = filesystem.read_is_rated_matrix(
    os.path.join(data_set_dir, "movielens_100k_converted_is_rated.csv")
)

analysis = evaluation.analyse_data_set(ratings_matrix, is_rated_matrix, 1)
print_analysis(analysis)

props = []

for q in [5, 10, 20]:
    num_of_ratings = analysis["num_of_ratings"]
    props.append(ac.SinglePredictionAccuracyEvaluationPropertiesBuilder() \
        .with_ratings_matrix(ratings_matrix, 1) \
        .with_is_rated_matrix(is_rated_matrix, 1) \
        .with_similarity(similarity.COSINE) \
        .with_approach(similarity.USER_BASED) \
        .with_selection_strategy(selection.select_indices_with_cross_validation) \
        .with_train_size(1 - ((num_of_ratings / q) / num_of_ratings)) \
        .with_error_measurement(ac.mean_absolute_error).build())

    props.append(ac.SinglePredictionAccuracyEvaluationPropertiesBuilder() \
        .with_ratings_matrix(ratings_matrix, 1) \
        .with_is_rated_matrix(is_rated_matrix, 1) \
        .with_similarity(similarity.COSINE) \
        .with_approach(similarity.ITEM_BASED) \
        .with_selection_strategy(selection.select_indices_with_cross_validation) \
        .with_train_size(1 - ((num_of_ratings / q) / num_of_ratings)) \
        .with_error_measurement(ac.mean_absolute_error).build())

    props.append(ac.SinglePredictionAccuracyEvaluationPropertiesBuilder() \
        .with_ratings_matrix(ratings_matrix, 1) \
        .with_is_rated_matrix(is_rated_matrix, 1) \
        .with_similarity(similarity.ADJUSTED_COSINE) \
        .with_approach(similarity.USER_BASED) \
        .with_selection_strategy(selection.select_indices_with_cross_validation) \
        .with_train_size(1 - ((num_of_ratings / q) / num_of_ratings)) \
        .with_error_measurement(ac.mean_absolute_error).build())

    props.append(ac.SinglePredictionAccuracyEvaluationPropertiesBuilder() \
        .with_ratings_matrix(ratings_matrix, 1) \
        .with_is_rated_matrix(is_rated_matrix, 1) \
        .with_similarity(similarity.ADJUSTED_COSINE) \
        .with_approach(similarity.ITEM_BASED) \
        .with_selection_strategy(selection.select_indices_with_cross_validation) \
        .with_train_size(1 - ((num_of_ratings / q) / num_of_ratings)) \
        .with_error_measurement(ac.mean_absolute_error).build())

    props.append(ac.SinglePredictionAccuracyEvaluationPropertiesBuilder() \
        .with_ratings_matrix(ratings_matrix, 1) \
        .with_is_rated_matrix(is_rated_matrix, 1) \
        .with_similarity(similarity.PEARSON) \
        .with_approach(similarity.USER_BASED) \
        .with_selection_strategy(selection.select_indices_with_cross_validation) \
        .with_train_size(1 - ((num_of_ratings / q) / num_of_ratings)) \
        .with_error_measurement(ac.mean_absolute_error).build())

    props.append(ac.SinglePredictionAccuracyEvaluationPropertiesBuilder() \
        .with_ratings_matrix(ratings_matrix, 1) \
        .with_is_rated_matrix(is_rated_matrix, 1) \
        .with_similarity(similarity.PEARSON) \
        .with_approach(similarity.ITEM_BASED) \
        .with_selection_strategy(selection.select_indices_with_cross_validation) \
        .with_train_size(1 - ((num_of_ratings / q) / num_of_ratings)) \
        .with_error_measurement(ac.mean_absolute_error).build())


evaluations = []

for prop in props:
    evaluations.append(EvaluationThread(ac.run_accuracy_evaluation, prop))

for evaluation in evaluations:
    evaluation.start()

for evaluation in evaluations:
    evaluation.join()
    print_result(evaluation)

props = []
for q in [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    num_of_ratings = analysis["num_of_ratings"]
    props.append(ac.SinglePredictionAccuracyEvaluationPropertiesBuilder() \
        .with_ratings_matrix(ratings_matrix, 1) \
        .with_is_rated_matrix(is_rated_matrix, 1) \
        .with_similarity(similarity.COSINE) \
        .with_approach(similarity.USER_BASED) \
        .with_selection_strategy(selection.select_indices_with_cross_validation) \
        .with_train_size(1 - ((num_of_ratings / q) / num_of_ratings)) \
        .with_error_measurement(ac.mean_absolute_error).build())

evaluations = []
for prop in props:
    evaluations.append(EvaluationThread(ac.run_accuracy_evaluation, prop))

for evaluation in evaluations:
    evaluation.start()
    evaluation.join()
    print_result(evaluation)
