import numpy as np
import os

from input import DATASET_DIR

def read_ratings_matrix(path=os.path.join(DATASET_DIR, "R.csv"), delimiter=","):
    return np.genfromtxt(path, delimiter=delimiter, dtype=np.float)

def read_is_rated_matrix(path=os.path.join(DATASET_DIR, "Y.csv"), delimiter=","):
    return np.genfromtxt(path, delimiter=delimiter, dtype=np.uint8).astype(bool)

def read_similarity_matrix(path=None, name=None, delimiter=","):
    if not (path is None or name is None):
        raise ValueError("Path and name cannot be both set.")
    if path is None and name is None:
        raise ValueError("Either path or name must be set.")
    if path is None:
        path = os.path.join(DATASET_DIR, name)

    return np.genfromtxt(path, delimiter=delimiter, dtype=np.float64)
