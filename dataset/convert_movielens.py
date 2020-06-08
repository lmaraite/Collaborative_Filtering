import getopt
import sys
import numpy as np
import tempfile
import os

try:
    opts, args = getopt.getopt(sys.argv[1:], "i:o:d:")
except getopt.GetoptError:
    print("convert_movielens.py -i <inputfile> -o <name of output file without extension> [-d delimiter=',']")
    sys.exit(1)

delimiter = ","

for opt, arg in opts:
    if opt == "-i":
        movielens_file = arg
    elif opt == "-o":
        converted_file = arg
    elif opt == "d":
        delimiter = arg

tmp_movielens_file_fd, tmp_movielens_file_path = tempfile.mkstemp()

#remove timestamp
with open(movielens_file, "r") as input_file, open(tmp_movielens_file_fd, "w") as tmp_movielens_file:
    for line in input_file:
        reduced_line = line.rsplit(delimiter, 1)[0]
        tmp_movielens_file.write(reduced_line + "\n")

movielens_matrix = np.loadtxt(
    tmp_movielens_file_path,
    dtype=np.float32,
    delimiter=delimiter,
    skiprows=1
)
print(movielens_matrix)
max_movie = np.count_nonzero(np.unique(movielens_matrix.T[1]))
max_user = np.int(np.amax(movielens_matrix.T[0]))
ratings_matrix = np.empty((max_movie, max_user), dtype=np.float32)

key_map = {}
lowest = 0

def get_id_increasing(movie_id):
    try:
        return key_map[movie_id]
    except KeyError:
        global lowest
        key_map[movie_id] = lowest
        lowest += 1
        return key_map[movie_id]

#convert to itemxuser matrix
for row in movielens_matrix:
    user_id = row[0] -1
    movie_id = get_id_increasing(row[1])
    ratings_matrix[np.int(movie_id), np.int(user_id)] = row[2]

is_rated_matrix = ratings_matrix != 0

np.savetxt(converted_file + ".csv", ratings_matrix, delimiter=delimiter, fmt="%1.1f")
np.savetxt(converted_file + "_is_rated.csv", is_rated_matrix, delimiter=delimiter, fmt="%d")

os.remove(tmp_movielens_file_path)
