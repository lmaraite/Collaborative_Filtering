Information about the movie dataset:



ex8_movies.mat: contains the variables Y and R
R: a (num_movies x num_users)-matrix of size 1682 x 943 containing the ratings (from 1 to 5) given by 943 users for 1682 movies (of course not all the movies are rated)
Y: a (num_movies x num_users)-matrix of size 1682 x 943. This matrix has only values of 0 or 1. 
	Each cell y_{ij} is 1 if the user i has rated the movie j. The cell y_{ij} is 0 if the user i did not rate the movie j.
movie_ids.text: contains the movie ids and the name of the movies.
		This list contains 1682 movies produced from 1930 till 1998 (pretty old but I might say there are many good movies among them)

R.csv: the R matrix only in csv format in case you have trouble working with mat files
Y.csv: the Y matrix only in csv format in case you have trouble working with mat files