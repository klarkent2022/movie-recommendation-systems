import sys
import numpy as np

def get_datapoints(file_path):
    lines = open(file_path, 'r').readlines()
    lines = [line.split()[0] for line in lines]
    lines = [line.split(',') for line in lines]
    for line in lines:
        line[0] = int(line[0])
        line[1] = int(line[1])
        line[2] = float(line[2])
        line[3] = int(line[3])
    
    users = []
    for line in lines:
        if line[0] not in users:
            users.append(line[0])
    users = sorted(users)
    # print( "# of users: " + str(len(users)))

    movies = []
    for line in lines:
        if line[1] not in movies:
            movies.append(line[1])
    movies = sorted(movies)
    # print("# of movies: " + str(len(movies)))
    
    return lines, users, movies

def get_util_matrix(lines, users, movies):
    matrix = np.zeros([len(users), len(movies)])
    for line in lines:
        matrix[line[0] - 2 , movies.index(line[1])] = line[2]
    
    norm_matrix = np.copy(matrix)

    for i in range(len(norm_matrix)):
        mean = np.mean(norm_matrix[i][norm_matrix[i].nonzero()])
        norm_matrix[i][norm_matrix[i].nonzero()] -= mean

    return matrix, norm_matrix

def similarUsers(norm_matrix, userID):
    similarities = []
    for i in range(len(norm_matrix)):
        if i != userID - 2:
            cos = (np.inner(norm_matrix[i], norm_matrix[userID - 2])) / (np.linalg.norm(norm_matrix[i]) * np.linalg.norm(norm_matrix[userID-2]) )
            similarities.append((i, cos))
        else:
            similarities.append((i, 0.0))
    
    top10 = sorted(similarities, key=lambda x: x[1], reverse=True)[0:10]
    top10 = [x[0] for x in top10]
    # print(top10)
    return top10

def recommendUserBased(matrix, userID, sim_users, movies):
    predicted_ratings = []
    i = 0
    while movies[i] <= 1000:
        sum_ = 0.0
        nonzero_count = 0
        avg = 0.0
        for j in sim_users:
            if matrix[j, i] != 0.0:
                sum_ += matrix[j, i]
                nonzero_count += 1
        if nonzero_count != 0:
            avg = sum_ / nonzero_count
        predicted_ratings.append((movies[i], avg))
        i += 1

    predicted_ratings = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[0:5]
    # print(predicted_ratings)
    return predicted_ratings

def similarMovies(norm_matrix, movies, movieID):
    similarities = []
    for j in range(norm_matrix.shape[1]):
        if ((j != movies.index(movieID)) and movies[j] > 1000):
            columnj = norm_matrix[:, j]
            columnMovieID = norm_matrix[:, movies.index(movieID)]
            innerProduct = np.inner(columnj, columnMovieID)
            divisor = np.linalg.norm(columnj) * np.linalg.norm(columnMovieID)
            if (divisor != 0):
                cos = float(innerProduct) / float(divisor)
                similarities.append((j, cos))
        else:
            similarities.append((j, 0.0))
    
    top10 = sorted(similarities, key=lambda x: x[1], reverse=True)[0:10]
    top10 = [x[0] for x in top10]
    # print(top10)
    return top10

def recommendItemBased(matrix, norm_matrix, userID, movies):
    predicted_ratings = []
    i = 0
    while movies[i] <= 1000:
        sum_ = 0.0
        nonzero_count = 0
        avg = 0.0
        sim_movies = similarMovies(norm_matrix, movies, movies[i])
        for j in sim_movies:
            if matrix[userID - 2, j] != 0.0:
                sum_ += matrix[userID - 2, j]
                nonzero_count += 1
        if nonzero_count != 0:
            avg = sum_ / nonzero_count
        predicted_ratings.append((movies[i], avg))
        i += 1
    
    predicted_ratings = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[0:5]
    return predicted_ratings

file_path = sys.argv[1]
lines, users, movies = get_datapoints(file_path)
matrix, norm_matrix = get_util_matrix(lines, users, movies)

sim_users = similarUsers(norm_matrix, 600)
recommendationsUserBased = recommendUserBased(matrix, 600, sim_users, movies)
for (movieID, rating) in recommendationsUserBased:
    print(str(movieID) + "\t" + str(rating))

recommendationsItemBased = recommendItemBased(matrix, norm_matrix, 600, movies)
for (movieID, rating) in recommendationsItemBased:
    print(str(movieID) + "\t" + str(rating))

#####################################################################
