# EE412 Movie Recommendation Challenge
import sys
import numpy as np
import math
import time
import os

start = time.time()
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
    matrix.fill(np.inf)
    for line in lines:
        matrix[line[0] - 2 , movies.index(line[1])] = line[2]
    
    norm_matrix = np.copy(matrix)

    diff = np.zeros([int(norm_matrix.shape[0]), int(norm_matrix.shape[1])])


    for i in range(len(norm_matrix)):
        mean = np.mean(norm_matrix[i][norm_matrix[i].nonzero()])
        norm_matrix[i][norm_matrix[i].nonzero()] -= mean
        diff[i, :] += mean
    
    for j in range(norm_matrix.shape[1]):
        mean = np.mean(norm_matrix[:, j][norm_matrix[:, j].nonzero()])
        norm_matrix[:, j][norm_matrix[:, j].nonzero()] -= mean
        diff[:, j] += mean
    
    return matrix, norm_matrix, diff

def initialize(u, m, d):
    # U = np.random.normal(0, 0.5, [u, d])
    # V = np.random.normal(0, 0.5, [d, m])
    U = np.zeros([u, d])
    V = np.zeros([d, m])
    return U, V


def rmse(UV, norm_matrix):
    Sum = 0
    count = 0
    for i in range(UV.shape[0]):
        for j in range(UV.shape[1]):
            if(norm_matrix[i,j] != 0):
                Sum += (norm_matrix[i,j]-UV[i,j])**2
                count += 1
    return math.sqrt(Sum/count)


def update(inU, m, U, V, r, s):
    if inU:
        x = 0
        for j in range(m.shape[1]):
            if m[r, j] != np.inf:
                temp = m[r, j]
                for k in range(U.shape[1]):
                    if k != s:
                        temp -= U[r, k] * V[k, j]
                x += V[s, j] * temp
        y = 0
        for j in range(V.shape[1]):
            if m[r, j] != np.inf:
                y += (V[s, j])**2
        return x/y
    else:
        x = 0
        for i in range(m.shape[0]):
            if m[i, s] != np.inf:
                temp = m[i, s]
                for k in range(U.shape[1]):
                    if k != r:
                        temp-=U[i, k]*V[k, s]
                x += U[i, r]*temp
        y = 0
        for i in range(U.shape[0]):
            if m[i, s] != np.inf:
                y += (U[i, r])**2
        return x/y

def train(norm_matrix, U, V):
    rmse_collection = []
    for i in range(30):
        for r in range(U.shape[0]):
            for s in range(U.shape[1]):
                U[r,s] = update(True, norm_matrix, U, V, r, s)
        for r in range(U.shape[1]):
            for s in range(V.shape[0]):
                V[r,s] = update(False, norm_matrix, U, V, r, s)
        RMSE = rmse(np.matmul(U, V), norm_matrix)
        # print(RMSE)
        rmse_collection.append((RMSE, np.matmul(U, V)))
    UV = sorted(rmse_collection, key=lambda x: x[0])[0][1]
    return UV

def find_avrg(matrix):
    return np.mean(np.array([point for point in matrix.flatten() if point != np.inf]))

def predict(UV, movieList, filePath):
    testFile = open(filePath, 'r')
    dir = os.getcwd()
    folder_path = os.path.join(dir, "output.txt")
    saveFile = open(folder_path, 'w')
    avg = find_avrg(UV)

    for lines in testFile:
        l = lines.split(",")
        user = int(l[0])
        movie = int(l[1])
        timeStamp = l[3]
        if(movie in movieList):
            uIdx = user - 2
            mIdx = movieList.index(movie)
            pred = UV[uIdx, mIdx]
            writeStr = str(user) + "," + str(movie) + "," + str(pred) + timeStamp + "\n"
            saveFile.write(writeStr)
        else:
            writeStr = str(user) + "," + str(movie) + "," + str(avg) + timeStamp + "\n"
            saveFile.write(writeStr)
    testFile.close()
    saveFile.close()


file_path = sys.argv[1]
test_path = sys.argv[2]
lines, users, movies = get_datapoints(file_path)
matrix, norm_matrix, diff = get_util_matrix(lines, users, movies)
U, V = initialize(int(norm_matrix.shape[0]), int(norm_matrix.shape[1]), 5)
UV = train(norm_matrix, U, V)
UV += diff
predict(UV, movies, test_path)
end = time.time()

