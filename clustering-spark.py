import sys
import numpy as np
from pyspark import SparkConf, SparkContext

def get_datapoints(filePath):
    all_points = [list(map(float, point.split())) 
    for point in open(filePath, "r").readlines()]
    all_points = np.array(all_points)

    return all_points

def initialization(all_points, k):
    centroids = np.array(np.ones((k, 58)) * np.inf)
    centroids[0,:] = all_points[0]
    i = 1
    while i < k:
        distances = []
        for j in range(len(all_points)):
            if all_points[j].tolist() not in centroids.tolist():
                diff = np.linalg.norm(all_points[j] - centroids, axis=1)
                min_ = np.min(diff)
                distances.append((min_, j))
        distances = sorted(distances, key=lambda x: x[0], reverse=True)
        farthest = distances[0]
        index = farthest[1]
        centroids[i,:] = all_points[index]
        i += 1

    return centroids

def closestCentroid(x, centroids):
    diff = np.linalg.norm(np.array(x) - centroids, axis=1)
    closestCentroid_ = np.argmin(diff)
    return int(closestCentroid_)
            
def findDiameter(cluster):
    cluster = np.array(cluster)
    candidates = []
    for i in range(len(cluster)):
        diff = np.linalg.norm(cluster - cluster[i], axis=1)
        candidates.append(np.max(diff))

    diameter = sorted(candidates, reverse=True)[0]
    return diameter


conf = SparkConf()
sc = SparkContext(conf=conf)
filePath = sys.argv[1]
k = int(sys.argv[2])

allPoints = get_datapoints(filePath)

centroids = initialization(allPoints, k)
centroidsBRC = sc.broadcast(centroids)

dataset = sc.parallelize(allPoints.tolist())
assigned = dataset.map(lambda point: (closestCentroid(point, np.array(centroidsBRC.value)), point))
groupedClusters = assigned.groupByKey()\
    .map(lambda x : (x[0], list(x[1])))
diameters = groupedClusters.map(lambda x: findDiameter(x[1])).collect()

averageDiameter = float(np.mean(np.array(list(diameters))))
print(averageDiameter)
