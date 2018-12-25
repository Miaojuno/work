import pandas as pd
from numpy import *
# import random

df=pd.read_csv(r'C:\Users\lenovo\Desktop\DRUG1n.csv')
date=pd.get_dummies(df)
date=date.values.tolist()

maxage = 0
maxna = 0
maxk = 0
x=[]
for row in date:
    lst = row
    if lst[0] > maxage:
        maxage = lst[0]
    if lst[1] > maxna:
        maxna = lst[1]
    if lst[2] > maxk:
        maxk = lst[2]
for row in date:
    lst=row
    lst[0] = lst[0]/maxage
    lst[1] = lst[1]/maxna
    lst[2] = lst[2]/maxk
    x.append(lst)
print(x[0])

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(array(dataSet)[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        sum = 0
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            sum+=minDist
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(sum)
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

dataMat = mat(x)
Centroids, clustAssing = kMeans(dataMat, 3)
print(Centroids)
