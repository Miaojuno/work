
# import csv
# from numpy import *
# from sklearn import linear_model
#
# filename = r"C:\Users\lenovo\Desktop\knn\cancer_X.csv"
#
# csvHand= open(filename,"r")
# readcsv= csv.reader(csvHand)
# buffer= []
# for i,row in enumerate(readcsv):
#     buffer.append(list(map(eval, row[:5])))
#
#
# filename=r"C:\Users\lenovo\Desktop\knn\cancer_y.csv"
#
# csvHand= open(filename,"r")
# readcsv= csv.reader(csvHand)
# buffer2=[]
# for i,row in enumerate(readcsv):
#     buffer2.append(eval(row[0]))
#
#
# logreg = linear_model.LogisticRegression(solver='newton-cg',C=1e5, multi_class='multinomial')
#
# a = logreg.fit(buffer, buffer2)
# print(a.coef_ ) #返回参数的系数
# print(a.predict(buffer))  # 预测类别
# print(a.score(buffer, buffer2))


import csv
from math import exp

import numpy as np

def sigmoid(inX):
    if inX>=0:
        return 1.0/(1+np.exp(-inX))
    else:
        return np.exp(inX)/(1+np.exp(inX))

def logistic(x, y):
    row, col = x.shape
    w = np.ones(col)
    loss = 10
    count = 0
    step_size = 0.0005
    max = 1000
    while loss > 0.001 and count < max:
        loss = 0
        error = np.zeros(col)
        for i in range(row):
            predict = np.dot(w.T, x[i])
            for j in range(col):
                error[j] += (y[i] - sigmoid(predict)) * x[i][j]
        for j in range(col):
            w[j] = w[j] * (1 - 10 * step_size / row) + step_size * error[j] / row


        for i in range(row):
            predict = np.dot(w.T, x[i])
            error = (1 / (row * col)) * np.power((sigmoid(predict) - y[i]), 2)
            loss += error
        count += 1
        print(count,loss)
    return w,count



filename = r"C:\Users\lenovo\Desktop\knn\cancer_X.csv"

csvHand= open(filename,"r")
readcsv= csv.reader(csvHand)
buffer= []
for i,row in enumerate(readcsv):
    buffer.append(list(map(eval, row)))


filename=r"C:\Users\lenovo\Desktop\knn\cancer_y.csv"

csvHand= open(filename,"r")
readcsv= csv.reader(csvHand)
buffer2=[]
for i,row in enumerate(readcsv):
    buffer2.append(eval(row[0]))

bu=[0]*len(buffer2)
weight=[]
for i in range(18):
    for j, row in enumerate(buffer2):
        if buffer2[j]!=i+1:
            bu[j]=0
        if buffer2[j]==i+1:
            bu[j]=1
    x=np.array(buffer)
    y=np.array(bu)
    w,count=logistic(x,y)
    weight.append(w)
weight=np.array(weight)
h=[]
for i,b in enumerate(x):
    pe=[]
    for j in range(18):
        pe.append(1.0 / (1 + np.exp(-np.dot(weight[j], b))))
    h.append(pe.index(max(pe))+1)

print(h)
# x=np.array(buffer)
# y=np.array(buffer2)
# w,count=logistic(x,y)
