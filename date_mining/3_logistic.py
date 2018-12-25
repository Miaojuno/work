
import csv
from math import exp

import numpy as np


filename = r"C:\Users\lenovo\Desktop\telco.csv"
csvHand= open(filename,"r")
readcsv= csv.reader(csvHand)

# 求均值
i=0
for row in readcsv:
    if i==0 :
        arv = [0] * (len(row) - 1)
        count = [0] * (len(row) - 1)
        i=1
        continue
    j=0
    for item in row[:-1]:
        if item != "":
            arv[j]+=eval(item)
            count[j]+=1
            j+=1
csvHand.close()

j=0
for item in arv:
    arv[j]/=count[j]
    j+=1


# 转换为两个列表
csvHand= open(filename,"r")
readcsv= csv.reader(csvHand)
buffer= []
buffer2= []
i=0
for row in readcsv:
    if i==0 :
        i=1
        continue
    j=0
    for item in row[:-1]:
        if item=="" :

            row[j]=str(arv[j])
        j+=1
    buffer.append(list(map(eval, row[:-1])))
    buffer2.append(list(map(eval, row[-1])))

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
    step_size = 0.00005
    max = 1000
    while loss > 0.001 and count < max:
        loss = 0
        error = np.zeros(col)
        for i in range(row):
            predict = np.dot(w.T, x[i])
            for j in range(col):
                error[j] += (y[i] - sigmoid(predict)) * x[i][j]
        for j in range(col):
            w[j] = w[j] * (1 - 20 * step_size / row) + step_size * error[j] / row


        for i in range(row):
            predict = np.dot(w.T, x[i])
            error = (1 / (row * col)) * np.power((sigmoid(predict) - y[i]), 2)
            loss += error
        count += 1
        print(count,loss)
    return w,count


# def sigmoid(inx):
#     return 1.0 / (1 + np.exp(-inx));


def gradAscent(dataMatIn, classLabels):
    # 转换为NumPy矩阵数据类型
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels)
    m, n = np.shape(dataMatrix)
    alpha = 0.001  # 向目标移动的步长
    maxCycles = 1000  # 迭代次数
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        # 　矩阵相乘
        h = sigmoid(dataMatrix * weights)  # 列向量的元素个数等于样本个数
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
        loss=abs(sum(abs(error))/(dataMatrix.shape[0]))
        print(loss)
    # getA() Return self as an ndarray object.
    return weights


# t= logRegres.gradAscent(buffer,buffer2)
# print(t)
x=np.array(buffer)
y=np.array(buffer2)
w,count=logistic(x,y)
# print(w)
loss=0
for i,k in enumerate(buffer2):
    h = 1.0 / (1 + np.exp(-np.dot(w.T, x[i])))
    loss+=abs(h-buffer2[i])

print(1-loss/len(buffer2))
#
