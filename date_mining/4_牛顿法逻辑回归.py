
import csv
from numpy import *


class LR:
    __row = 0
    __col = 0
    # 迭代次数
    __numIterations = 10
    __trainData = []
    __theta = []
    __Y = []
    __Cost = []

    def __init__(self, data):
        self.__row, self.__col = shape(data)
        self.__trainData = data[:, 0:self.__col - 1]
        self.__Y = data[:, self.__col - 1:self.__col]
        self.__col = self.__col - 1
        # 系数的初始值为0  如果是其他值有可能会算不出结果
        self.__theta = mat(zeros((self.__col, 1)))

    # 设置迭代次数
    def setnumIterations(self, numIterations):
        self.__numIterations = numIterations

    # 获取theta
    def getTheta(self):
        return self.__theta

    # 获取cost损失
    def getCost(self):
        return self.__Cost

    # 训练数据模型
    def train(self):
        # 存储损失
        self.__Cost = mat(zeros((self.__numIterations, 1)))
        for i in range(0, self.__numIterations):
            # 更新Theta
            self.__updateTheta(i)

    # 迭代theta
    def __updateTheta(self, i):
        # 获取预测值h(x) 1.0 / (1 + exp(-z))
        h = 1.0 / (1 + exp(-(self.__trainData * self.__theta)))

        # 获取损失
        self.__getCost(i, h)
        # .T 矩阵的转置
        # 一阶导数矩阵算法 (1 / m) * x.T * (h-y)
        J = multiply(1.0 / self.__row,
                     self.__trainData.T * (h - self.__Y))
        # 获取Hession矩阵
        # getA() 矩阵转换为数组
        # diag(x) 生成对角线为x其余为0的矩阵
        # Hession矩阵算法(1 / m) * x.T * U * x   U表示用(h * (1 - h))构成对角，其余为0的矩阵
        H = multiply(1.0 / self.__row, self.__trainData.T *
                     diag(multiply(h, (1 - h)).T.getA()
                          [0]) * self.__trainData)
        # .I 矩阵的逆
        self.__theta = self.__theta - H.I * J

    # 计算损失
    def __getCost(self, i, h):
        l1 = self.__Y.T * log(h)
        l2 = (1 - self.__Y).T * log((1 - h))
        self.__Cost[i, :] = multiply(1.0 / self.__row, sum(-l1 - l2))




filename = r"C:\Users\lenovo\Desktop\telco.csv"
csvHand= open(filename,"r")
readcsv= csv.reader(csvHand)

# 求均值
k=0
for row in readcsv:
    if k==0 :
        arv = [0] * (len(row) - 1)
        count = [0] * (len(row) - 1)
        k=1
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



csvHand= open(filename,"r")
readcsv= csv.reader(csvHand)
buffer= []
k=0
for row in readcsv:
    if k==0 :
        k=1
        continue
    j=0
    for item in row[:-1]:
        if item=="" :

            row[j]=str(arv[j])
        j+=1
    buffer.append(list(map(eval, row)))

lr = LR(mat(buffer))
lr.setnumIterations(100)
lr.train()
# print(lr.getTheta())
# print(lr.getCost())
#
