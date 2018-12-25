import csv
import numpy as np
from keras.utils import to_categorical

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))
def sigmoidDerivationx(y):
    return y * (1 - y)

datas = np.array(list(csv.reader(open(r"C:\Users\lenovo\Desktop\X_data.csv","r"))))
datas = datas.astype(float)
# xb=np.ones((5000,1))
# data=np.column_stack((data,xb))

labels = np.array(list(csv.reader(open(r"C:\Users\lenovo\Desktop\y_label.csv","r"))))
labels = labels.astype(float)

def BackPropagate(self, x, y, lr):
    '''
    the implementation of BP algorithms on one slide of sample

    @param x, y: array, input and output of the data sample
    @param lr: float, the learning rate of gradient decent iteration
    '''

    # dependent packages
    import numpy as np

    # get current network output
    self.Pred(x)

    # calculate the gradient based on output
    o_grid = np.zeros(self.o_n)
    for j in range(self.o_n):
        o_grid[j] = (y[j] - self.o_v[j]) * self.afd(self.o_v[j])

    h_grid = np.zeros(self.h_n)
    for h in range(self.h_n):
        for j in range(self.o_n):
            h_grid[h] += self.ho_w[h][j] * o_grid[j]
        h_grid[h] = h_grid[h] * self.afd(self.h_v[h])

    # updating the parameter
    for h in range(self.h_n):
        for j in range(self.o_n):
            self.ho_w[h][j] += lr * o_grid[j] * self.h_v[h]

    for i in range(self.i_n):
        for h in range(self.h_n):
            self.ih_w[i][h] += lr * h_grid[h] * self.i_v[i]

    for j in range(self.o_n):
        self.o_t[j] -= lr * o_grid[j]

    for h in range(self.h_n):
        self.h_t[h] -= lr * h_grid[h]










#
# w1=np.random.rand(25,400)
# w2=np.random.rand(10,25)
#
# # w1=np.zeros((25,400))
# # w2=np.zeros((10,25))
#
# # b1=np.zeros((5000,1))
# # b2=np.zeros((5000,1))
# b1=0.1
# b2=0.1
# for time in range(10):
#     for i in range(5000):
#         data=datas[i]
#         label=labels[i]
#         z1=np.matmul(data,w1.T)+b1
#         a1=sigmoid(z1)
#         z2=np.matmul(a1,w2.T)+b2
#         a2=sigmoid(z2)
#
#         alpha = 0.1
#         numIter = 5  # 迭代次数
#         for n in range(numIter):
#                 delta2 = np.multiply(-(label-a2), np.multiply(a2, 1-a2))
#                 # for i in range(len(w2)):
#                 #     print(w2[i] - alpha * delta2[i] * a1)
#                 #计算非最后一层的错误
#                 # print(delta2)
#                 delta1 = np.multiply(np.dot( delta2,np.array(w2)), np.multiply(a1, 1 - a1))
#                 # print(delta1)
#                 # for i in range(len(w1)):
#                     # print(w1[i] - alpha * delta1[i] * np.array(x))
#                 #更新权重
#                 for i in range(len(w2)):
#                     w2[i] = w2[i] - alpha * delta2[i] * a1
#                 for i in range(len(w1)):
#                     w1[i] = w1[i] - alpha * delta1[i] * np.array(data)
#                 #继续前向传播，算出误差值
#                 z1 = np.dot(w1, data) + b1
#                 a1 = sigmoid(z1)
#                 z2 = np.dot(w2, a1) + b2
#                 a2 = sigmoid(z2)
#                 # print(str(n) + "  error1:" + str(y[0] - a2[0]) + ", error2:" +str(y[1] - a2[1]))
#
#
# y1=sigmoid(np.matmul(datas,w1.T)+b1)
# y2=sigmoid(np.matmul(y1,w2.T)+b2)
#
# print(y2)
# acc_count=0
# for i,line in enumerate(list(y2)):
#     max = 0
#     max_x = -100
#     for j,x in enumerate(line):
#         if x>max_x:
#             max=j
#             max_x=x
#     if max+1 == labels[i]:
#         acc_count+=1
# print("精确率:",acc_count/(i+1))